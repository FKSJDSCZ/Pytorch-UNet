import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from datetime import datetime
from pathlib import Path
from torch import optim
from torch.nn.functional import bilinear
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from prettytable import PrettyTable
from wandb.util import downsample

import wandb
from evaluate import evaluate_metrics
from unet import UNet
from utils.data_loading import ExperimentDataset
from utils.dice_score import dice_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_image_dir = "data/patch256/train/images"
train_mask_dir = "data/patch256/train/masks"
val_image_dir = "data/patch256/val/images"
val_mask_dir = "data/patch256/val/masks"


def train_model(model, config):
	down_sample: float = config['down_sample']
	dataset_name: str = config['dataset_name']
	patch_size: int = config['patch_size']
	epochs: int = config['epochs']
	batch_size: int = config['batch_size']
	learning_rate: float = config['learning_rate']
	weight_decay: float = config['weight_decay']
	momentum: float = config['momentum']
	gradient_clipping: float = config['gradient_clipping']
	save_checkpoint: bool = config['save_checkpoint']
	img_scale: float = config['img_scale']
	amp: bool = config['amp']
	use_weighted_sampling: bool = config['use_weighted_sampling']
	priority_list: list[int] = config['priority_list']
	device: torch.device = config['device']
	config['device'] = device.type

	# 1. Create dataset
	train_dataset = ExperimentDataset(
		train_image_dir,
		train_mask_dir,
		img_scale,
		use_weighted_sampling=use_weighted_sampling,
		priority_list=priority_list
	)

	val_dataset = ExperimentDataset(
		val_image_dir,
		val_mask_dir,
		img_scale,
	)

	# 2. Create data loaders. Use weighted sampler if enabled
	loader_args = dict(batch_size=batch_size, num_workers=min(os.cpu_count(), 16), pin_memory=True)
	if use_weighted_sampling and priority_list:
		logging.info('Using weighted sampling with priority list: {}'.format(priority_list))
		sampler = train_dataset.get_sampler()
		if sampler:
			logging.info('Weighted sampler created successfully')
			train_loader = DataLoader(train_dataset, sampler=sampler, **loader_args)
		else:
			logging.warning('Failed to create weighted sampler, falling back to random sampling')
			train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
	else:
		train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)

	val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

	# (Initialize logging)
	run_id = wandb.util.generate_id()
	checkpoint_dir = f"checkpoint_{datetime.now().strftime('%Y%m%d%H%M%S')}_{run_id}"
	experiment = wandb.init(
		config=config,
		project=config['project'],
		name=f"{model.__class__.__name__}-{'' if down_sample == 1 else f'{down_sample}x'}{dataset_name}-{model.n_classes}c-{patch_size}p",
		resume='allow',
		anonymous='allow',
		id=run_id
	)

	logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Weighted Sampling: {use_weighted_sampling}
        Priority List:   {priority_list if priority_list else "None"}
    ''')

	# 3. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, cooldown=5, min_lr=1e-8)  # goal: maximize Dice score
	# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
	# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, epochs, 1, 1e-7)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-7)
	grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
	criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
	global_step = 0

	# 4. Begin training
	for epoch in range(1, epochs + 1):
		model.train()
		with tqdm(total=len(train_dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
			for batch in train_loader:
				images, true_masks = batch['image'], batch['mask']

				assert images.shape[1] == model.n_channels, \
					f'Network has been defined with {model.n_channels} input channels, ' \
					f'but loaded images have {images.shape[1]} channels. Please check that ' \
					'the images are loaded correctly.'

				images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
				true_masks = true_masks.to(device=device, dtype=torch.long)

				with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
					masks_pred = model(images)
					if model.n_classes == 1:
						loss = criterion(masks_pred.squeeze(1), true_masks.float())
						loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
					else:
						loss = criterion(masks_pred, true_masks)
						loss += dice_loss(
							F.softmax(masks_pred, dim=1).float(),
							F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
							multiclass=True
						)

				optimizer.zero_grad(set_to_none=True)
				grad_scaler.scale(loss).backward()
				grad_scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
				grad_scaler.step(optimizer)
				grad_scaler.update()

				pbar.update(images.shape[0])
				global_step += 1
				experiment.log({
					'train loss': loss.item(),
					'step': global_step,
					'epoch': epoch
				})
				pbar.set_postfix(**{'loss (batch)': loss.item()})

		# Final evaluation of the epoch
		train_metrics = evaluate_metrics(model, train_loader, device, amp, criterion)
		val_metrics = evaluate_metrics(model, val_loader, device, amp, criterion)
		# scheduler.step(val_metrics['dice_score'])
		scheduler.step()

		log_data = {
			'learning rate': optimizer.param_groups[0]['lr'],
			'epoch': epoch,
			'step': global_step,
			'train/dice': train_metrics['dice_score'],
			'train/miou': train_metrics['miou'],
			'train/loss': train_metrics['loss'],
		}

		for c in range(model.n_classes):
			log_data[f'train/precision_class_{c}'] = train_metrics['precision'][c].item()
			log_data[f'train/recall_class_{c}'] = train_metrics['recall'][c].item()
			log_data[f'train/iou_class_{c}'] = train_metrics['iou'][c].item()
			log_data[f'train/f1_class_{c}'] = train_metrics['f1_score'][c].item()

		log_data.update({
			'val/dice': val_metrics['dice_score'],
			'val/miou': val_metrics['miou'],
			'val/loss': val_metrics['loss'],
		})

		for c in range(model.n_classes):
			log_data[f'val/precision_class_{c}'] = val_metrics['precision'][c].item()
			log_data[f'val/recall_class_{c}'] = val_metrics['recall'][c].item()
			log_data[f'val/iou_class_{c}'] = val_metrics['iou'][c].item()
			log_data[f'val/f1_class_{c}'] = val_metrics['f1_score'][c].item()

		experiment.log(log_data)

		logging.info(f"Epoch {epoch}, Dice: {train_metrics['dice_score']}, MIoU: {train_metrics['miou']}")
		train_table = PrettyTable(["class\\score", "P", "R", "IoU", "F1"])
		for i in range(model.n_classes):
			train_table.add_row(
				[
					i,
					train_metrics['precision'][i].item(),
					train_metrics['recall'][i].item(),
					train_metrics['iou'][i].item(),
					train_metrics['f1_score'][i].item(),
				]
			)
		print(train_table)

		logging.info(f"Epoch {epoch}, Dice: {val_metrics['dice_score']}, MIoU: {val_metrics['miou']}")
		val_table = PrettyTable(["class\\score", "P", "R", "IoU", "F1"])
		for i in range(model.n_classes):
			val_table.add_row(
				[
					i,
					val_metrics['precision'][i].item(),
					val_metrics['recall'][i].item(),
					val_metrics['iou'][i].item(),
					val_metrics['f1_score'][i].item(),
				]
			)
		print(val_table)

		if save_checkpoint and epoch % 5 == 0:
			Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
			state_dict = model.state_dict()
			state_dict['mask_values'] = train_dataset.mask_values
			torch.save(state_dict, str(Path(checkpoint_dir) / 'checkpoint_epoch{}.pth'.format(epoch)))
			logging.info(f'Checkpoint {epoch} saved!')


if __name__ == '__main__':
	# channels=3 for RGB images
	# classes is the number of probabilities you want to get per pixel
	config = dict(
		project='U-Net',

		channels=3,
		classes=7,
		down_sample=4,
		dataset_name='type',
		patch_size=256,

		epochs=300,
		batch_size=128,
		learning_rate=1e-4,
		weight_decay=1e-8,
		momentum=0.999,
		gradient_clipping=1.0,

		preload='',
		save_checkpoint=True,
		img_scale=1.0,
		amp=True,
		bilinear=False,
		use_weighted_sampling=True,
		priority_list=[6, 5, 4, 3, 1, 2, 0],
		# priority_list=[3, 1, 2, 0],

		device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	)

	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
	logging.info(f"Using device {config['device']}")

	if config['use_weighted_sampling'] and len(config['priority_list']) != config['classes']:
		logging.warning(f"Priority list length ({len(config['priority_list'])}) must match number of classes ({config['classes']})")
		logging.warning("Disabling weighted sampling")
		config['use_weighted_sampling'] = False

	model = UNet(config['channels'], config['classes'], config['bilinear'])
	model = model.to(memory_format=torch.channels_last)

	logging.info(
		f'Network: {model.__class__.__name__}\n'
		f'\t{model.n_channels} input channels\n'
		f'\t{model.n_classes} output channels (classes)\n'
		f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
	)

	if config['preload']:
		state_dict = torch.load(config['preload'], map_location=config['device'])
		del state_dict['mask_values']
		model.load_state_dict(state_dict)
		logging.info(f"Model loaded from {config['preload']}")

	model.to(device=config['device'])
	train_model(model, config)
