import os
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from utils.dice_score import *
from utils.data_loading import BasicDataset


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
	net.eval()
	num_val_batches = len(dataloader)
	dice_score = 0

	# iterate over the validation set
	with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
		for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
			image, mask_true = batch['image'], batch['mask']

			# move images and labels to correct device and type
			image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
			mask_true = mask_true.to(device=device, dtype=torch.long)

			# predict the mask
			mask_pred = net(image)

			if net.n_classes == 1:
				assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
				mask_pred = (F.sigmoid(mask_pred.squeeze(1)) > 0.5).float()
				# compute the Dice score
				dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
			else:
				assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
				# convert to one-hot format
				mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
				mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
				# compute the Dice score, ignoring background
				dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

	net.train()
	return dice_score / max(num_val_batches, 1)


@torch.inference_mode()
def evaluate_metrics(net, dataloader, device, amp, criterion=None):
	net.eval()
	num_batches = len(dataloader)
	n_classes = net.n_classes

	# Initialize counters for each class
	if n_classes > 1:
		tp = torch.zeros(n_classes, device=device)
		fp = torch.zeros(n_classes, device=device)
		fn = torch.zeros(n_classes, device=device)
	else:
		tp = torch.tensor(0, device=device, dtype=torch.float)
		fp = torch.tensor(0, device=device, dtype=torch.float)
		fn = torch.tensor(0, device=device, dtype=torch.float)
	total_loss = 0

	# Iterate over the validation set
	with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
		for batch in tqdm(dataloader, total=num_batches, desc='Evaluation', unit='batch', leave=False):
			image, mask_true = batch['image'], batch['mask']

			# Move images and labels to correct device and type
			image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
			mask_true = mask_true.to(device=device, dtype=torch.long)

			# Predict the mask
			mask_pred = net(image)

			# Calculate loss if criterion is provided
			if criterion:
				if n_classes == 1:
					loss = criterion(mask_pred.squeeze(1), mask_true.float())
					loss += dice_loss(F.sigmoid(mask_pred.squeeze(1)), mask_true.float(), multiclass=False)
				else:
					loss = criterion(mask_pred, mask_true)
					loss += dice_loss(
						F.softmax(mask_pred, dim=1).float(),
						F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float(),
						multiclass=True
					)
				total_loss += loss.item()

			# Calculate metrics
			if n_classes == 1:
				assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
				mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

				# Binary case
				tp += ((mask_pred == 1) & (mask_true == 1)).sum().float()
				fp += ((mask_pred == 1) & (mask_true == 0)).sum().float()
				fn += ((mask_pred == 0) & (mask_true == 1)).sum().float()
			else:
				assert mask_true.min() >= 0 and mask_true.max() < n_classes, 'True mask indices should be in [0, n_classes]'
				mask_pred_argmax = mask_pred.argmax(dim=1)

				# Calculate TP, FP, FN for each class
				for c in range(n_classes):
					tp[c] += ((mask_pred_argmax == c) & (mask_true == c)).sum().float()
					fp[c] += ((mask_pred_argmax == c) & (mask_true != c)).sum().float()
					fn[c] += ((mask_pred_argmax != c) & (mask_true == c)).sum().float()

	# Calculate precision, recall, IoU for each class
	smooth = 1e-7
	precision = tp / (tp + fp + smooth)
	recall = tp / (tp + fn + smooth)
	iou = tp / (tp + fp + fn + smooth)
	dice_score = 2 * tp / (2 * tp + fp + fn + smooth)

	# Calculate average metrics
	avg_loss = total_loss / num_batches if criterion is not None else None

	# Prepare results dictionary
	results = {
		'precision': precision,
		'recall': recall,
		'dice_scores': dice_score,
		'avg_dice': dice_score.mean(),
		'iou': iou,
		'miou': iou.mean(),
		'loss': avg_loss
	}

	net.train()
	return results


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

	dir_img = "data/patch256/test/images"
	dir_mask = "data/patch256/test/masks"
	model_paths = [
		"checkpoints_256_1e-6/checkpoint_epoch41.pth",
		"checkpoints_256_1e-6/checkpoint_epoch42.pth",
		"checkpoints_256_1e-6/checkpoint_epoch43.pth",
		"checkpoints_256_1e-6/checkpoint_epoch44.pth",
		"checkpoints_256_1e-6/checkpoint_epoch45.pth",
		"checkpoints_256_1e-6/checkpoint_epoch46.pth",
		"checkpoints_256_1e-6/checkpoint_epoch47.pth",
		"checkpoints_256_1e-6/checkpoint_epoch48.pth",
		"checkpoints_256_1e-6/checkpoint_epoch49.pth",
		"checkpoints_256_1e-6/checkpoint_epoch50.pth",
	]
	img_scale = 1.0
	batch_size = 256

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model_list = list()

	for path in model_paths:
		model = UNet(n_channels=3, n_classes=3)
		model = model.to(memory_format=torch.channels_last)
		state_dict = torch.load(path, map_location=device)
		state_dict.pop("mask_values")
		model.load_state_dict(state_dict)
		model.to(device=device)
		model_list.append(model)

	dataset = BasicDataset(dir_img, dir_mask, img_scale)
	test_loader = DataLoader(dataset, shuffle=False, drop_last=True, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

	for i, model in enumerate(model_list):
		val_score = evaluate(model, test_loader, device, amp=True)
		logging.info(f"Model: {model_paths[i]}, validation Dice score: {val_score}")
