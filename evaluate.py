import os
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from utils.dice_score import multiclass_dice_coeff, dice_coeff
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
				assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
				# convert to one-hot format
				mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
				mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
				# compute the Dice score, ignoring background
				dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

	net.train()
	return dice_score / max(num_val_batches, 1)


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

	dir_img = "data/patch256/test/images"
	dir_mask = "data/patch256/test/masks/severe"
	model_paths = [
		"checkpoints_256_1e-6/checkpoint_epoch10.pth",
		"checkpoints_256_1e-6/checkpoint_epoch11.pth",
		"checkpoints_256_1e-6/checkpoint_epoch12.pth",
		"checkpoints_256_1e-6/checkpoint_epoch13.pth",
		"checkpoints_256_1e-6/checkpoint_epoch14.pth",
		"checkpoints_256_1e-6/checkpoint_epoch15.pth",
		"checkpoints_256_1e-6/checkpoint_epoch16.pth",
		"checkpoints_256_1e-6/checkpoint_epoch17.pth",
		"checkpoints_256_1e-6/checkpoint_epoch18.pth",
	]
	img_scale = 1.0
	batch_size = 64

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model_list = list()

	for path in model_paths:
		model = UNet(n_channels=3, n_classes=1)
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
