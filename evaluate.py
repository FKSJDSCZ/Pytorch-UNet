import os
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from prettytable import PrettyTable

from unet import UNet
from utils.dice_score import *
from utils.data_loading import ExperimentDataset


@torch.inference_mode()
def evaluate_metrics(net, dataloader, device, amp, criterion):
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
	dice_score = 0
	total_loss = torch.tensor(0, device=device, dtype=torch.float)

	# Iterate over the validation set
	with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
		for batch in tqdm(dataloader, total=num_batches, desc='Evaluation', unit='batch', leave=False):
			image, mask_true = batch['image'], batch['mask']

			# Move images and labels to correct device and type
			image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
			mask_true = mask_true.to(device=device, dtype=torch.long)

			# Predict the mask
			mask_pred = net(image)

			# Calculate metrics
			if n_classes == 1:
				assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
				mask_pred = mask_pred.squeeze(1)
				total_loss += criterion(mask_pred, mask_true.float())

				mask_pred = F.sigmoid(mask_pred)
				total_loss += dice_loss(mask_pred, mask_true.float(), multiclass=False)

				mask_pred = (mask_pred > 0.5).float()
				dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

				# Binary case
				tp += ((mask_pred == 1) & (mask_true == 1)).sum().float()
				fp += ((mask_pred == 1) & (mask_true == 0)).sum().float()
				fn += ((mask_pred == 0) & (mask_true == 1)).sum().float()
			else:
				assert mask_true.min() >= 0 and mask_true.max() < n_classes, 'True mask indices should be in [0, n_classes]'
				total_loss += criterion(mask_pred, mask_true)
				total_loss += dice_loss(
					F.softmax(mask_pred, dim=1).float(),
					F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
					multiclass=True
				)
				mask_pred = mask_pred.argmax(dim=1)

				# Convert to one-hot format, then compute the Dice score ignoring background
				dice_score += multiclass_dice_coeff(
					F.one_hot(mask_pred, net.n_classes).permute(0, 3, 1, 2).float()[:, 1:],
					F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()[:, 1:],
					reduce_batch_first=False
				)

				# Calculate TP, FP, FN for each class
				for c in range(n_classes):
					tp[c] += ((mask_pred == c) & (mask_true == c)).sum().float()
					fp[c] += ((mask_pred == c) & (mask_true != c)).sum().float()
					fn[c] += ((mask_pred != c) & (mask_true == c)).sum().float()

	# Calculate precision, recall, IoU for each class
	smooth = 1e-7
	precision = tp / (tp + fp + smooth)
	recall = tp / (tp + fn + smooth)
	iou = tp / (tp + fp + fn + smooth)
	f1_score = 2 * precision * recall / (precision + recall + smooth)
	avg_loss = total_loss / num_batches

	# Prepare results dictionary
	results = {
		'precision': precision,
		'recall': recall,
		'iou': iou,
		'f1_score': f1_score,
		'dice_score': dice_score / max(num_batches, 1),
		'miou': iou.mean(),
		'loss': avg_loss.item(),
	}

	net.train()
	return results


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

	dir_img = "data/patch256/test/images"
	dir_mask = "data/patch256/test/masks"
	model_paths = [
		"checkpoint_202505101110_15yc5hp8/checkpoint_epoch205.pth",
		"checkpoint_202505101110_myoq5iwa/checkpoint_epoch205.pth",
	]
	img_scale = 1.0
	batch_size = 16

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model_list = list()

	for path in model_paths:
		model = UNet(3, 4)
		model = model.to(memory_format=torch.channels_last)
		state_dict = torch.load(path, map_location=device)
		state_dict.pop("mask_values")
		model.load_state_dict(state_dict)
		model.to(device=device)
		model_list.append(model)

	dataset = ExperimentDataset(dir_img, dir_mask, img_scale)
	test_loader = DataLoader(dataset, shuffle=False, drop_last=True, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

	for model_idx, model in enumerate(model_list):
		test_metrics = evaluate_metrics(model, test_loader, device, True, torch.nn.CrossEntropyLoss() if model.n_classes > 1 else torch.nn.BCEWithLogitsLoss())
		logging.info(f"Model {model_paths[model_idx]}:\n"
					 f"\t\tLoss: {test_metrics['loss']},\n"
					 f"\t\tDice: {test_metrics['dice_score']},\n"
					 f"\t\tMIoU: {test_metrics['miou']},\n"
					 f"\t\tMacro-avg P: {test_metrics['precision'].mean()},\n"
					 f"\t\tMacro-avg R: {test_metrics['recall'].mean()},\n"
					 f"\t\tMacro-avg F1: {test_metrics['f1_score'].mean()},")
		test_table = PrettyTable(["class\\score", "P", "R", "IoU", "F1"])
		for i in range(model.n_classes):
			test_table.add_row(
				[
					i,
					test_metrics['precision'][i].item(),
					test_metrics['recall'][i].item(),
					test_metrics['iou'][i].item(),
					test_metrics['f1_score'][i].item(),
				]
			)
		print(test_table)
