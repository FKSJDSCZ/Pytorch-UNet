import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm


def load_image(filename):
	ext = splitext(filename)[1]
	if ext == '.npy':
		return Image.fromarray(np.load(filename))
	elif ext in ['.pt', '.pth']:
		return Image.fromarray(torch.load(filename).numpy())
	else:
		return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
	mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
	mask = np.asarray(load_image(mask_file))
	if mask.ndim == 2:
		return np.unique(mask)
	elif mask.ndim == 3:
		mask = mask.reshape(-1, mask.shape[-1])
		return np.unique(mask, axis=0)
	else:
		raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
	def __init__(self,
	             images_dir: str,
	             mask_dir: str,
	             scale: float = 1.0,
	             mask_suffix: str = '',
	             use_weighted_sampling: bool = False,
	             priority_list: list = None):
		self.images_dir = Path(images_dir)
		self.mask_dir = Path(mask_dir)
		assert 0 < scale <= 1, 'Scale must be between 0 and 1'
		self.scale = scale
		self.mask_suffix = mask_suffix
		self.use_weighted_sampling = use_weighted_sampling
		self.priority_list = priority_list

		self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
		if not self.ids:
			raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

		logging.info(f'Creating dataset with {len(self.ids)} examples')
		logging.info('Scanning mask files to determine unique values')
		with Pool() as p:
			unique = list(tqdm(
				p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
				total=len(self.ids)
			))

		self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
		logging.info(f'Unique mask values: {self.mask_values}')

		# Initialize sample weights if weighted sampling is enabled
		self.sample_weights = None
		if self.use_weighted_sampling and self.priority_list:
			self._calculate_sample_weights()

	def _calculate_sample_weights(self):
		logging.info('Calculating sample weights based on priority list')

		# Verify that priority list is valid
		n_classes = len(self.mask_values)
		assert all(0 <= c < n_classes for c in self.priority_list), f'Priority list contains invalid class indices. Must be in range [0, {n_classes - 1}]'
		assert len(self.priority_list) == n_classes, f'Priority list must have the same length as number of classes ({n_classes})'

		# Check which classes are present in each mask
		group_indexes = []
		for idx in tqdm(self.ids, desc='Analyzing masks for class presence'):
			mask_file = list(self.mask_dir.glob(idx + self.mask_suffix + '.*'))[0]
			mask = np.asarray(load_image(mask_file))

			# Determine which group this sample belongs to
			for class_idx in self.priority_list:
				if class_idx in np.unique(mask):
					assigned_group = class_idx
					break
			else:
				assigned_group = self.priority_list[-1]

			group_indexes.append(assigned_group)

		# Count samples in each group
		group_indexes = np.array(group_indexes)
		group_counts = {idx: group_indexes[group_indexes == idx].size for idx in self.priority_list}
		logging.info(f'Sample distribution across groups: {group_counts}')

		# Calculate weights as inverse of group size
		group_weights = {}
		total_samples = len(group_indexes)
		for group, count in group_counts.items():
			# Calculate inverse proportion (total / count) as weight
			group_weights[group] = total_samples / count if count > 0 else 0

		logging.info(f'Group weights: {group_weights}')

		# Assign weights to each sample
		self.sample_weights = [group_weights[group] for group in group_indexes]

	def get_sampler(self):
		if not self.use_weighted_sampling or self.sample_weights is None:
			return None

		# Create and return the sampler
		return WeightedRandomSampler(
			weights=self.sample_weights,
			num_samples=len(self.sample_weights),
			replacement=True
		)

	def __len__(self):
		return len(self.ids)

	@staticmethod
	def preprocess(mask_values, pil_img, scale, is_mask):
		w, h = pil_img.size
		newW, newH = int(scale * w), int(scale * h)
		assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
		pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
		img = np.asarray(pil_img)

		if is_mask:
			mask = np.zeros((newH, newW), dtype=np.int64)
			for i, v in enumerate(mask_values):
				if img.ndim == 2:
					mask[img == v] = i
				else:
					mask[(img == v).all(-1)] = i

			return mask

		else:
			if img.ndim == 2:
				img = img[np.newaxis, ...]
			else:
				img = img.transpose((2, 0, 1))

			if (img > 1).any():
				img = img / 255.0

			return img

	def __getitem__(self, idx):
		name = self.ids[idx]
		mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
		img_file = list(self.images_dir.glob(name + '.*'))

		assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
		assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
		mask = load_image(mask_file[0])
		img = load_image(img_file[0])

		assert img.size == mask.size, \
			f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

		img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
		mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

		return {
			'image': torch.as_tensor(img.copy()).float().contiguous(),
			'mask': torch.as_tensor(mask.copy()).long().contiguous()
		}


class CarvanaDataset(BasicDataset):
	def __init__(self, images_dir, mask_dir, scale=1, use_weighted_sampling=False, priority_list=None):
		super().__init__(
			images_dir,
			mask_dir,
			scale,
			mask_suffix='_mask',
			use_weighted_sampling=use_weighted_sampling,
			priority_list=priority_list
		)
