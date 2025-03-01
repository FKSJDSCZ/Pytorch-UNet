import os
import sys
import numpy as np
from PIL import Image
import torch
import imgviz
from tqdm import tqdm
from typing import Tuple

Image.MAX_IMAGE_PIXELS = None


def lblsave(filename, lbl):
	if os.path.splitext(filename)[1] != ".png":
		filename += ".png"
	# Assume label ranses [-1, 254] for int32,
	# and [0, 255] for uint8 as VOC.
	if lbl.min() >= -1 and lbl.max() < 255:
		lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode="P")
		colormap = imgviz.label_colormap()
		lbl_pil.putpalette(colormap.flatten())
		lbl_pil.save(filename)
	else:
		raise ValueError(
			"[%s] Cannot save the pixel-wise class label as PNG. "
			"Please consider using the .npy format." % filename
		)


class LargeImagePredictor:
	def __init__(
			self,
			model_path: str,
			patch_size: int = 512,
			padding: int = 32,
			device: str = None
	):
		self.patch_size = patch_size
		self.padding = padding
		self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

		self.model = self._load_model(model_path)
		self.model.to(self.device)
		self.model.eval()

	def _load_model(self, model_path):
		state_dict = torch.load(model_path, map_location=self.device)
		self.mask_values = state_dict.pop('mask_values')

		from unet import UNet  # Import here to avoid circular import
		model = UNet(n_channels=3, n_classes=4)
		model.load_state_dict(state_dict)

		print("Model loaded")
		return model

	def _pad_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
		h, w = image.shape[:2]

		new_h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
		new_w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size

		pad_h = new_h - h
		pad_w = new_w - w

		padded = np.pad(
			image,
			((self.padding, self.padding + pad_h), (self.padding, self.padding + pad_w), (0, 0)),
			mode='constant',
			constant_values=(255, 255)
		)

		print(f"Padding image finished. Original shape: {image.shape}, padded shape: {padded.shape}")
		return padded, (pad_h, pad_w)

	def _extract_patches(self, image: np.ndarray) -> Tuple[list, list]:
		h, w = image.shape[:2]
		patches = []
		coordinates = []

		total = ((h - 2 * self.padding) // self.patch_size) * ((w - 2 * self.padding) // self.patch_size)
		print(f"Cutting image to {total} patches")

		for y in range(0, h - 2 * self.padding, self.patch_size):
			for x in range(0, w - 2 * self.padding, self.patch_size):
				patch = image[y:y + self.patch_size + 2 * self.padding, x:x + self.patch_size + 2 * self.padding]
				patches.append(patch)
				coordinates.append((y, x))

		print("Extract patches finished")
		return patches, coordinates

	def _predict_patch(self, patch: np.ndarray) -> np.ndarray:
		img = torch.from_numpy(patch.transpose(2, 0, 1))
		img = img.unsqueeze(0).float().to(self.device) / 255.0

		with torch.no_grad():
			output = self.model(img)
			if self.model.n_classes > 1:
				mask = output.argmax(dim=1)
			else:
				mask = torch.sigmoid(output) > 0.5

		return mask[0].cpu().numpy()

	def _assemble_predictions(
			self,
			predictions: list,
			coordinates: list,
			image_size: Tuple[int, int]
	) -> np.ndarray:
		h, w = image_size
		full_mask = np.zeros((h - 2 * self.padding, w - 2 * self.padding))

		for pred, (y, x) in zip(predictions, coordinates):
			if self.padding:
				pred = pred[self.padding:-self.padding, self.padding:-self.padding]
			full_mask[y:y + self.patch_size, x:x + self.patch_size] = pred

		print("Assemble finished")
		return full_mask

	def predict(self, image_path: str, vizout_path: str = None, maskout_path: str = None) -> np.ndarray:
		image = np.array(Image.open(image_path))
		padded_image, pad_sizes = self._pad_image(image)
		patches, coordinates = self._extract_patches(padded_image)

		predictions = []
		for patch in tqdm(patches, desc="Predicting", unit="patch"):
			predictions.append(self._predict_patch(patch))
		print("Predict finished")

		full_mask = self._assemble_predictions(predictions, coordinates, (padded_image.shape[0], padded_image.shape[1]))
		full_mask = full_mask[:image.shape[0], :image.shape[1]].astype(np.uint8)

		print("Start saving results")
		if vizout_path:
			mask_viz = imgviz.label2rgb(
				full_mask,
				imgviz.asgray(image),
				0.5,
				["background", "severe", "medium", "slight"]
			)
			Image.fromarray(mask_viz).save(vizout_path)

		if maskout_path:
			lblsave(maskout_path, full_mask)

		# Image.fromarray(patches[0]).save("data/test-patch0.png")
		# Image.fromarray(patches[1]).save("data/test-patch1.png")
		# Image.fromarray(patches[2]).save("data/test-patch2.png")
		# Image.fromarray(patches[3]).save("data/test-patch3.png")
		# lblsave("data/test-mask0.png", predictions[0])
		# lblsave("data/test-mask1.png", predictions[1])
		# lblsave("data/test-mask2.png", predictions[2])
		# lblsave("data/test-mask3.png", predictions[3])

		return full_mask


if __name__ == "__main__":
	predictor = LargeImagePredictor(
		model_path="checkpoints_256_1e-6/checkpoint_epoch3.pth",
		patch_size=256,
		padding=0
	)

	mask = predictor.predict(
		image_path=f"data/{sys.argv[1]}.png",
		# vizout_path=f"data/{sys.argv[1]}-pred.png",
		maskout_path=f"data/{sys.argv[1]}-mask.png"
	)
