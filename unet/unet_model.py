""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
	def __init__(self, n_channels, n_classes, bilinear=False):
		super().__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		self.inc = (DoubleConv(n_channels, 64))
		self.down1 = (Down(64, 128))
		self.down2 = (Down(128, 256))
		self.down3 = (Down(256, 512))
		factor = 2 if bilinear else 1
		self.down4 = (Down(512, 1024 // factor))

		self.up1 = (Up(1024, 512 // factor, bilinear))
		self.up2 = (Up(512, 256 // factor, bilinear))
		self.up3 = (Up(256, 128 // factor, bilinear))
		self.up4 = (Up(128, 64, bilinear))
		self.outc = (OutConv(64, n_classes))

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		logits = self.outc(x)
		return logits


class UNetSE1(UNet):
	"""SE block before every downscaling"""

	def __init__(self, n_channels, n_classes, bilinear=False):
		super().__init__(n_channels, n_classes, bilinear)
		self.SE1 = SEBlock(64)
		self.SE2 = SEBlock(128)
		self.SE3 = SEBlock(256)
		self.SE4 = SEBlock(512)

	def forward(self, x):
		x1 = self.inc(x)
		x1 = self.SE1(x1)
		x2 = self.down1(x1)
		x2 = self.SE2(x2)
		x3 = self.down2(x2)
		x3 = self.SE3(x3)
		x4 = self.down3(x3)
		x4 = self.SE4(x4)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		logits = self.outc(x)
		return logits


class UNetSE2(UNet):
	"""SE block before every upscaling"""

	def __init__(self, n_channels, n_classes, bilinear=False):
		super().__init__(n_channels, n_classes, bilinear)
		factor = 2 if self.bilinear else 1
		self.SE1 = SEBlock(1024 // factor)
		self.SE2 = SEBlock(512 // factor)
		self.SE3 = SEBlock(256 // factor)
		self.SE4 = SEBlock(128 // factor)

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x5 = self.SE1(x5)
		x = self.up1(x5, x4)
		x = self.SE2(x)
		x = self.up2(x, x3)
		x = self.SE3(x)
		x = self.up3(x, x2)
		x = self.SE4(x)
		x = self.up4(x, x1)
		logits = self.outc(x)
		return logits


class UNetSE3(UNetSE2):
	"""SE block before the first upscaling"""

	def __init__(self, n_channels, n_classes, bilinear=False):
		super().__init__(n_channels, n_classes, bilinear)

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x5 = self.SE1(x5)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		logits = self.outc(x)
		return logits


class UNetSE4(UNet):
	"""SE block before out convolution"""

	def __init__(self, n_channels, n_classes, bilinear=False):
		super().__init__(n_channels, n_classes, bilinear)
		self.SE = SEBlock(64)

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.SE(x)
		logits = self.outc(x)
		return logits
