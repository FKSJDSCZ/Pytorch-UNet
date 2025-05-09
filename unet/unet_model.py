""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
	def __init__(self, n_channels, n_classes, using_se=False, bilinear=False):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear
		self.using_se = using_se if using_se else ''

		self.inc = (DoubleConv(n_channels, 64))
		if 'E' in self.using_se:
			self.encode_SE1 = SEBlock(64)
			self.encode_SE2 = SEBlock(128)
			self.encode_SE3 = SEBlock(256)
			self.encode_SE4 = SEBlock(512)
		self.down1 = (Down(64, 128))
		self.down2 = (Down(128, 256))
		self.down3 = (Down(256, 512))
		factor = 2 if bilinear else 1
		self.down4 = (Down(512, 1024 // factor))
		if 'D' in self.using_se:
			self.decode_SE1 = SEBlock(1024 // factor)
			self.decode_SE2 = SEBlock(512 // factor)
			self.decode_SE3 = SEBlock(256 // factor)
			self.decode_SE4 = SEBlock(128 // factor)
		self.up1 = (Up(1024, 512 // factor, bilinear))
		self.up2 = (Up(512, 256 // factor, bilinear))
		self.up3 = (Up(256, 128 // factor, bilinear))
		self.up4 = (Up(128, 64, bilinear))
		self.outc = (OutConv(64, n_classes))

	def forward(self, x):
		x1 = self.inc(x)
		if 'E' in self.using_se:
			x1 = self.encode_SE1(x1)
		x2 = self.down1(x1)
		if 'E' in self.using_se:
			x2 = self.encode_SE2(x2)
		x3 = self.down2(x2)
		if 'E' in self.using_se:
			x3 = self.encode_SE3(x3)
		x4 = self.down3(x3)
		if 'E' in self.using_se:
			x4 = self.encode_SE4(x4)
		x5 = self.down4(x4)
		if 'D' in self.using_se:
			x5 = self.decode_SE1(x5)
		x = self.up1(x5, x4)
		if 'D' in self.using_se:
			x = self.decode_SE2(x)
		x = self.up2(x, x3)
		if 'D' in self.using_se:
			x = self.decode_SE3(x)
		x = self.up3(x, x2)
		if 'D' in self.using_se:
			x = self.decode_SE4(x)
		x = self.up4(x, x1)
		logits = self.outc(x)
		return logits

	def use_checkpointing(self):
		self.inc = torch.utils.checkpoint(self.inc)
		self.down1 = torch.utils.checkpoint(self.down1)
		self.down2 = torch.utils.checkpoint(self.down2)
		self.down3 = torch.utils.checkpoint(self.down3)
		self.down4 = torch.utils.checkpoint(self.down4)
		if 'D' in self.using_se:
			self.decode_SE1 = torch.utils.checkpoint(self.decode_SE1)
			self.decode_SE2 = torch.utils.checkpoint(self.decode_SE2)
			self.decode_SE3 = torch.utils.checkpoint(self.decode_SE3)
			self.decode_SE4 = torch.utils.checkpoint(self.decode_SE4)
		self.up1 = torch.utils.checkpoint(self.up1)
		self.up2 = torch.utils.checkpoint(self.up2)
		self.up3 = torch.utils.checkpoint(self.up3)
		self.up4 = torch.utils.checkpoint(self.up4)
		self.outc = torch.utils.checkpoint(self.outc)
