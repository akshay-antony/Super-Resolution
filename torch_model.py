import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
from torch.utils.data import DataLoader


class upSample(nn.Module):
	def __init__(self, in_features, out_features, stride):
		super(upSample, self).__init__()
		self.upsample_layer = nn.Sequential(
								 nn.ConvTranspose2d(in_features, out_features, kernel_size=(3,3), stride=stride, padding=(1,1), output_padding=(1,0)),
								 nn.BatchNorm2d(out_features),
							  	 nn.ReLU(),
								 nn.ConvTranspose2d(out_features, out_features, kernel_size=(3,3), stride=stride, padding=(1,1), output_padding=(1,0)),
								 nn.BatchNorm2d(out_features),
							  	 nn.ReLU())

	def forward(self,x):
		return self.upsample_layer(x)

class upBlock(nn.Module):
	def __init__(self, in_features, out_features, stride, padding=(1,0), out_padding=(1,0)):
		super(upBlock, self).__init__()
		self.upblock_layer = nn.Sequential(
								 nn.ConvTranspose2d(in_features, out_features, kernel_size=(3,3), stride=stride, padding=padding, output_padding=out_padding),
								 nn.BatchNorm2d(out_features),
								 nn.ReLU())

	def forward(self, x):
		return self.upblock_layer(x)

class conv(nn.Module):
	def __init__(self, in_features, out_features):
		super(conv, self).__init__()
		self.conv_layer = nn.Sequential(
							  nn.Conv2d(in_features, out_features, kernel_size=(3,3), padding=(1,1)),
							  nn.BatchNorm2d(out_features),
							  nn.ReLU(),
							  nn.Conv2d(out_features, out_features, kernel_size=(3,3), padding=(1,1)),
							  nn.BatchNorm2d(out_features),
							  nn.ReLU())

	def forward(self, x):
		return self.conv_layer(x)

class Network(nn.Module):
	def __init__(self, image_height=16, image_width=1800):
		super(Network,self).__init__()
		self.h = image_height
		self.w = image_width
		self.upsample_layer = upSample(1, 64, (2,1))
		self.conv1 = conv(64, 64)
		
		self.encoder1 = nn.Sequential(
							nn.AvgPool2d(kernel_size=(2, 2)),
							nn.Dropout(p=0.25),
							conv(64, 64*2))
		
		self.encoder2 = nn.Sequential(
							nn.AvgPool2d(kernel_size=(2, 4)),
							nn.Dropout(p=0.25),
							conv(128, 64*4))
		
		self.encoder3 = nn.Sequential(
							nn.AvgPool2d(kernel_size=(2, 2)),
							nn.Dropout(p=0.25),
							conv(256, 64*8))
		
		self.encoder4 = nn.Sequential(
							nn.AvgPool2d(kernel_size=(2, 4)),
							nn.Dropout(p=0.25),
							conv(512, 64*16))

		self.encoder5 = nn.Sequential(
							nn.AvgPool2d(kernel_size=(2,2)),
							nn.Dropout(p=0.25),
							conv(64*16, 64*32))



		self.dropout = nn.Dropout(p=0.25)
		
		self.upblock = upBlock(64*16, 64*8, (2,4), (1,0), (0,0))
		
		self.decoder1 = nn.Sequential(
							conv(1024, 64*8),
							nn.Dropout(p=0.25),
							upBlock(512, 64*4, (2,2), (1,0), (1,0)))
		
		self.decoder2 = nn.Sequential(
							conv(512, 64*4),
							nn.Dropout(p=0.25),
							upBlock(256, 64*2, (2,4), (1,0), (1,1)))
		
		self.decoder3 = nn.Sequential(
							conv(256, 64*2),
							nn.Dropout(p=0.25),
							upBlock(128, 64, (2,2), (1,1), (1,1)))
		
		self.last_layer = nn.Sequential(
							conv(128, 64),
							nn.Conv2d(64, 1, kernel_size=(1,1)),
							nn.ReLU())

	def forward(self, x):
		x0 = self.upsample_layer(x)
		x1 = self.conv1(x0)
		x2 = self.encoder1(x1)
		print(f"x2, {x2.shape}")
		x3 = self.encoder2(x2)
		print("x3 ", x3.shape)
		x4 = self.encoder3(x3)
		print("x4 ", x4.shape)

		x5 = self.encoder4(x4)
		print("x5 ", x5.shape)
		
		# y5 = self.encoder5(x5)
		# print("y5 ", y5.shape)
	
		# y5 = self.dropout(y5)
		
		y4 = self.encoder4(x4)
		
		y4 = self.dropout(y4)
		y4 = self.upblock(y4)
		print("y4 ", y4.shape)
		y3 = torch.cat((x4, y4), dim=1)
		y3 = self.decoder1(y3)
		print("y3 ", y3.shape)
		y2 = torch.cat((x3, y3), dim=1)
		y2 = self.decoder2(y2)
		print("y2 ", y2.shape)
		y1 = torch.cat((x2, y2), dim=1)
		y1 = self.decoder3(y1)
		print("y1 ", y1.shape)
		y0 = torch.cat((x1, y1), dim=1)
		output = self.last_layer(y0)
		return output

def initialize_weights(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')

if __name__ == '__main__':
	trial = Network()
	x = torch.randn((1,1,16,1024))
	y = trial(x)
	print(y.shape)
