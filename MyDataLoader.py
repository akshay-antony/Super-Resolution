import torch 
from torch.utils.data import DataLoader, Dataset 
from data import prepare_data
import os
import numpy as np


class MyDatasetBEV(Dataset):
	def __init__(self, filename="/home/akshay/bag/bev_16_np", basepath="/home/akshay/bag"):
		self.filename = sorted(os.listdir(filename))
		self.basepath = basepath

	def __getitem__(self, idx):
		bev_path = self.basepath + "/bev_16_np/" + self.filename[idx].split(".npy")[0] + ".npy"
		bev = np.load(bev_path)
		bev = np.tile(bev, (3,1,1))

		range_image16_path = self.basepath + "/range_16_1024/" + self.filename[idx].split(".npy")[0] + ".npy"
		range_image64_path = self.basepath + "/range_64_1024/" + self.filename[idx].split(".npy")[0] + ".npy"
		range_image_16 = np.load(range_image16_path)
		range_image_16 = np.expand_dims(range_image_16, axis=0)
		range_image_64 = np.load(range_image64_path)
		range_image_64 = np.expand_dims(range_image_64, axis=0)

		bev = torch.from_numpy(bev)
		range_image_16 = torch.from_numpy(range_image_16)
		range_image_64 = torch.from_numpy(range_image_64)
		range_image_16 = range_image_16.type(torch.FloatTensor)
		range_image_64 = range_image_64.type(torch.FloatTensor)
		bev = bev.type(torch.FloatTensor)
		return {'bev': bev, "range_16": range_image_16, "range_64": range_image_64}

	def __len__(self):
		return len(self.filename)

class MyDatasetRange(Dataset):
	def __init__(self, filename="/home/akshay/bag/five_channel_input/", basepath="/home/akshay/bag"):
		self.filename = sorted(os.listdir(filename))
		self.basepath = basepath

	def __getitem__(self, idx):
		input_image = np.load(self.basepath + "/five_channel_input/" + self.filename[idx].split(".npy")[0] + ".npy")
		output_image = np.load(self.basepath + "/five_channel_output/" + self.filename[idx].split(".npy")[0] + ".npy")
		
		input_image = torch.from_numpy(input_image)
		output_image = torch.from_numpy(output_image)
		input_image = input_image.type(torch.FloatTensor)
		output_image = output_image.type(torch.FloatTensor)
		return {"input_image": input_image, "output_image": output_image}

	def __len__(self):
		return len(self.filename)

class MyDataset_without_BEV(Dataset):
	def __init__(self, filename="/home/akshay/bag/bev_16_np", basepath="/home/akshay/bag"):
		self.filename = sorted(os.listdir(filename))
		self.basepath = basepath

	def __getitem__(self, idx):
		range_image16_path = self.basepath + "/range_16_1024/" + self.filename[idx].split(".npy")[0] + ".npy"
		range_image64_path = self.basepath + "/range_64_1024/" + self.filename[idx].split(".npy")[0] + ".npy"
		range_image_16 = np.load(range_image16_path)
		range_image_16 = np.expand_dims(range_image_16, axis=0)
		range_image_64 = np.load(range_image64_path)
		range_image_64 = np.expand_dims(range_image_64, axis=0)

		range_image_16 = torch.from_numpy(range_image_16)
		range_image_64 = torch.from_numpy(range_image_64)
		range_image_16 = range_image_16.type(torch.FloatTensor)
		range_image_64 = range_image_64.type(torch.FloatTensor)
		return {"range_16": range_image_16, "range_64": range_image_64}

	def __len__(self):
		return len(self.filename)