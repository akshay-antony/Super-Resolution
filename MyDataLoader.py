import imp
import torch 
from torch.utils.data import DataLoader, Dataset 
import os
import numpy as np
import open3d as o3d

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

class MyDatasetChamferDistance(Dataset):
	def __init__(self, 
				 basepath="./bag",
				 range_folder_name="range_{}_1024",
				 lidar_folder_name="lidar_{}_org_full",
				 validity_folder_name="validity_{}_1024",
				 is_chamfer=True):
		self.filename = sorted(os.listdir(os.path.join(basepath, range_folder_name.format('16'))))
		self.range_folder_name = range_folder_name
		self.lidar_folder_name = lidar_folder_name
		self.validity_folder_name = validity_folder_name
		self.basepath = basepath
		self.is_chamfer = is_chamfer

	def __getitem__(self, idx):
		range_image16_path = os.path.join(self.basepath,
										  self.range_folder_name.format('16'),
										  self.filename[idx])

		#self.basepath + "/range_16_1024/" + self.filename[idx].split(".npy")[0] + ".npy"
		range_image64_path = os.path.join(self.basepath,
										  self.range_folder_name.format('64'),
										  self.filename[idx])
		#self.basepath + "/range_64_1024/" + self.filename[idx].split(".npy")[0] + ".npy"
		range_image_16 = np.load(range_image16_path)
		range_image_16 = np.expand_dims(range_image_16, axis=0)
		range_image_64 = np.load(range_image64_path)
		range_image_64 = np.expand_dims(range_image_64, axis=0)

		if self.lidar_folder_name:
    		### loading original lidar files
			o3d_pcd_16_path = os.path.join(self.basepath,
										   self.lidar_folder_name.format('16'),
										   self.filename[idx].split(".npy")[0] + ".pcd")
			o3d_pcd_64_path = os.path.join(self.basepath,
										   self.lidar_folder_name.format('64'),
										   self.filename[idx].split(".npy")[0] + ".pcd")
			pcd_16 = o3d.io.read_point_cloud(o3d_pcd_16_path)
			pcd_64 = o3d.io.read_point_cloud(o3d_pcd_64_path)
			pcd_16_np = np.asarray(pcd_16)
			pcd_64_np = np.asarray(pcd_64)
			pcd_16_np = torch.from_numpy(pcd_16_np).type(torch.FloatTensor)
			pcd_64_np = torch.from_numpy(pcd_64_np).type(torch.FloatTensor)
			
			### loading validity images
			validity_img_64_path = os.path.join(self.basepath,
												self.validity_folder_name.format('64'),
												self.filename[idx])
			validity_img_64 = np.load(validity_img_64_path)

		range_image_16 = torch.from_numpy(range_image_16)
		range_image_64 = torch.from_numpy(range_image_64)
		range_image_16 = range_image_16.type(torch.FloatTensor)
		range_image_64 = range_image_64.type(torch.FloatTensor)

		if self.lidar_folder_name:
			return {"range_16": range_image_16,
					"range_64": range_image_64,
					"pcd_64": pcd_64,
					"val_64": validity_img_64}
		else:
			return {"range_16": range_image_16, "range_64": range_image_64}

	def __len__(self):
		return len(self.filename)