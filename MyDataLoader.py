from genericpath import exists
import imp
from importlib.resources import path
from ntpath import join
from requests import patch
import torch 
from torch.utils.data import DataLoader, Dataset 
import os
import numpy as np
import open3d as o3d
import errno


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
                 range_folder_name="range_{}_",
                 lidar_folder_name="lidar{}_org_full",
                 validity_folder_name="validity_{}_1024",
                 is_chamfer=True,
                 max_len=33256,
                 is_train=True):	
        if is_train:
            range_folder_name += "train"
        else:
            range_folder_name += "train"
        self.max_len = max_len
        self.filename = sorted(os.listdir(os.path.join(basepath, range_folder_name.format('16'))))
        if not os.path.exists(os.path.join(basepath, range_folder_name.format('16'))) \
            or not os.path.exists(os.path.join(basepath, lidar_folder_name.format('16'))):
            raise FileNotFoundError(errno.ENONET,
                                    os.strerror(errno.ENOENT),
                                    os.path.join(basepath, range_folder_name.format('16')))

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
            o3d_pcd_64_path = os.path.join(self.basepath,
                                           self.lidar_folder_name.format('64'),
                                           self.filename[idx].split(".npy")[0] + ".pcd")
            pcd_64 = o3d.io.read_point_cloud(o3d_pcd_64_path)
            pcd_64_np = np.asarray(pcd_64.points)
            pcd_64_np = torch.from_numpy(pcd_64_np).type(torch.FloatTensor)
            if pcd_64_np.shape[0] < self.max_len:
                pad = torch.zeros((self.max_len - pcd_64_np.shape[0], 3),
                                   dtype=torch.float32)
                pcd_64_np = torch.cat([pcd_64_np, pad], axis=0)

        range_image_16 = torch.from_numpy(range_image_16)
        range_image_64 = torch.from_numpy(range_image_64)
        range_image_16 = range_image_16.type(torch.FloatTensor)
        range_image_64 = range_image_64.type(torch.FloatTensor)

        if self.lidar_folder_name:
            return {"range_16": range_image_16,
                    "range_64": range_image_64,
                    "pcd_64": pcd_64_np}
        else:
            return {"range_16": range_image_16, "range_64": range_image_64}

    def __len__(self):
        return len(self.filename)

def downsample_data(high_res_data,
                    downsample_ratio=4):
    downsampled_data = high_res_data[:, ::downsample_ratio, :]
    return downsampled_data

def add_gaussian_noise(data,
                       sensor_noise=0.03,
                       max_range=80,
                       min_range=2,
                       normalize_ratio=80):
    noise = np.random.normal(0, sensor_noise, data.shape)
    noise[data == 0] = 0
    data = data + noise
    data[data > max_range] = 0
    data[data < min_range] = 0
    data = data / normalize_ratio
    return data

class OusterLidar(Dataset):
    def __init__(self, 
                range_folder_name="./bag/ouster/range_{}_",
                pcd_folder_name="./bag/ouster/pcd_ouster",
                is_train=True
                ):
        if is_train:
            print("Train Dataset")
            range_folder_name += "train"
            pcd_folder_name += "_train"
        else:
            print("Test Dataset")
            range_folder_name += "test"
            pcd_folder_name += "_test"

        self.pcd_folder_name = pcd_folder_name
        self.range_16_foldername = range_folder_name.format('16')
        self.range_64_foldername = range_folder_name.format('64')
        assert(len(os.listdir(self.range_16_foldername)) == \
               len(os.listdir(self.range_64_foldername)) == \
               len(os.listdir(self.pcd_folder_name)))
        print("Found: ", self.__len__(), "files ")

    def __getitem__(self, index):
        range_16 = np.load(os.path.join(self.range_16_foldername,
                                        str(index) + ".npy"))
        range_64 = np.load(os.path.join(self.range_64_foldername,
                                        str(index) + ".npy"))
        range_16 = np.expand_dims(range_16, 0)
        range_64 = np.expand_dims(range_64, 0)

        range_16 = add_gaussian_noise(range_16)
        range_64 = add_gaussian_noise(range_64)

        range_16 = torch.from_numpy(range_16).type(torch.FloatTensor)
        range_64 = torch.from_numpy(range_64).type(torch.FloatTensor)
        pcd_64 = torch.from_numpy(np.load(os.path.join(self.pcd_folder_name, 
                                                       str(index) + ".npy"))).type(torch.FloatTensor)
        return {"range_16": range_16,
                "range_64": range_64,
                "pcd_64": pcd_64}

    def __len__(self):
        return len(os.listdir(self.range_16_foldername))
