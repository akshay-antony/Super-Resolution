import numpy as np
import os
import open3d as o3d


for filename in sorted(os.listdir("/home/akshay/python_carla/lidar_64")):
	name16 = "/home/akshay/python_carla/lidar_16/" + filename
	name64 = "/home/akshay/python_carla/lidar_64/" + filename
	pcd_file16 = o3d.io.read_point_cloud(name16)
	pcd_file64 = o3d.io.read_point_cloud(name64)
	o3d.visualization.draw_geometries([pcd_file64])