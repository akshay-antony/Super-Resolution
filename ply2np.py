import os 
import open3d as o3d 
import numpy as np


if __name__ == '__main__':
	filename = 0 
	names16 = ["p16", "p16_1", "p16_2", "p16_3"]
	names64 = ["p64", "p64_1", "p64_2", "p64_3"]

	basepath = "/home/akshay/dynamic_obstacles/"
	for name16, name64 in zip(names16, names64):
		for file_16, file_64 in sorted(zip(os.listdir(os.path.join("/home/akshay/dynamic_obstacles", name16)), os.listdir(os.path.join("/home/akshay/dynamic_obstacles", name64)))):
			filename_16 = os.path.join(basepath, name16, file_16)
			filename_64 = os.path.join(basepath, name64, file_64)

			pcl_16 = o3d.io.read_point_cloud(filename_16)
			pcl_64 = o3d.io.read_point_cloud(filename_64)

			pcl_16_np = np.asarray(pcl_16.points)
			pcl_64_np = np.asarray(pcl_64.points)

			filename += 1
			np.save(os.path.join(basepath, "point_16", str(filename) + ".npy"), pcl_16_np)
			np.save(os.path.join(basepath, "point_64", str(filename) + ".npy"), pcl_64_np)
			print(filename)
