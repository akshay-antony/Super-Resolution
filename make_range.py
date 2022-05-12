import numpy as np 
import open3d as o3d
import cv2
import os


def create_range_image(rows=16, channel_str="_16_1024/"):
    # range image size, depends on your sensor, i.e., VLP-16: 16x1800, OS1-64: 64x1024
    image_rows_full = rows
    image_cols = 1024

    # Ouster OS1-64 (gen1)
    ang_res_x = 360.0 / float(image_cols) # horizontal resolution
    ang_res_y = 30 / float(image_rows_full-1) # vertical resolution
    ang_start_y = 15 # bottom beam angle
    max_range = 100.0
    min_range = 3

    for count, filename in enumerate(sorted(os.listdir("/home/akshay/bag/lidar16_org_full"))):
        if(count % 4 != 0):
            continue
        name16 = "/home/akshay/bag/lidar16_org_full/" + filename
        name64 = "/home/akshay/bag/lidar64_org_full/" + filename

        pcd_file16 = o3d.io.read_point_cloud(name16)
        pcd_file64 = o3d.io.read_point_cloud(name64)
        #filename = filename.split('.',1)[0] + filename.split('.')[0]

        points_obj = np.asarray(pcd_file64.points)
        points_array = np.array(list(points_obj), dtype=np.float32)
        
        # project points to range image
        range_image = np.zeros((image_rows_full, image_cols), dtype=np.float32)
        elevation_image = np.zeros((image_rows_full, image_cols), dtype=np.float32)
        height_image = np.zeros((image_rows_full, image_cols), dtype=np.float32)
        validity_image = np.zeros((image_rows_full, image_cols), dtype=np.float32)
        azimuth_image = np.zeros((image_rows_full, image_cols), dtype=np.float32)
        #

        x = points_array[:,0]
        y = points_array[:,1]
        z = points_array[:,2]
        
        # find row id
        vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
        relative_vertical_angle = vertical_angle + ang_start_y
        rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))
        rowId = rowId.astype(np.int32)

        # find column id
        horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi
        colId = -np.int_((horitontal_angle-90.0)/ang_res_x) + image_cols/2;
        shift_ids = np.where(colId >= image_cols)
        colId[shift_ids] = colId[shift_ids] - image_cols
        colId = colId.astype(np.int32)
        
        # filter range
        thisRange = np.sqrt(x * x + y * y + z * z)
        thisRange[thisRange > max_range] = 0
        thisRange[thisRange < min_range] = 0

        first = filename.split(".pcd")[0]
        file16 = "/home/akshay/bag/range" + channel_str + first + ".npy"
        file16_ele = "/home/akshay/bag/ele" + channel_str + first + ".npy"
        file16_hei = "/home/akshay/bag/height" + channel_str + first + ".npy"
        file16_val = "/home/akshay/bag/validity" + channel_str + first + ".npy"
        file16_azi = "/home/akshay/bag/azimuth" + channel_str + first + ".npy"
        #file16 = "/home/akshay/bag/range_16/" + first + ".npy"

        range_image[rowId,colId] = thisRange / 100
        height_image[rowId, colId] = z / 26
        elevation_image[rowId, colId] = vertical_angle  # / 15.
        azimuth_image[rowId, colId] = horitontal_angle
        valid_idx = np.where(range_image != 0)
        validity_image[valid_idx] = 1

        #print(np.sum(range_image != 0), rowId, colId)
        print(count)
        #np.save(file16, range_image)
        np.save(file16_ele, elevation_image)
        np.save(file16_hei, height_image)
        np.save(file16_val, validity_image)
        np.save(file16_azi, azimuth_image)

if __name__ == '__main__':
    create_range_image()
    # for a, b, c in zip(sorted(os.listdir("/home/akshay/bag/range_16")), sorted(os.listdir("/home/akshay/bag/bev_16_np")),sorted(os.listdir("/home/akshay/bag/range_64"))):
    #      print(a == b == c)



