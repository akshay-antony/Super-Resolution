
import numpy as np
import torch


class RangeTwoPCD:
    def __init__(self,
                 image_rows=16,
                 image_cols=1024,
                 ang_start_y=15.0,
                 max_range=100.0,
                 min_range=3.0):

        self.ang_res_x = 360.0 / float(image_cols) # horizontal resolution
        self.ang_res_y = 30.0 / float(image_rows-1) # vertical resolution
        self.ang_start_y = ang_start_y # bottom beam angle
        self.max_range = max_range
        self.min_range = min_range

        self.image_rows = image_rows
        self.image_cols = image_cols
        elevation_col = np.arange(self.image_rows).reshape(-1, 1)
        elevation_image = np.tile(elevation_col, (1, self.image_cols))

        azimuth_row = np.arange(self.image_cols).reshape(1, -1)
        azimuth_image = np.tile(azimuth_row, (self.image_rows, 1))
        
        self.elevation_image = np.float32(elevation_image * self.ang_res_y) - self.ang_res_y
        self.azimuth_image = -np.float32(azimuth_image + 1 - (image_cols//2)) * self.ang_res_x + 90

        self.elevation_image = self.elevation_image / 180 * np.pi
        self.azimuth_image = self.azimuth_image / 180 * np.pi

        self.elevation_image = torch.from_numpy(self.elevation_image).cuda()
        self.azimuth_image = torch.from_numpy(azimuth_image).cuda()

    def convert_range_to_pcd(self,
                             range_image
                            ):
        range_image = range_image.squeeze(1)
        range_image *= self.max_range
        range_image[range_image > self.max_range] = 0
        range_image[range_image < self.min_range] = 0 

        ####
        #range_image = range_image[torch.where(range_image != 0)]
        ####
        x = torch.cos(self.elevation_image) * torch.sin(self.azimuth_image) * range_image #n, row, col
        y = torch.cos(self.elevation_image) * torch.cos(self.azimuth_image) * range_image
        z = torch.sin(self.elevation_image) * range_image

        x = x.reshape((x.shape[0], -1, 1)) #n, row*col
        y = x.reshape((y.shape[0], -1, 1))
        z = x.reshape((z.shape[0], -1, 1))

        pcd = torch.concat([x, y, z], dim=-1)
        return pcd

if __name__ == '__main__':
    range_2_pcd = RangeTwoPCD(image_rows=64)
    x = torch.randn((8, 1, 64, 1024)).cuda()
    pcd = range_2_pcd.convert_range_to_pcd(x)
    print(pcd.shape)