import numpy as np
import torch
import wandb


def plot_3d(pcd,
            range_2_pcd,
            wandb_name="pred"):
    pcd = pcd.detach().cpu().numpy()
    intensity = range_2_pcd.intensity.reshape(-1, 1)
    intensity /= intensity.max()
    red = np.int16(intensity*255)
    pcd = np.concatenate([pcd, red, red[::-1], red[::-1]], axis=-1)
    wandb.log({wandb_name: wandb.Object3D({
               "type": "lidar/beta",
               "points": pcd})
              })

if __name__ == '__main__':
    pass