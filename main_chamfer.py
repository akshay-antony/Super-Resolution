from tkinter.messagebox import NO
import numpy as np
from sklearn import datasets
from tomlkit import item 
import torch
from yaml import parse 
from model_chamfer import Model
from MyDataLoader import MyDatasetChamferDistance, OusterLidar
from torch.utils.data import DataLoader, random_split
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
import wandb
from range_to_pcd import RangeToPCD
from utils import plot_3d
import argparse

max_range = 100
if torch.cuda.is_available():
    device = torch.device("cuda")

def train(epochs=20,
          train_loader=None,
          loss_fn1=None,
          model=None,
          optimizer=None,
          val_freq=1,
          test_loader=None,
          print_freq=20,
          loss_fn2=None,
          range_2_pcd=None,
          scheduler=None,
          loss_fn3=None):
    
    test(model, test_loader, loss_fn1, loss_fn2, 0, range_2_pcd, loss_fn3=loss_fn3)
    for i in range(epochs):
        chamfer_losses = 0
        l1_losses = 0
        scaled_losses = 0
        bce_losses = 0
        no_data = 0
        model = model.train()
        for j, batch_data in tqdm(enumerate(train_loader),
                                total=len(train_loader),
                                leave=False):
            input_range = batch_data['range_16'].to(device)
            label_range = batch_data['range_64'].to(device)
            label_pcd = batch_data['pcd_64'].to(device)
            val_label = batch_data['val_64'].to(torch.float32).to(device).reshape(label_pcd.shape[0], -1)
            
            batch_size = input_range.shape[0]
            pred_range = model(input_range)
            scaled_pred_range = pred_range * range_2_pcd.max_range
            val_pred = torch.ones_like(scaled_pred_range).to(torch.float32)
            val_pred[scaled_pred_range > range_2_pcd.max_range] = 0
            val_pred[scaled_pred_range < range_2_pcd.min_range] = 0 
            val_pred = val_pred.reshape(val_pred.shape[0], -1).to(device)

            bce_loss = loss_fn3(val_pred, val_label)
            l1_loss = loss_fn2(pred_range, label_range)
            pred_pcd = range_2_pcd.convert_range_to_pcd(pred_range.clone())
            chamfer_loss_val = loss_fn1(label_pcd, pred_pcd)[0]            
            #loss = chamfer_loss + l1_loss
            loss = l1_loss + torch.sqrt(chamfer_loss_val) / 80

            optimizer.zero_grad()		
            loss.backward()
            optimizer.step()
            
            chamfer_losses += chamfer_loss_val.item() * batch_size
            l1_losses += l1_loss.item() * batch_size
            scaled_losses += loss.item() * batch_size
            bce_losses += bce_loss.item() * batch_size
            no_data += batch_size
            torch.cuda.empty_cache()

            if j % print_freq == 0 and j != 0:
                print("Batch {}/{}, l1 loss: {}, chamfer loss: {}, bce loss: {}, total_scaled_loss: {} "
                                                 .format(j, 
                                                        len(train_loader), 
                                                        round(l1_losses / no_data, 4), 
                                                        round(chamfer_losses / no_data, 4),
                                                        round(bce_losses / no_data),
                                                        round(scaled_losses / no_data, 4)))
                wandb.log({"Train_l1_loss": l1_losses / no_data,
                           "Train_chamfer_loss": chamfer_losses / no_data,
                           "Train bce loss": bce_losses / no_data,
                           "step": i * len(train_loader)+j})

        print("Epoch {}, Epoch l1 loss: {}, Epoch chamfer loss: {}".format(i, 
                                                        round(l1_losses / no_data, 4), 
                                                        round(chamfer_losses / no_data, 4)))
        wandb.log({"Train_l1_loss": l1_losses / no_data,
                   "Train_chamfer_loss": chamfer_losses / no_data,
                   "Train bce loss": bce_losses / no_data,
                   "step": i * len(train_loader)+j})

        if i % val_freq == 0:
            test(model, test_loader, loss_fn1, loss_fn2, i, range_2_pcd, loss_fn3=loss_fn3)
        scheduler.step()

        checkpoint = {
        'epoch': i + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }

        torch.save(checkpoint, "./training_info_with_chamfer.pth")

def test(model,
         test_loader,
         loss_fn1,
         loss_fn2,
         epoch,
         range_2_pcd,
         loss_fn3):
    chamfer_losses = 0
    bce_losses = 0
    l1_losses = 0
    data_count = 0
    model = model.eval()

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader),
                            total=len(test_loader),
                            leave=False):
            input_range = data['range_16'].to(device)
            label_range = data['range_64'].to(device)
            label_pcd = data['pcd_64'].to(device)
            val_label = data['val_64'].to(torch.float32).to(device).reshape(label_pcd.shape[0], -1)

            curr_batch_size = input_range.shape[0]
            pred_range = model(input_range)
            scaled_pred_range = pred_range * range_2_pcd.max_range
            val_pred = torch.ones_like(scaled_pred_range).to(torch.float32)
            val_pred[scaled_pred_range > range_2_pcd.max_range] = 0
            val_pred[scaled_pred_range < range_2_pcd.min_range] = 0 
            val_pred = val_pred.reshape(val_pred.shape[0], -1).to(device)
            pred_pcd = range_2_pcd.convert_range_to_pcd(pred_range.clone())
            chamfer_loss_val = loss_fn1(pred_pcd, label_pcd)[0]
            bce_loss = loss_fn3(val_pred, val_label)

            if i % 20 == 0:
                plot_3d(pred_pcd[0], range_2_pcd, "pred")
                plot_3d(label_pcd[0], range_2_pcd, "gt")

            l1_loss = loss_fn2(pred_range, label_range)
            chamfer_losses += chamfer_loss_val.item() * curr_batch_size
            l1_losses += l1_loss.item() * curr_batch_size
            bce_losses += bce_loss.item() * curr_batch_size
            data_count += curr_batch_size
            torch.cuda.empty_cache()

    print("\nEpoch {}: Test l1 loss: {}, Test chamfer loss {}, Test BCE loss {}, \n".format(epoch,
                                                                    round(l1_losses / data_count, 4),
                                                                    round(chamfer_losses / data_count, 4),
                                                                    round(bce_losses / data_count)))
    wandb.log({"Test_l1_loss": l1_losses / data_count,
                "Test_chamfer_loss": chamfer_losses / data_count,
                "step": epoch})
    model = model.train()

if __name__ == '__main__':
    wandb.init(project="Super-Resolution-Chamfer-pointnet-bce")

    parser = argparse.ArgumentParser(description="Super Res")
    parser.add_argument("--data_name",
                        default="ouster",
                        choices=["ouster", "vlp"])
    parser.add_argument("--lr",
                        default=0.001)
    parser.add_argument("--batch_size",
                        default=32,
                        type=int)
    parser.add_argument("--model",
                        choices=["normal", "pointnet"],
                        default="normal")
    args = parser.parse_args()

    epochs = 20
    batch_size = args.batch_size
    loss_fn1 = chamfer_distance
    loss_fn2 = nn.L1Loss()
    
    if args.data_name == "ouster":
        print("Loading Ouster Dataset...")
        max_range = 80
        range_2_pcd = RangeToPCD(64, 
                                 ang_start_y=16.6,
                                 max_range=80,
                                 min_range=2,
                                 fov_y=33.2)
        train_dataset = OusterLidar(is_train=True)
        test_dataset = OusterLidar(is_train=False)
        loss_fn3 = nn.BCELoss()
    else:
        print("Loading VLP Dataset...")
        range_2_pcd = RangeToPCD(image_rows=64)
        train_dataset = MyDatasetChamferDistance(is_chamfer=True,
                                                 is_train=True,)
        test_dataset = MyDatasetChamferDistance(is_chamfer=True,
                                                is_train=False)

    if args.model == "pointnet":
        range_2_pcd_16 = RangeToPCD(16)
        model = Model(is_pointnet=True,
                      A=range_2_pcd_16.azimuth_image,
                      E=range_2_pcd_16.elevation_image)
        model = model.to(device)
    else:
        model = Model()
        model = model.to(device)

    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=8,
                             shuffle=False,
                             num_workers=4)
    print("Starting training...")
    train(epochs=epochs,
          train_loader=train_loader,
          loss_fn1=loss_fn1,
          loss_fn2=loss_fn2,
          model=model,
          optimizer=optimizer,
          test_loader=test_loader,
          range_2_pcd=range_2_pcd,
          scheduler=scheduler,
          loss_fn3=loss_fn3)