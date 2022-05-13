import numpy as np 
import torch 
from torch_model import Network, initialize_weights, Network_depth, BEVModel, Network_org, densenet, Network_1024, Network_1024_without_bev
from MyDataLoader import MyDatasetChamferDistance
from torch.utils.data import DataLoader, random_split
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm

if torch.cuda.is_available():
	device = torch.device("cuda")

def train(epochs=20,
		  train_loader=None,
		  loss_fn1=None,
		  model=None,
		  optimizer=None,
		  val_freq=5,
		  test_loader=None,
		  print_freq=500,
		  loss_fn2=None):
	
	for i in range(epochs):
		chamfer_losses = 0
		l1_losses = 0
		no_data = 0
		model = model.train()
		for j, batch_data in tqdm(enumerate(train_loader),
								  total=len(train_loader),
								  leave=False):
			input_range = batch_data['range_16'].to(device)
			label_range = batch_data['range_64'].to(device)
			label_pcd = batch_data['pcd_64']
			batch_size = input_range.shape[0]
			pred_pcd, pred_range = model(input_range)
			chamfer_loss = torch.tensor([0]).cuda() if loss_fn1 == None else (pred_pcd, label_pcd)
			l1_loss = loss_fn2(pred_range, label_range)
			loss = chamfer_loss + l1_loss
			
			optimizer.zero_grad()		
			loss.backward()
			optimizer.step()
			
			chamfer_losses += chamfer_loss.item() * batch_size
			l1_losses += l1_loss.item() * batch_size
			no_data += batch_size
			torch.cuda.empty_cache()

			if no_data % print_freq == 0 and no_data != 0:
				print("Batch {}/{}, l1 loss: {}, chamfer loss: {}".format(j, 
														len(train_loader), 
														round(l1_losses / no_data, 2), 
														round(chamfer_losses / no_data, 2)))
			
		if i % val_freq == 0:
			test(model, test_loader, loss_fn1, loss_fn2, i)

		print("Batch {}/{}, Epoch l1 loss: {}, Epoch chamfer loss: {}".format(j, 
														len(train_loader), 
														round(l1_losses / no_data, 2), 
														round(chamfer_losses / no_data, 2)))

	checkpoint = {
	  'epoch': i + 1,
	  'state_dict': model.state_dict(),
	  'optimizer': optimizer.state_dict(),
	}

	torch.save(checkpoint, "./training_info.pth")
	

def test(model,
         test_loader,
		 loss_fn1,
		 loss_fn2,
		 epoch):
	loss_fn1 = chamfer_loss() if loss_fn1 == None else loss_fn1
	chamfer_losses = 0
	l1_losses = 0
	data_count = 0
	model = model.eval()
	for i, data in tqdm(enumerate(test_loader)):
		input_range = data['range_16'].to(device)
		label_range = data['range_64'].to(device)
		label_pcd = data['pcd_64'].to(device)

		curr_batch_size = input_range.shape[0]
		pred_pcd, pred_range = model(input_range)
		chamfer_loss = loss_fn1(pred_pcd, label_pcd)
		l1_loss = loss_fn2(pred_range, label_range)
		chamfer_losses += chamfer_loss.item() * curr_batch_size
		l1_losses += l1_loss.item() * curr_batch_size
		data_count += curr_batch_size
	print("Epoch {}: Test l1 loss: {}, Test chamfer loss{} ".format(epoch,
																	round(l1_losses / data_count, 2),
																	round(chamfer_losses / data_count, 2)))
	model = model.train()

if __name__ == '__main__':
	epochs = 20
	batch_size = 8
	loss_fn1 = chamfer_distance()
	loss_fn2 = nn.L1Loss()
	dataset = MyDatasetChamferDistance(is_chamfer=True)
	model = Network_1024_without_bev()
	lr = 0.0001
	optimizer = torch.optim.Adam(lr)

	train_dataset_len = int(0.70 * len(dataset))
	train_dataset, test_dataset = random_split(dataset, 
											   [train_dataset_len, len(dataset) - train_dataset_len],
											   generator=torch.Generator.manual_seed(100))
	train_loader = DataLoader(train_dataset,
							  batch_size,
							  shuffle=True)
	test_loader = DataLoader(test_dataset,
							 batch_size,
							 shuffle=False)

	train(epochs=epochs,
		  train_loader=train_loader,
		  loss_fn1=loss_fn1,
		  loss_fn2=loss_fn2,
		  model=model,
		  optimizer=optimizer)