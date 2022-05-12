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
		  loss_fn=None,
		  model=None,
		  optimizer=None,
		  val_freq=5,
		  test_loader=None,
		  print_freq=500):
	
	for i in range(epochs):
		losses = 0
		total_losses = 0
		no_data = 0
		b_no = 1


		for j, batch_data in tqdm(enumerate(train_loader),
								  total=len(train_loader),
								  leave=False):
			input_image = batch_data['range_16'].to(device)
			output_image = batch_data['range_64'].to(device)
			label_pcd = batch_data['pcd_64']
			output = model(input_image)
			loss = loss_fn(output, label_pcd)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()		
			
			total_losses += loss.item() * input_image.shape[0]
			no_data += input_image.shape[0]
			torch.cuda.empty_cache()

			if no_data % print_freq == 0 and no_data != 0:
				print("Batch {}/{} loss: {}".format(j, 
													len(train_loader), 
													round(total_losses / no_data, 2)))
													
			
		if i % val_freq == 0:
			test(model, test_loader, loss_fn)

		print("\n","epoch=", i, "loss:=", total_losses / no_data)

	checkpoint = {
	  'epoch': i + 1,
	  'state_dict': model.state_dict(),
	  'optimizer': optimizer.state_dict(),
	}

	torch.save(checkpoint, "/home/akshay/python_carla/weights/checkpoint_dec_3.pth")
	

def test(model,
         test_loader,
		 loss_fn):
	total_losses = 0
	data_count = 0
	model.eval()
	for i, data in tqdm(enumerate(test_loader)):
		input_image = data['range_16'].to(device)
		label_pcd = data['pcd_64'].to(device)
		pred_pcd = model(input_image)
		loss = loss_fn(pred_pcd, label_pcd)
		total_losses += loss.item()
		data_count += input_image.shape[0]
	print("Test loss: ", total_losses / data_count)
	model.train()

if __name__ == '__main__':
	epochs = 20
	batch_size = 8
	loss_fn = chamfer_distance()
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
		  loss_fn=loss_fn,
		  model=model,
		  optimizer=optimizer)