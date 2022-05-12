import numpy as np 
import torch 
from data import their_data, prepare_data
from torch_model import Network, initialize_weights, Network_depth, BEVModel, Network_org, densenet, Network_1024, Network_1024_without_bev
from MyDataset import MyDataset
from MyDataLoader import MyDatasetBEV, MyDataset_without_BEV, MyDatasetRange
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms 
from transforms import RandomHorizontalFlip, RandomVerticalFlip, normalize
from loss import ssim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import ssim_, MGELoss, avg_loss, norma

def train():
	if torch.cuda.is_available():
		device = torch.device("cuda")

	#checkpoint = torch.load("/home/akshay/python_carla/weights/checkpoint_nov_16_mae.pth")
	model = Network_1024_without_bev()
	#model.load_state_dict(checkpoint['state_dict'])
	model.train()
	#model.apply(initialize_weights)
	model = model.to(device)
	loss_func = nn.L1Loss().to(device)
	# custom_transform = transforms.Compose([RandomHorizontalFlip(),
	# 				            	RandomVerticalFlip(),
	# 				           		normalize()])

	dataset = MyDataset_without_BEV()

	print("Data Loaded, Starting Training....")

	lr = 0.0001
	optimizer = optim.Adam(model.parameters() , lr)
	#optimizer.load_state_dict(checkpoint['optimizer'])
	#scheduler.load_state_dict(checkpoint['scheduler'])
	training_data = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

	for i in range(0, 20, 1):
		losses = 0
		total_losses = 0
		no_data = 0
		b_no = 1

		for j, batch_data in enumerate(training_data):
			input_image = batch_data['range_16'].to(device)
			output_image = batch_data['range_64'].to(device)
			output = model(input_image)

			loss = loss_func(output, output_image)

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()		
			out = str(b_no) + "/" + str(502)
			
			b_no += 1

			total_losses += loss.item() * input_image.shape[0]
			no_data += input_image.shape[0]
			print("epoch=", i, "batch no=", out, "loss=", loss.item(), end="\r")
			torch.cuda.empty_cache()
		
		print("\n","epoch=", i, "loss:=", total_losses / no_data)

	checkpoint = {
	  'epoch': i + 1,
	  'state_dict': model.state_dict(),
	  'optimizer': optimizer.state_dict(),
	}

	torch.save(checkpoint, "/home/akshay/python_carla/weights/checkpoint_dec_3.pth")
	
def train_bev():
	if torch.cuda.is_available():
		device = torch.device("cuda")
		print("cuda found")

	#checkpoint = torch.load("/home/akshay/python_carla/weights/checkpoint_nov_16_mae.pth")
	model_main = Network_1024()
	#model.load_state_dict(checkpoint['state_dict'])
	
	model_main.train()
	model_main.apply(initialize_weights)
	model_main = model_main.to(device)
	loss_func = nn.L1Loss()	
	
	model_bev = densenet()
	model_bev.train()
	model_bev = model_bev.to(device)

	dataset = MyDatasetBEV()
	lr = 0.0001
	optimizer = optim.Adam(model_main.parameters(), lr)

	epochs = 20

	for i in range(epochs):
		epoch_loss = 0
		no_data = 0
		training_data = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=16)
		
		for j, batch_data in enumerate(training_data):
			bev = batch_data['bev']
			#print(bev.shape)
			input_image = batch_data['range_16']
			target_image = batch_data['range_64']

			bev = bev.to(device)
			bev_out = model_bev(bev)
			input_image = input_image.to(device)
			bev_out = bev_out.to(device)
			pred_image = model_main(input_image, bev_out)

			target_image = target_image.to(device)
			loss = loss_func(pred_image, target_image)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			epoch_loss += loss .item() * bev.shape[0]
			no_data += bev.shape[0]
			print("epoch=", i, "batch no=", j, "loss=", loss.item(), end="\r")
			torch.cuda.empty_cache()

		# if i == 0:
		# 	s = pred_image.detach().numpy()
		# 	np.save("/home/akshay/python_carla/a.npy",s[0])
		# print("\n","epoch=", i, "loss:=", epoch_loss / no_data)

	checkpoint = {
	  'state_dict': model_main.state_dict(),
	  'state_dict_bev': model_bev.state_dict(),
	  'optimizer': optimizer.state_dict(),
	}
	torch.save(checkpoint, "/home/akshay/python_carla/weights/checkpoint_dec_4.pth")

def test():
	model_main = Network()
	model_main.train()
	checkpoint = torch.load("/home/akshay/python_carla/weights/")
	#model_main.
	dataset = MyDatas
	etBEV()
	training_data = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
	input_image = training_data['range_16']
	output = model_main(input_image)
	print(output.shape)

def train_five_channel():
	if torch.cuda.is_available():
		device = torch.device("cuda")
	model = Network_1024_without_bev()
	model.train()
	model = model.to(device)
	dataset = MyDatasetRange()
	loss_func = nn.L1Loss().to(device)
	lr = 0.0001
	optimizer = optim.Adam(model.parameters() , lr)
	for i in range(0, 20, 1):
		losses = 0
		total_losses = 0
		no_data = 0
		training_data = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=16)
		b_no = 1

		for j, batch_data in enumerate(training_data):
			input_image = batch_data['input_image'].to(device)
			output_image = batch_data['output_image'].to(device)
			output = model(input_image)

			loss = loss_func(output, output_image)
			loss_range = loss_func(output[:,0,:,:], output_image[:,0,:,:])
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()		
			out = str(b_no) + "/" + str(502)
			
			b_no += 1

			total_losses += loss.item() * input_image.shape[0]
			no_data += input_image.shape[0]
			print("epoch=", i, "batch no=", out, "loss=", loss.item(), "loss range", loss_range.item(), end="\r")
			torch.cuda.empty_cache()
		
		print("\n","epoch=", i, "loss:=", total_losses / no_data)

	checkpoint = {
	  'epoch': i + 1,
	  'state_dict': model.state_dict(),
	  'optimizer': optimizer.state_dict(),
	}
	torch.save(checkpoint, "/home/akshay/python_carla/weights/checkpoint_dec_29.pth")

if __name__ == '__main__':
	train_five_channel()