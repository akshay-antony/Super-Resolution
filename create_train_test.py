from cgi import test
from itertools import count
import random
import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm


def vlp_dataset():  # My data
    source_16 = "./bag/range_16_1024/"
    dest_16_train = './bag/range_16_train/'
    dest_16_test = './bag/range_16_test/'

    source_64 = "./bag/range_64_1024/"
    dest_64_train = './bag/range_64_train/'
    dest_64_test = './bag/range_64_test/'

    files = os.listdir(source_16)
    train_no_of_files = int(0.80 * len(files))
    train_filenames = random.sample(files, train_no_of_files)
    #print(train_filenames, len(train_filenames))

    if os.path.exists('./bag/range_16_train/'):
        print("Found existing so deleting range_16_train")
        shutil.rmtree('./bag/range_16_train')

    if os.path.exists('./bag/range_64_train/'):
        print("Found existing so deleting range_64_train")
        shutil.rmtree('./bag/range_64_train')
    
    if os.path.exists('./bag/range_16_test/'):
        print("Found existing so deleting range_16_test")
        shutil.rmtree('./bag/range_16_test')
    
    if os.path.exists('./bag/range_64_test/'):
        print("Found existing so deleting range_64_test")
        shutil.rmtree('./bag/range_64_test')

    os.mkdir('./bag/range_16_train')
    os.mkdir('./bag/range_64_train')
    os.mkdir('./bag/range_16_test')
    os.mkdir('./bag/range_64_test')

    for i, filename in enumerate(os.listdir(source_16)):
        if filename not in train_filenames:
            # 16 channel
            shutil.copy(os.path.join(source_16, filename),
                        os.path.join(dest_16_test, filename))
            # 64 channel
            shutil.copy(os.path.join(source_64, filename),
                        os.path.join(dest_64_test, filename))
        else:
            # 16 channel
            shutil.copy(os.path.join(source_16, filename),
                        os.path.join(dest_16_train, filename))
            # 64 channel
            shutil.copy(os.path.join(source_64, filename),
                        os.path.join(dest_64_train, filename))    
    train_files_16 = os.listdir(dest_16_train)
    train_files_64 = os.listdir(dest_64_train)
    test_files_16 = os.listdir(dest_16_test)
    test_files_64 = os.listdir(dest_64_test)

    print(len(train_files_64),
          len(train_files_16),
          len(test_files_16),
          len(test_files_64))

def ouster_dataset(): # theirs
    high_res_data = np.load("./bag/carla_ouster_range_image.npy")
    pcd_folder_name = "./bag/ouster/pcd_ouster/"

    if os.path.exists("./bag/ouster/pcd_ouster_train"):
        print("Found existing train, deleting")
        shutil.rmtree("./bag/ouster/pcd_ouster_train")
    os.makedirs("./bag/ouster/pcd_ouster_train")

    if os.path.exists("./bag/ouster/pcd_ouster_test"):
        print("Found existing test, deleting")
        shutil.rmtree("./bag/ouster/pcd_ouster_test")
    os.makedirs("./bag/ouster/pcd_ouster_test")

    indices = np.arange(high_res_data.shape[0])
    random_train_indices = random.sample(indices.tolist(), int(0.8 * indices.shape[0]))
    train_ouster_64 = np.zeros((0,
                                high_res_data.shape[1],
                                high_res_data.shape[2],
                                high_res_data.shape[3]), dtype=np.float32)
    test_ouster_64 = np.zeros((0,
                               high_res_data.shape[1],
                               high_res_data.shape[2],
                               high_res_data.shape[3]), dtype=np.float32)
                               
    train_ouster_16 = np.zeros((0, 
                                high_res_data.shape[1] // 4,
                                high_res_data.shape[2],
                                high_res_data.shape[3]), dtype=np.float32)
    test_ouster_16 = np.zeros((0,
                               high_res_data.shape[1] // 4,
                               high_res_data.shape[2],
                               high_res_data.shape[3]), dtype=np.float32)
    test_count = 0
    train_count = 0
    for i in tqdm(range(high_res_data.shape[0])):
        if i in random_train_indices:
            curr_img = np.expand_dims(high_res_data[i], 0)
            train_ouster_64 = np.concatenate([train_ouster_64, curr_img], axis=0)
            down_curr_img = curr_img[:, ::4, :, :]
            train_ouster_16 = np.concatenate([train_ouster_16, down_curr_img], axis=0)
            shutil.copy(os.path.join(pcd_folder_name, str(i) + ".npy"),
                        "./bag/ouster/pcd_ouster_train/" + str(train_count) + ".npy")
            train_count += 1
        else:
            curr_img = np.expand_dims(high_res_data[i], 0)
            test_ouster_64 = np.concatenate([test_ouster_64, curr_img], axis=0)
            down_curr_img = curr_img[:, ::4, :, :]
            test_ouster_16 = np.concatenate([test_ouster_16, down_curr_img], axis=0)
            shutil.copy(os.path.join(pcd_folder_name, str(i) + ".npy"),
                        "./bag/ouster/pcd_ouster_test/" + str(test_count) + ".npy")
            test_count += 1

        # print(i, train_ouster_64.shape, 
        #          train_ouster_16.shape,
        #          test_ouster_64.shape, 
        #          test_ouster_16.shape)

    print("Final: ", train_ouster_64.shape, 
                     train_ouster_16.shape,
                     test_ouster_64.shape, 
                     test_ouster_16.shape)
    if not os.path.exists("./bag/ouster/"):
        os.makedirs("./bag/ouster")
    np.save("./bag/ouster/range_16_ouster_train", train_ouster_16)
    np.save("./bag/ouster/range_64_ouster_train", train_ouster_64)
    np.save("./bag/ouster/range_16_ouster_test", test_ouster_16)
    np.save("./bag/ouster/range_64_ouster_test", test_ouster_64)

def split_to_files(filename="./bag/ouster/range_64_ouster_train.npy",
                   channel="64",
                   train="_train"):
    if os.path.exists("./bag/ouster/range_" + channel + train):
        print("Found exiting and deleting")
        shutil.rmtree("./bag/ouster/range_" + channel + train)

    os.makedirs("./bag/ouster/range_" + channel + train)
    data = np.load(filename)
    for i in range(data.shape[0]):
        curr_img = data[i].squeeze(-1)
        np.save("./bag/ouster/range_" + channel + train + "/" + str(i), curr_img)
        print(i)
    
if __name__ == "__main__":
    # ouster_dataset()
    split_to_files("./bag/ouster/range_64_ouster_train.npy",
                   "64",
                   "_train")
    split_to_files("./bag/ouster/range_64_ouster_test.npy",
                   "64",
                   "_test")
    split_to_files("./bag/ouster/range_16_ouster_train.npy",
                   "16",
                   "_train")
    split_to_files("./bag/ouster/range_16_ouster_test.npy",
                   "16",
                   "_test")