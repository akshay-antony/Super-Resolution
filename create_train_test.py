from cgi import test
from itertools import count
import random
import os
import shutil
import argparse


if __name__ == '__main__':
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