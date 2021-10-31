import os
import matplotlib.pyplot as plt
import torch, torchvision
from torchvision import transforms

DATASET_DIR = os.path.dirname(os.path.abspath(__file__)) + './dataset'
IMG_SIZE = 768 
BATCH_SIZE = 32
SHUFFLE = True

#Applying Transformation
transforms = transforms.Compose([
    transforms.Resize([IMG_SIZE,IMG_SIZE]), #resizing every image into the desired values
    transforms.RandomHorizontalFlip(), #Flips images horizontally with a probability of 0.5
    transforms.ToTensor()])     #size normalization and conversation to tensor

#Loads the images and labels from the specified folder and applies the given transformation
data = torchvision.datasets.ImageFolder(DATASET_DIR, transform=transforms)                                       

TRAIN_DATA_SIZE = round(len(data) * 0.85)
TEST_DATA_SIZE  = round(len(data) * 0.15)

#Splitting data into test and train
train_data,test_data = torch.utils.data.random_split(data,[TRAIN_DATA_SIZE,TEST_DATA_SIZE])

#Loading train data into a generator which provides images in a batch 
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

#Testing
images, labels = next(iter(train_data_loader))
plt.imshow(images[0].permute(1, 2, 0), label = labels[0])
plt.show()
