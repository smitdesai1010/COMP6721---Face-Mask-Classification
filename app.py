import os
import matplotlib.pyplot as plt
import torch, torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

DATASET_DIR = os.path.dirname(os.path.abspath(__file__)) + './dataset'
IMG_SIZE = 64 
BATCH_SIZE = 32
SHUFFLE = True
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
# images, labels = next(iter(train_data_loader))
# plt.imshow(images[0].permute(1, 2, 0), label = labels[0])
# plt.show()

#---------------------------------------Model----------------------------------------------------




class CNN_MODEL(nn.Module):
   def __init__(self):
     super(CNN_MODEL, self).__init__()   
     #dimension of our tensor: [BATCH_SIZE,3,IMG_SIZE,IMG_SIZE]
     
     self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=6)   
     self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=6)      

     # Fully Connetected linear OR dense layers (flattaned from previous layer of conv)
     self.fc1 = nn.Linear(in_features=12*12*12, out_features=120)     #4x4 is dimension of each of the 12 channels 
     self.fc2 = nn.Linear(in_features=120, out_features=60)
     self.out = nn.Linear(in_features=60, out_features=4)

     #FORMULA: (width_of_input - kernel_size + 1)/stride

   def forward(self, tensor):        
     #here our input tensor is transformed as we move through our layers

     #1st input layer 
     tensor = tensor
     
     #2nd : hidden convolutional layer 1
     tensor = self.conv1(tensor)
     tensor = F.relu(tensor) 
     tensor = F.max_pool2d(tensor, kernel_size=2, stride=2)
     #print('3: ',tensor.size())

     #3rd : hidden convolutional layer 2
     tensor = self.conv2(tensor)
     tensor = F.relu(tensor)
     #print('4: ',tensor.size())
     tensor = F.max_pool2d(tensor, kernel_size=2, stride=2)         
     #print('5: ',tensor.size())

     #4th : 1st fully connected Linear layer 
     tensor = tensor.reshape(-1, 12*12*12)
     tensor = self.fc1(tensor)
     tensor = F.relu(tensor)

     #5th : 2nd fully connected Linear layer 
     tensor = self.fc2(tensor)
     tensor = F.relu(tensor)

     #6th : 3nd fully connected Linear layer - Final output layer
     tensor = self.out(tensor)

     return tensor


model = CNN_MODEL()
model.to(DEVICE)


print(summary(model,input_size=(3,IMG_SIZE,IMG_SIZE)))

optimizer = torch.optim.Adam(model.parameters(), lr=0.07)
loss = nn.CrossEntropyLoss()

def get_num_correct(prediction, labels):
  return prediction.argmax(dim=1).eq(labels).sum().item()

for epoch in range(3):    #10 ephochs

  total_loss = 0
  total_correct = 0 
    
  for batch in train_data_loader:
    images, labels = batch
    images, labels = images.to(DEVICE), labels.to(DEVICE)          

    prediction = model(images)

    loss = F.cross_entropy(prediction, labels)       
  
    optimizer.zero_grad()                       
    loss.backward()                            
    optimizer.step()                           
    
    total_loss += loss.item()
    total_correct += get_num_correct(prediction, labels)

  print(f"Epoch {epoch+1}: Total number of correct: {total_correct} and loss: {total_loss}")