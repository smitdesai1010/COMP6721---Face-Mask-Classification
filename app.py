import os
import matplotlib.pyplot as plt
import torch, torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Dataset')
IMG_SIZE = 64 
BATCH_SIZE = 32
SHUFFLE = True
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
EPOCH = 10
MODEL_FILEPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Model/model.pth')
CLASSES = ['Cloth-Mask','FFP2-Mask','No-Mask','Surgical-Mask']

#Applying Transformation
transforms = transforms.Compose([
    transforms.Resize([IMG_SIZE,IMG_SIZE]), #resizing every image into the desired values
    transforms.RandomHorizontalFlip(), #Flips images horizontally with a probability of 0.5
    transforms.ToTensor() #size normalization and conversation to tensor
  ])     

#Loads the images and labels from the specified folder and applies the given transformation
data = torchvision.datasets.ImageFolder(DATASET_DIR, transform=transforms)                                       

TRAIN_DATA_SIZE = round(len(data) * 0.85)
TEST_DATA_SIZE  = round(len(data) * 0.15)

#Splitting data into test and train
train_data,test_data = torch.utils.data.random_split(data,[TRAIN_DATA_SIZE,TEST_DATA_SIZE])

#Loading train data into a generator which provides images in a batch 
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
#Loading test data into a generator which provides images in a batch 
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

# Testing
images, labels = next(iter(train_data_loader))
print(CLASSES[labels[0]])
plt.imshow(images[0].permute(1, 2, 0), label = labels[0])
plt.show()


#---------------------------------------Model----------------------------------------------------

class CNN_MODEL(nn.Module):
   def __init__(self):
     super(CNN_MODEL, self).__init__()   
     #dimension of our tensor: [BATCH_SIZE,3,IMG_SIZE,IMG_SIZE]
     
     self.convolutional_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=6)   
     self.convolutional_2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=6)      

     # Fully Connetected linear OR dense layers (flattaned from previous layer of conv)
     self.fully_connected_1 = nn.Linear(in_features=12*12*12, out_features=120)     #4x4 is dimension of each of the 12 channels 
     self.fully_connected_2 = nn.Linear(in_features=120, out_features=60)
     self.output = nn.Linear(in_features=60, out_features=4)

     #FORMULA: (width_of_input - kernel_size + 1)/stride

   def forward(self, tensor):        
     
     #1: Convolutional layer 1
     tensor = self.convolutional_1(tensor)
     tensor = F.relu(tensor) 
     tensor = F.max_pool2d(tensor, kernel_size=2, stride=2)
     #print('3: ',tensor.size())

     #2: Convolutional layer 2
     tensor = self.convolutional_2(tensor)
     tensor = F.relu(tensor)
     #print('4: ',tensor.size())
     tensor = F.max_pool2d(tensor, kernel_size=2, stride=2)         
     #print('5: ',tensor.size())

     #3: Fully connected Linear layer - 1 
     tensor = tensor.reshape(-1, 12*12*12)
     tensor = self.fully_connected_1(tensor)
     tensor = F.relu(tensor)

     #4: Fully connected Linear layer - 2
     tensor = self.fully_connected_2(tensor)
     tensor = F.relu(tensor)

     #5: Fully connected Linear layer - 3
     tensor = self.output(tensor)

     return tensor

# Loading the model and uploading it to the system device (CPU or GPU)
model = CNN_MODEL().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.07)
loss = nn.CrossEntropyLoss()

print(summary(model,input_size=(3,IMG_SIZE,IMG_SIZE)))


#---------------------------------------Training -------------------------------------
for epoch in range(EPOCH):  

  training_loss = 0
  correct_prediction = 0 
    
  for batch in train_data_loader:
    images, labels = batch
    images, labels = images.to(DEVICE), labels.to(DEVICE)          

    prediction = model(images)

    loss = F.cross_entropy(prediction, labels)       
  
    optimizer.zero_grad()                       
    loss.backward()                            
    optimizer.step()                           
    
    training_loss += loss.item()
    correct_prediction += (prediction.argmax(dim=1) == labels).sum().item()

  accuracy = correct_prediction/round(len(data) * 0.85)
  training_loss = training_loss/round(len(data) * 0.85)
  print(f"Epoch {epoch+1}: Total number of correct: {correct_prediction}/{round(len(data) * 0.85)} and accuracy: {accuracy} and loss: {training_loss}")


#----------------------------------------Model save and load--------------------------
checkpoint = {
          'model': CNN_MODEL().to(DEVICE),
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict()
        }

torch.save(checkpoint, MODEL_FILEPATH)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model

model = load_checkpoint(MODEL_FILEPATH)


#-----------------------------------------Testing-------------------------------------

def test(): 
  testing_loss = 0
  correct_prediction = 0 
      
  for batch in test_data_loader:
    images, labels = batch
    images, labels = images.to(DEVICE), labels.to(DEVICE)          

    prediction = model(images)

    testing_loss += F.cross_entropy(prediction, labels).item()
    correct_prediction += (prediction.argmax(dim=1) == labels).sum().item()

  accuracy = correct_prediction/round(len(data) * 0.15)
  testing_loss = testing_loss/round(len(data) * 0.15)
  print(f"Total number of correct: {correct_prediction}/{round(len(data) * 0.15)} and accuracy: {accuracy} and loss: {testing_loss}")

test()