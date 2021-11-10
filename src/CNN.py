import torch
import torch.nn as nn
import torch.nn.functional as F

class MODEL(nn.Module):
   def __init__(self):
     super(MODEL, self).__init__()   
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


def save_model(model,optimizer,MODEL_FILEPATH):
      model_info = {
            'model': MODEL(),
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()
            }

      torch.save(model_info, MODEL_FILEPATH)


def load_model(MODEL_FILEPATH,DEVICE):
      model_info = torch.load(MODEL_FILEPATH)
      model = model_info['model'].to(DEVICE)
      model.load_state_dict(model_info['state_dict'])

      return model