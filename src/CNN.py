import torch
import torch.nn as nn
import torch.nn.functional as F

class MODEL2(nn.Module):
      def __init__(self):
            super(MODEL2, self).__init__()
            self.conv_layer = nn.Sequential(
                  nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                  nn.BatchNorm2d(32),
                  nn.LeakyReLU(inplace=True),

                  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                  nn.BatchNorm2d(32),
                  nn.LeakyReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2),

                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                  nn.BatchNorm2d(64),
                  nn.LeakyReLU(inplace=True),

                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                  nn.BatchNorm2d(64),
                  nn.LeakyReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.fc_layer = nn.Sequential(
                  nn.Dropout(p=0.1),

                  nn.Linear(32 * 32 * 64, 1000),
                  nn.ReLU(inplace=True),

                  nn.Linear(1000, 512),
                  nn.ReLU(inplace=True),
                  nn.Dropout(p=0.1),

                  nn.Linear(512, 4),
                  nn.ReLU(inplace=True)
            )

      def forward(self, tensor):
            tensor = self.conv_layer(tensor)
            # print(tensor.size())
            tensor = tensor.view(tensor.size(0), -1)
            tensor = self.fc_layer(tensor)
            return tensor

class MODEL1(nn.Module):
   def __init__(self):
     super(MODEL1, self).__init__()   
     #dimension of our tensor: [BATCH_SIZE,3,IMG_SIZE,IMG_SIZE]
     # pytoch NN class has 2 types of layers - liner and convolutional layers
     # 1 input channel as out image is rgb, output channel is filter , stride is 1 default and padding is 0 default
     self.convolutional_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=6)   
     self.convolutional_2 = nn.Conv2d(in_channels=32, out_channels=120, kernel_size=6)

     # Fully Connetected linear OR dense layers (flattaned from previous layer of conv)
     self.fully_connected_1 = nn.Linear(in_features=12*12*120, out_features=120)     #4x4 is dimension of each of the 12 channels 
     self.fully_connected_2 = nn.Linear(in_features=120, out_features=60)
     self.output = nn.Linear(in_features=60, out_features=4)
    # no. of output channels are result of applying filter(kernal) on it
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
     print('5: ',tensor.size())

     #3: Fully connected Linear layer - 1 
     tensor = tensor.reshape(-1, 12*12*120)
     tensor = self.fully_connected_1(tensor)
     tensor = F.relu(tensor)

     #4: Fully connected Linear layer - 2
     tensor = self.fully_connected_2(tensor)
     tensor = F.relu(tensor)

     #5: Fully connected Linear layer - 3
     tensor = self.output(tensor)

     return tensor

#saving the model into the folder model
def save_model(model,optimizer,MODEL_FILEPATH):
      model_info = {
            'model': MODEL2(),
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()
            }

      torch.save(model_info, MODEL_FILEPATH)

#loading the model
def load_model(MODEL_FILEPATH,DEVICE):
      model_info = torch.load(MODEL_FILEPATH)
      model = model_info['model'].to(DEVICE)
      model.load_state_dict(model_info['state_dict'])

      return model