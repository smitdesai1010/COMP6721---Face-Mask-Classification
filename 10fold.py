import os
import warnings
import torch, torchvision
from torchvision import transforms
from sklearn.model_selection import KFold
import src.CNN as CNN
import src.train as train
import src.test as test
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

warnings.filterwarnings('always')
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
DATASET_DIR = os.path.join(os.path.abspath(os.curdir),'Dataset')
CLASSES = ['Cloth-Mask','FFP2-Mask','No-Mask','Surgical-Mask']
IMG_SIZE = 128 
BATCH_SIZE = 32
SHUFFLE = True
EPOCH = 10

# Applying Transformation
transforms = transforms.Compose([
        transforms.Resize([IMG_SIZE,IMG_SIZE]), #resizing every image into the desired values
        transforms.RandomHorizontalFlip(), #Flips images horizontally with a probability of 0.5
        transforms.ToTensor()   #size normalization and conversation to tensor
    ])   

data = torchvision.datasets.ImageFolder(DATASET_DIR, transform=transforms) 
X = data.imgs
Y = data.targets
k_fold = KFold(n_splits=10, shuffle = True)

for train_index, test_index in k_fold.split(data):
    print('--------------------------------------------------------')
    print('--------------------------------------------------------')
    
    train_data = []
    test_data = []

    # python 10fold.py

    for i in range(len(data)):
        if i in test_index:
            test_data.append(data[i])
        else:
            train_data.append(data[i])
    
    # print(len(train_data))
    # print(len(test_data))
    x_train_fold = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    x_test_fold = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    model = CNN.MODEL2().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    train.train_model(model,x_train_fold,EPOCH,optimizer,DEVICE)
    test.test_model(model,x_test_fold,DEVICE)

    with torch.no_grad():
        labels_N_prediction = test.get_labels_N_prediction(model, x_test_fold, DEVICE)

    print(classification_report(labels_N_prediction[0], labels_N_prediction[1], target_names = CLASSES, zero_division = 1))
    conf_matrix = confusion_matrix(labels_N_prediction[0], labels_N_prediction[1])
    print(conf_matrix)