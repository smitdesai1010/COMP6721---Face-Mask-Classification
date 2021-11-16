import os
import torch
import src.preprocess as preprocess
import src.CNN as CNN
import src.train as train
import src.test as test
from torchsummary import summary
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
MODEL_FILEPATH = os.path.join(os.path.abspath(os.curdir),'Model/model.pth')
EPOCH = 5


# Loading training and testing preprocessed data
training_data = preprocess.get_training_data()
testing_data = preprocess.get_testing_data()



# Loading the model and uploading it to the system device (CPU or GPU)
model = CNN.MODEL().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.07)
print(summary(model,input_size=(3,preprocess.IMG_SIZE,preprocess.IMG_SIZE)))



# Training and saving
# train.train_model(model,training_data,EPOCH,optimizer,DEVICE)
# CNN.save_model(model,optimizer,MODEL_FILEPATH)



#loading and testing
# model = CNN.load_model(MODEL_FILEPATH,DEVICE)
# test.test_model(model,testing_data,DEVICE)


# Evaluation
# with torch.no_grad():
#     labels_N_prediction = test.get_labels_N_prediction(model, testing_data, DEVICE)

# conf_matrix = confusion_matrix(labels_N_prediction[0], labels_N_prediction[1])
# plt.figure(figsize=(10, 10))
# test.display_confusion_matrix(conf_matrix, preprocess.CLASSES)

