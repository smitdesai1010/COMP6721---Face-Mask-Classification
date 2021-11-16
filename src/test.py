import torch
import src.preprocess as preprocess
from torch.nn.functional import cross_entropy
import itertools
import numpy as np
import matplotlib.pyplot as plt

def test_model(model,testing_data,DEVICE):
    testing_loss = 0
    correct_prediction = 0 
        
    for batch in testing_data:
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)          

            prediction = model(images)

            testing_loss += cross_entropy(prediction, labels).item()
            correct_prediction += (prediction.argmax(dim=1) == labels).sum().item()


    accuracy = correct_prediction/preprocess.TEST_DATA_SIZE
    testing_loss = testing_loss/preprocess.TEST_DATA_SIZE

    print('\nTesting:')
    print(f"Correct prediction: {correct_prediction}/{preprocess.TEST_DATA_SIZE} and accuracy: {accuracy} and loss: {testing_loss}")


def get_labels_N_prediction(model,loader,DEVICE):
    all_labels = []
    all_prediction = []

    for batch in loader:
        images, labels = batch
        images = images.to(DEVICE)

        prediction = model(images).to(torch.device("cpu")).argmax(dim=1).detach().numpy()
        labels = labels.to(torch.device("cpu")).detach().numpy()

        all_prediction = np.append(all_prediction,prediction)
        all_labels = np.append(all_labels,labels)

    return [all_labels,all_prediction]


def display_confusion_matrix(conf_matrix, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(conf_matrix)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

