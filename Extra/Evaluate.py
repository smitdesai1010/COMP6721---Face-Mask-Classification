from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Evaluation
with torch.no_grad():
    labels_N_prediction = test.get_labels_N_prediction(model, testing_data, DEVICE)

conf_matrix = confusion_matrix(labels_N_prediction[0], labels_N_prediction[1])
plt.figure(figsize=(10, 10))
test.display_confusion_matrix(conf_matrix, preprocess.CLASSES)
