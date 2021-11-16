import src.preprocess as preprocess
from torch.nn.functional import cross_entropy


def train_model(model, training_data, EPOCH, optimizer, DEVICE):
    for epoch in range(EPOCH):

        training_loss = 0
        correct_prediction = 0

        for batch in training_data:
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            prediction = model(images)

            loss = cross_entropy(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            correct_prediction += (prediction.argmax(dim=1) == labels).sum().item()

        accuracy = correct_prediction / preprocess.TRAIN_DATA_SIZE
        training_loss = training_loss / preprocess.TRAIN_DATA_SIZE
        print(
            f"Epoch {epoch + 1}: Correct Prediction: {correct_prediction}/{preprocess.TRAIN_DATA_SIZE} and accuracy: {accuracy} and loss: {training_loss}")




