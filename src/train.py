import src.preprocess as preprocess
from torch.nn.functional import cross_entropy


def train_model(model, training_data, EPOCH, optimizer, DEVICE):
    for epoch in range(EPOCH):

        training_loss = 0
        correct_prediction = 0
        data_size = 0

        for batch in training_data:
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            data_size += len(images)

            predictions = model(images)

            loss = cross_entropy(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            correct_prediction += (predictions.argmax(dim=1) == labels).sum().item()

        accuracy = correct_prediction / data_size
        training_loss = training_loss / data_size
        print(f"Epoch {epoch + 1}: Correct Prediction: {correct_prediction}/{data_size} and accuracy: {accuracy} and loss: {training_loss}")




