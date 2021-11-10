import src.preprocess as preprocess
from torch.nn.functional import cross_entropy


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
