# COMP6721---Face-Mask-Classification

Get added to our private github repo and download the repository

Updates for Phase 2

- Used a bigger model
- Decresed learning rate from 0.07 to 0.000001
- Balanced the dataset as earlier No-mask images were constituting 50% of the dataset
- Added 10fold and bias evaluation code 


### Analysis 

- Bias
    - We saw bias in the following categories: 
        - MALE (cloth-mask)
        - MALE (surgical-mask)
        - OLD  (cloth-mask)

    - We lacked images in the following categories:
        - MALE (FFP2-mask)
        - TEEN (all categories)
        - OLD  (all categories)

- k-fold
    - Fixed train-test split: 
        - Accuracy: 95
    - 10fold Prebias:
        - Accuracy: 88.6 
        - Standard Deviation: 3.16
    - 10fold Postbias:
        - Accuracy: 92.1 
        - Standard Deviation: 1.75


```
https://github.com/smitdesai1010/COMP6721---Face-Mask-Classification.git
```

- Install Python 3.8 and pip. 
- Build and activate a python virtual environment
- Run the following command

```
pip install requirements.txt
```

To run the project: 
```
python app.py
```


#### File description

- app.py : Main file that imports other classes and methods defined by us. It first loads the preprocessed data, then builds the model and optimizer, trains the model and saves it and then the model is loaded again and evaluated

- preprocess.py : Loads the dataset, applies various transformation, splits into training-testing and loads data into batches

- CNN.py : Contains a class that builds, trains and returns a CNN network. Also there are methods to load and save a method.

- train.py : trains the model on training data

- test.py : Evaluates the model on testing data and generates the confusion matrix