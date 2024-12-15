# Forest Cover Type Prediction

This project involves training a neural network to classify forest cover types using the Covertype dataset from the sklearn.datasets module. The dataset consists of various features that describe the characteristics of different forest areas.

## Dataset
The `Covertype` dataset contains the following:

* **Features**: 54 numerical features representing different characteristics of the forest areas.
* **Target**: A categorical variable representing the forest cover type, with values ranging from 1 to 7.
  
## Data Preprocessing
1. **Loading the Data**: The dataset is loaded using the fetch_covtype function from sklearn.datasets.
2. **Normalization**: The feature values are normalized using StandardScaler to have zero mean and unit variance.
3. **Label Encoding**: The target values are encoded to start from 0 using LabelEncoder.
4. **Train-Validation-Test Split**: The data is split into training (64%), validation (16%), and test (20%) sets using train_test_split.
   
## Model Architecture
The neural network architecture is defined as follows:

* **Input Layer**: 54 input features.
* **Hidden Layer**: One hidden layer with ReLU activation.
* **Output Layer**: 7 output nodes, one for each class of forest cover type.

```python
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```
## Training
The model is trained using the following parameters:

* **Loss Function**: Cross-Entropy Loss
* **Optimizer**: Adam Optimizer
* **Batch Size**: 32 or 64
* **Number of Epochs**: 10
* **Learning Rate**: 0.001 or 0.01
  
Early stopping is implemented with a patience of 2 epochs to prevent overfitting.

## Hyperparameter Tuning
A grid search is performed over the following hyperparameters to find the best model:

* **Hidden Size**: [64, 128]
* **Learning Rate**: [0.001, 0.01]
* **Batch Size**: [32, 64]
  
The model with the highest accuracy on the validation set is selected as the best model.

## Evaluation
The best model is evaluated on the test set using accuracy and macro-averaged recall.

## Results
The best model configuration is:

* **Hidden Size**: 128
* **Learning Rate**: 0.001
* **Batch Size**: 64
  
The evaluation metrics for the best model are:

* **Accuracy**: 0.8391
* **Recall**: 0.7349
  
## How to Run
1. Clone the repository.
2. Install the required dependencies.
3. Run the training script to train the model and evaluate it on the test set.
   
```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
python train.py
```

## Dependencies
* torch
* sklearn
