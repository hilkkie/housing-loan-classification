# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader


ROOT = Path(os.getcwd())
DATA_ROOT = ROOT / "data"

# %%
# FUNCTIONS
# function transforming relevant columns in the data
def transform_data(df):
    df_trans = (
        df
        .pipe(lambda x: x[~pd.isna(x["LoanAmount"]) & ~pd.isna(x["Loan_Status"])])
        .assign(
            Gender = lambda x: x["Gender"].map({"Male": 0, "Female": 1, np.nan: 2}),
            Married = lambda x: x["Married"].map({"No": 0, "Yes": 1, np.nan: 2}),
            Dependents = lambda x: x["Dependents"].map({"0": 0, "1": 1, "2": 2, "3+": 3, np.nan: 4}),
            Education = lambda x: x["Education"].map({"Not Graduate": 0, "Graduate": 1}),
            ApplicantIncomeNorm = lambda x: (x["ApplicantIncome"] - x["ApplicantIncome"].mean()) / x["ApplicantIncome"].std(),
            CoapplicantIncomeNorm = lambda x: (x["CoapplicantIncome"] - x["CoapplicantIncome"].mean()) / x["CoapplicantIncome"].std(),
            LoanAmountNorm = lambda x: (x["LoanAmount"] - x["LoanAmount"].mean()) / x["LoanAmount"].std(),
            Credit_History = lambda x: x["Credit_History"].fillna(2),
            Property_Area = lambda x: x["Property_Area"].map({"Rural": 0, "Semiurban": 1, "Urban": 1}),
            Loan_Status = lambda x: x["Loan_Status"].map({"Y": 1, "N": 0}))
    )
    
    return df_trans


# class representing the dataset
class HousingLoanData(Dataset):
    def __init__(self, selected_cols, train=True):
        self.data = transform_data(pd.read_csv(DATA_ROOT / "loan_data.csv"))
        self.inputs = torch.Tensor(self.data.loc[:, selected_cols].values)
        targets = torch.Tensor(self.data.loc[:, "Loan_Status"].values)
        self.targets = torch.reshape(targets, (targets.shape[0], 1))
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        sample = self.inputs[idx,:]
        label = self.targets[idx]
        return sample, label
    
    
# function performing training/test split given a list of fractions
# and data sampling for batches
def get_data_loaders(dataset, data_split, batch_size=5, shuffle=True):
    training_dataset, test_dataset = torch.utils.data.random_split(dataset, data_split)
    
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, test_dataloader


# function calculating model accuracy over given data
def compute_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            outputs = (model.forward(x) > 0.5).float()
            total += y.size(0)
            correct += (outputs == y).sum().float()
    return correct / total

# %%
# DATA
# define set of model variables
model_attr = ["Gender", "Dependents", "Education",
              "ApplicantIncomeNorm", "CoapplicantIncomeNorm", "LoanAmountNorm",
              "Credit_History", "Property_Area"]

housing_data = HousingLoanData(model_attr)

# %%
# inspect model variables
housing_dataset = housing_data.data

fig, axes = plt.subplots(2, 4, figsize=(10,6))
axes = np.ravel(axes)

for ii, ax in enumerate(axes):
    ax.hist(housing_dataset[model_attr[ii]], align="mid")
    ax.set_title(model_attr[ii])
    ax.set_axisbelow(True)
    ax.grid(which="both")
    
np.reshape(axes, (2, 4))
fig.tight_layout()

plt.show()

# %%
# create data loaders
train_dataloader, test_dataloader = get_data_loaders(housing_data, [0.8, 0.2], batch_size=5)

# %%
# LOGISTIC REGRESSION
# model for logistic regression
class LogisticRegression(nn.Module):
    def __init__(self, n_inputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)
        
    def forward(self, x):
        return self.linear(x)
    
# create logistic regression model for training
logreg = LogisticRegression(len(model_attr))

# set up optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(logreg.parameters(), lr = learning_rate) # create optimizer

n_epochs = 200
accuracy = np.zeros(n_epochs)

for n in range(n_epochs):
    for inputs, outputs in train_dataloader:
        optimizer.zero_grad() # zero the gradient buffers
        ypred = logreg(inputs) # calculate the output for training set
        loss = F.binary_cross_entropy(F.sigmoid(ypred), outputs) # calculate binary cross entropy loss
        loss.backward() # backpropagation of the error
        optimizer.step() # update the weights
        
    accuracy[n] = compute_accuracy(logreg, test_dataloader)
    
plt.figure()
plt.plot(np.arange(1, n_epochs+1), accuracy)
plt.show()