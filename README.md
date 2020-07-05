# Predicting Pulsars using the HTRU 2 Data set

To learn about pulsars, watch these two videos-

- [https://www.youtube.com/watch?v=gjLk_72V9Bw](https://www.youtube.com/watch?v=gjLk_72V9Bw) -This is by NASA's Goddard Space Center
- [https://www.youtube.com/watch?v=bKkh7viXjqs](https://www.youtube.com/watch?v=bKkh7viXjqs) - By Astronomate.  

I would **highly** recommend that you read this post:
- [https://as595.github.io/classification/](https://as595.github.io/classification/)
## Sample(of size 4) of the data used

| #   | Mean of the integrated profile | Standard deviation of the integrated profile | Excess kurtosis of the integrated profile | Skewness of the integrated profile | Mean of the DM-SNR curve | Standard deviation of the DM-SNR curve | Excess kurtosis of the DM-SNR curve | Standard deviation of the DM-SNR curve | Skewness of the DM-SNR curve | target_class |
| --- | ------------------------------ | -------------------------------------------- | ----------------------------------------- | ---------------------------------- | ------------------------ | -------------------------------------- | ----------------------------------- | -------------------------------------- | ---------------------------- | ------------- |
| 0   | 140.562500                     | 55.683782                                    | -0.234571                                 | -0.699648                          | 3.199833                 | 19.110426                              | 7.975532                            | 74.242225                              | 74.242225                    | 0             |
| 1   | 102.507812                     | 58.882430                                    | 0.465318                                  | -0.515088                          | 1.677258                 | 14.860146                              | 10.576487                           | 127.393580                             | 127.393580                   | 0             |
| 2   | 103.015625                     | 39.341649                                    | 0.323328                                  | 1.051164                           | 3.121237                 | 21.744669                              | 7.735822                            | 63.171909                              | 63.171909                    | 0             |
| 3   | 136.750000                     | 57.178449                                    | -0.068415                                 | -0.636238                          | 3.642977                 | 20.959280                              | 6.896499                            | 53.593661                              | 53.593661                    | 0             |
| 4   | 88.726562                      | 40.672225                                    | 0.600866                                  | 1.123492                           | 1.178930                 | 11.468720                              | 14.269573                           | 252.567306                             | 252.567306                   | 0             |

## Designing and Creating the model
In this project, I decided to create a very simple Logistic Regression model (no ResNets be seen) that classifies pulsars.  

### Ingest and preprocess data
#### Ingestion
The **easiest** way to do this would be to add [this dataset](https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star) on Kaggle to your notebook.[See how to add data sources here on Kaggle](https://www.kaggle.com/docs/notebooks#adding-data-sources)  

Otherwise, I have created a tiny script that does it for you, if you are running locally or with other Jupyter notebook providers(ie Google Colab or Binder).
```python
from torchvision.datasets.utils import download_url
import zipfile
data_url="https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip"
download_url(data_url, ".")
with zipfile.ZipFile("./HTRU2.zip", 'r') as zip_ref:
    zip_ref.extractall(".")
!rm -rf HTRU2.zip Readme.txt
```
#### Convert to PyTorch Tensors
1. Create a dataframe(replace `PATH_TO_CSV` with actual path)
```python
import pandas as pd
filename = "PATH_TO_CSV"
df = pd.read_csv(filename)
```
2. Convert to numpy arrays- We need to split inputs and outputs.  
  Reminder- The output is the target_class
```python
import numpy as np
# Inputs
# This will get everything but the target_class into a dataframe
inputs_df = df.drop("target_class",axis=1)
# Convert Inputs
inputs_arr=inputs_df.to_numpy()
# Targets-Same thing
targets_df = df["target_class"]
targets_arr=targets_df.to_numpy()
```
3. Convert to PyTorch tensors
```python
import torch
inputs=torch.from_numpy(inputs_arr).type(torch.float64)
targets=torch.from_numpy(targets_arr).type(torch.long)
```
4. Create a Tensor Dataset for PyTorch
```python
from torch.utils.data import TensorDataset
dataset = TensorDataset(inputs,targets)
```  

#### Split the dataset
Now we can split the dataset into training and validation(this is a supervised model after all)
1. Set the size of the two datasets
```python
num_rows=df.shape[0]
val_percent = .1 # Controls(%) how much of the dataset to use as validation
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size
```
2. Random split
```python
from torch.utils.data import random_split
torch.manual_seed(2)#Ensure that we get the same validation each time.
train_ds, val_ds = random_split(dataset, (train_size, val_size))
train_ds[5]
```
3. I would recommend to set the batch size right about now.
I am going to pick 200, but adjust this to you needs.
```python
batch_size=200
```

### Designing the model
Here I decided to use a simple feed-forward neural network for two reasons- it was a going to be a simple baseline for other potential architectures  

---
## Credits

R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles,
Fifty Years of Pulsar Candidate Selection: From simple filters to a new
principled real-time classification approach, Monthly Notices of the
Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
## Links
- You can use an "image" data set called [HTRU1](https://github.com/as595/HTRU1) to achieve the same results(It also has a greater number of entry points at 60,000).
- An easier way to implement this would be to use SKLearn. This would also allow you to use KNN and SVN(and other classifier models) really easily. [Check out this awesome data set on Kaggle](https://www.kaggle.com/ytaskiran/predicting-class-of-pulsars-with-ml-algorithms)
