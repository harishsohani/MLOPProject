
# libraries for data manipulation
import pandas as pd
import sklearn

# libraries for creating a folder
import os

# libraries for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# libraries for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

# libraries for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))

# store file path of data set (csv file) in variable
DATASET_PATH = "hf://datasets/harishsohani/MLOP-Project-Tourism/tourism.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20   # final test split


# read data from csv as data frame (Note: data is read from hugging face space)
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the following columns (as they not useful for modeling)
'''
'Unnamed: 0'  --> ID of each row
'CustomerID'  --> Customer identity
'''

# drop columns from data set
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True)

# Data cleaning


# ----------------------
# Column : Gender
# ----------------------

# As per analysis only Gender column needs to be addressed for data cleaning.
# Values in Other columns look fine
# replace value 'Fe male' with 'Female' in Gender column. Gender column will have only two values 'Male' and 'Female'

# update column 'Gender' and replace 'Fe male' as 'Female'
df['Gender'] = df['Gender'].replace ('Fe Male', 'Female')



# ----------------------
# Column : MaritalStatus
# ----------------------

# Also looked at column --> MaritalStatus
# This has 4 values, where Single can be generic to 'Divorced', 'Unmarried'
# However this value is retained as one may not want to disclose exact status.
# Having this value will make user comfirmatble with right chouce

# No other processing is required.
# Preprocessor wil be used duirng model building based on nature of input variables


# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split.
# Note 80% of data is used for training and 20% for testing
# Perform train-test split
# Since the data set is imbalanced in class - using stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# save tran and test data set as scv files
X_train.to_csv("Xtrain.csv",index=False)
X_test.to_csv("Xtest.csv",index=False)
y_train.to_csv("ytrain.csv",index=False)
y_test.to_csv("ytest.csv",index=False)

# define files list for uploading
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# copy train and test data (csv files)
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="harishsohani/MLOP-Project-Tourism",
        repo_type="dataset",
    )
