
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

# read data from csv as data frame
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the following columns (as they not useful for modeling)
'''
'Unnamed: 0'  --> ID of each row
'CustomerID'  --> Customer identity
'''

# drop columns from data set
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True)

##############            <<ToDo: Continue from here>>      ####################
# Data cleaning
'''
As per analysis only Gender column needs to be addressed for data cleaning.
Values in Other columns look fine

?? Do we need to update Martial Status column?

There is no outlier treatment is needed
'''
# replace value 'Fe male' with 'Female' in Gender column. Gender column will have only two values 'Male' and 'Female'

# update column 'Gender' and replace 'Fe male' as 'Female'
df['Gender'] = df['Gender'].replace ('Fe Male', 'Female')



# data encoding
'''
This needs to be understood in details to see if this is the right way of doing it
We have following category columns to be encoded
'TypeofContact',
 'Occupation',
 'Gender',
 'ProductPitched',
 'MaritalStatus',
 'Designation'

Besides that we also have following numeric columns which represent fixed number of values. Do we encode them as well?

Following are ordinal
 'CityTier',  (1 to 3)
 'PreferredPropertyStar', (3 to 1)

 Following are logical, indicate presence or absense
  'Passport',
  'OwnCar',
'''

# Encode categorical columns --> One hot encoding
label_encoder = LabelEncoder()
df['TypeofContact'] = label_encoder.fit_transform(df['TypeofContact'])
df['Occupation'] = label_encoder.fit_transform(df['Occupation'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['ProductPitched'] = label_encoder.fit_transform(df['ProductPitched'])
df['MaritalStatus'] = label_encoder.fit_transform(df['MaritalStatus'])
df['Designation'] = label_encoder.fit_transform(df['Designation'])

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split.
# Note 80% of data is used for training and 20% for testing
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# save tran and test data set as scv files
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

# define files list for uploading
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="harishsohani/MLOP-Project-Tourism",
        repo_type="dataset",
    )
