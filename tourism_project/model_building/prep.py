# for data manipulation
import pandas as pd
import sklearn
Random_State = 42
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
#for cleaning the texts
import re
from huggingface_hub import HfApi, CommitOperationAdd

# Define constants for the dataset and output paths
api = HfApi(token = os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/samdurai102024/Tourism-Package-Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

#“Free Lancer” is merged into “Small Business” because it has only 2 observations (0.05%), causing statistical instability and one-hot encoding mismatches.
#Freelancing is behaviorally aligned with small business income patterns, making ‘Small Business’ the most appropriate and domain-consistent merged category.
df["Occupation"] = df["Occupation"].replace({"Free Lancer": "Small Business"})

#The categorical values such as "Self Enquiry", "Company Invited", "Small Business", "Large Business", "Free Lancer", "Fe Male", "Senior Manager"
#appear clean but contain internal spaces, which cause OneHotEncoding to treat them as different categories whenever spacing varies across train, validation, and test splits.
#To guarantee category consistency and prevent missing-column errors like “ValueError: columns are missing:
#{'Occupation_Free Lancer'}”, we standardize all categorical values by removing spaces and special characters before splitting the dataset.
#This produces deterministic, uniform category labels (e.g., "Free Lancer" → "FreeLancer") and ensures the encoding pipeline works reliably across all datasets.

# Select all categorical and object columns for one-hot encoding
categorical_cols = df.select_dtypes(include=['category', 'object']).columns

#Universal Categorical Cleaning Function
def clean_category(x):
    if pd.isna(x):
        return x
    x = x.strip()
    x = re.sub(r"[^A-Za-z0-9]", "", x)  # remove spaces & special chars
    return x

for col in categorical_cols:
    df[col] = df[col].apply(clean_category)

# defined X predictors and y target datasets
target = "ProdTaken"

# Split X and y with correct shapes
X = df.loc[:, df.columns != "ProdTaken"].copy()
y = df.loc[:, ["ProdTaken"]].copy()  # dataframe

# Perform train-validation-test split
# First variation, use one with stratify with 60:20:20 splitting and scaling
# splitting the data for 60:20:20 ratio between train, validation and test sets
# stratify ensures the training, validation and test sets have a similar distribution of the response variable
# Let's split data into temporary and test - 2 parts
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.2, random_state = Random_State, stratify = y, shuffle = True)

# First variation, use one with stratify with 60:20:20 splitting and scaling
# splitting the data for 60:20:20 ratio between train, validation and test sets
# stratify ensures the training, validation and test sets have a similar distribution of the response variable
# then we split the temporary set into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.25, random_state = Random_State, stratify = y_temp, shuffle = True)

# Drop Customer ID column as no significance to prediction or impact to data analysis
X_train.drop(["CustomerID"], axis=1, inplace=True)
X_val.drop(["CustomerID"], axis=1, inplace=True)
X_test.drop(["CustomerID"], axis=1, inplace=True)

# Drop Unnamed: 0 column as no significance or impact to data analysis
X_train.drop(["Unnamed: 0"], axis=1, inplace=True)
X_val.drop(["Unnamed: 0"], axis=1, inplace=True)
X_test.drop(["Unnamed: 0"], axis=1, inplace=True)

#Consolidate and Combine Fe Male and Female in all the data sets
X_train["Gender"] = X_train["Gender"].str.strip().str.title()
X_train["Gender"] = X_train["Gender"].replace({"Fe Male": "Female"})

X_val["Gender"] = X_val["Gender"].str.strip().str.title()
X_val["Gender"] = X_val["Gender"].replace({"Fe Male": "Female"})

X_test["Gender"] = X_test["Gender"].str.strip().str.title()
X_test["Gender"] = X_test["Gender"].replace({"Fe Male": "Female"})

# Select all categorical and object columns for one-hot encoding
#Cat_Var = X_train.select_dtypes(include=['category', 'object']).columns

#Encoding categorical variables without dropping the first column
#X_train = pd.get_dummies(X_train, columns = Cat_Var, dtype=int)
#X_val   = pd.get_dummies(X_val, columns = Cat_Var, dtype=int)
#X_test  = pd.get_dummies(X_test, columns = Cat_Var, dtype=int)

# Ensure y is DataFrame
#y_train = y_train.to_frame()
#y_val = y_val.to_frame()
#y_test = y_test.to_frame()

# Convert target values (not column names) to integer
y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)

#Convert datasets into csv files
X_train.to_csv("X_train.csv",index=False)
X_val.to_csv("X_val.csv",index=False)
X_test.to_csv("X_test.csv",index=False)
y_train.to_csv("y_train.csv",index=False)
y_val.to_csv("y_val.csv",index=False)
y_test.to_csv("y_test.csv",index=False)

#Checking the files in the folder
def check_dataset_consistency(X_train, X_val, X_test, name_train="Train", name_val="Val", name_test="Test"):
    print("=== DATASET CONSISTENCY CHECK ===\n")

    # SHAPE CHECK
    print("SHAPE")
    print(f"{name_train}: {X_train.shape}")
    print(f"{name_val}:   {X_val.shape}")
    print(f"{name_test}:  {X_test.shape}\n")

    #COLUMN SET CHECK
    print("COLUMN SET DIFFERENCE")

    train_cols = set(X_train.columns)
    val_cols   = set(X_val.columns)
    test_cols  = set(X_test.columns)

    print("Missing in VAL:", train_cols - val_cols)
    print("Extra in VAL:", val_cols - train_cols)
    print()
    print("Missing in TEST:", train_cols - test_cols)
    print("Extra in TEST:", test_cols - train_cols)
    print()

    #ORDER CHECK
    print("COLUMN ORDER CHECK")
    if list(X_train.columns) == list(X_val.columns) == list(X_test.columns):
        print("Okay, ORDER MATCHES for all datasets\n")
    else:
        print("Not Okay, Column order mismatch!\n")

    #DTYPE CHECK
    print("DTYPE CHECK")
    for col in X_train.columns:
        dt_train = X_train[col].dtype
        dt_val = X_val[col].dtype
        dt_test = X_test[col].dtype
        if not (dt_train == dt_val == dt_test):
            print(f"Not Okay, Dtype mismatch in column '{col}': Train={dt_train}, Val={dt_val}, Test={dt_test}")
    print()

    print("CHECK ENDED")

check_dataset_consistency(X_train, X_val, X_test)
check_dataset_consistency(y_train, y_val, y_test)

###Delete Old datasets

from huggingface_hub import HfApi, CommitOperationDelete, CommitOperationAdd

api = HfApi()

repo_id = "samdurai102024/Tourism-Package-Prediction"
#processed_folder = "processed_data/"   # your local folder with split files

#remove old processed files ONLY ---
tree = api.list_repo_tree(repo_id, repo_type="dataset")

delete_ops = []
for file in tree:
    path = file.path

    # Skip raw files (keep them always)
    if path.lower() in ["tourism.csv", "raw_data.csv", "data.csv"]:
        continue

    # Delete only processed files (train/val/test splits)
    if any(x in path.lower() for x in [
        "train", "val", "test", "x_", "y_"
    ]):
        delete_ops.append(CommitOperationDelete(path_in_repo=path))

# Perform delete commit if needed
if delete_ops:
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=delete_ops,
        commit_message="Cleanup old processed dataset before upload"
    )
### end of deletion

#Checking data sets shape before upload
print("LOCAL SHAPES JUST BEFORE UPLOAD")
for f in ["X_train.csv", "X_val.csv", "X_test.csv", "y_train.csv", "y_val.csv", "y_test.csv"]:
    df = pd.read_csv(f)
    print(f, df.shape)

# New set upload and commmit changes
from huggingface_hub import HfApi, CommitOperationAdd

api = HfApi()

import os

base_path = "/content/"   # or wherever your new processed files are saved

files = [
    "X_train.csv", "X_val.csv", "X_test.csv",
    "y_train.csv", "y_val.csv", "y_test.csv"
]

for f in files:
    print("Checking:", f)
    print("CWD:", os.getcwd())
    print("Exists in CWD:", os.path.exists(f))
    print("Absolute path:", os.path.abspath(f))
    print("-" * 50)

from huggingface_hub import HfApi, CommitOperationAdd
import os

#Upload new files
files = ["X_train.csv","X_val.csv","X_test.csv","y_train.csv","y_val.csv","y_test.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj = file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="samdurai102024/Tourism-Package-Prediction",
        repo_type="dataset",
    )

###Commit changes
api = HfApi()

for file_path in files:
    api.create_commit(
        repo_id="samdurai102024/Tourism-Package-Prediction",
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=file_path.split("/")[-1], path_or_fileobj = file_path)],
        commit_message="Updated datasets after change"
    )
