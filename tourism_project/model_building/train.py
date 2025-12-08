# for data manipulation
import pandas as pd
import numpy as np
Random_State = 42

from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score, precision_score
from sklearn.metrics import make_scorer

# for creating a folder
import os

# for model serialization
import joblib

#Handling imbalance data with oversampling technique
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# for creating a folder
import os, getpass

# for hugging face space authentication to upload files
from huggingface_hub import hf_hub_download
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Set MLflow tracking URI for PROD (publicly accessible via ngrok)
from pyngrok import ngrok
import mlflow
import time

#Add a simple delay before each MLflow log call
def safe_log_metrics(metric_dict, delay=0.6):
    #Logs multiple metrics to MLflow with a delay to avoid rate limits.
    for name, value in metric_dict.items():
        mlflow.log_metric(name, value)
        time.sleep(delay)

#Add a wrapper around all MLflow calls
def slow_call(fn, *args, delay=0.6, **kwargs):
    result = fn(*args, **kwargs)
    time.sleep(delay)  # delay after EVERY MLflow API call
    return result

# Try to read from environment (GitHub Actions)
ngrok_token = os.getenv("NGROK_TOKEN")

# If running locally / Colab → ask securely
if ngrok_token is None:
    print("NGROK_TOKEN not set. Please enter your Ngrok auth token.")
    ngrok_token = getpass.getpass("Enter Ngrok Token: ")

# Set the token
ngrok.set_auth_token(ngrok_token)

ngrok.kill()

ENV = "prod"

# Start public tunnel (simpler for MLflow)
tunnel = ngrok.connect(5000, proto="http")  # do NOT use private=True
public_url = tunnel.public_url
print("MLflow Public URL (PROD):", public_url)

# Use ngrok http URL for MLflow tracking
mlflow.set_tracking_uri(public_url)
mlflow.set_experiment("Tourism_Experiment_Prod")

print("Tracking URI (prod):", public_url)
print("Experiment: Tourism_Experiment_Prod")

api = HfApi()

# Download the files locally
Xtrain_path = hf_hub_download(
    repo_id="samdurai102024/Tourism-Package-Prediction",
    filename="X_train.csv",
    repo_type="dataset"    # important!
)
Xval_path = hf_hub_download(
    repo_id="samdurai102024/Tourism-Package-Prediction",
    filename="X_val.csv",
    repo_type="dataset"    # important!
)
Xtest_path = hf_hub_download(
    repo_id="samdurai102024/Tourism-Package-Prediction",
    filename="X_test.csv",
    repo_type="dataset"    # important!
)
ytrain_path = hf_hub_download(
    repo_id="samdurai102024/Tourism-Package-Prediction",
    filename="y_train.csv",
    repo_type="dataset"    # important!
)
yval_path = hf_hub_download(
    repo_id="samdurai102024/Tourism-Package-Prediction",
    filename="y_val.csv",
    repo_type="dataset"    # important!
)
ytest_path = hf_hub_download(
    repo_id="samdurai102024/Tourism-Package-Prediction",
    filename="y_test.csv",
    repo_type="dataset"    # important!
)

X_train = pd.read_csv(Xtrain_path)
X_val = pd.read_csv(Xval_path)
X_test = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path)
y_val = pd.read_csv(yval_path)
y_test = pd.read_csv(ytest_path)

# Drop Customer ID column as no significance to prediction or impact to data analysis
#X_train.drop(["CustomerID"], axis=1, inplace=True)
#X_val.drop(["CustomerID"], axis=1, inplace=True)
#X_test.drop(["CustomerID"], axis=1, inplace=True)

# Drop Unnamed: 0 column as no significance or impact to data analysis
#X_train.drop(["Unnamed: 0"], axis=1, inplace=True)
#X_val.drop(["Unnamed: 0"], axis=1, inplace=True)
#X_test.drop(["Unnamed: 0"], axis=1, inplace=True)

#Consolidate and Combine Fe Male and Female in all the data sets
#X_train["Gender"] = X_train["Gender"].str.strip().str.title()
#X_train["Gender"] = X_train["Gender"].replace({"Fe Male": "Female"})

#X_val["Gender"] = X_val["Gender"].str.strip().str.title()
#X_val["Gender"] = X_val["Gender"].replace({"Fe Male": "Female"})

#X_test["Gender"] = X_test["Gender"].str.strip().str.title()
#X_test["Gender"] = X_test["Gender"].replace({"Fe Male": "Female"})

# Convert target values (not column names) to integer
y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)

# Set the class weight to handle class imbalance
class_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
#class_weight

# Define categorical and numerical column names (these are the original, raw column names)
categorical_cols = X_train.select_dtypes(include=['category', 'object']).columns
numeric_cols = X_train.select_dtypes(include=[np.number]).columns

# Define the preprocessing steps using the original categorical and numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight = class_weight, random_state = Random_State)

# Create pipeline using imbpipeline
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('resampler', SMOTE(random_state = Random_State)),  # handles class imbalance
    ('classifier',xgb_model)
])

# Define hyperparameter grid
param_grid ={
  "classifier__max_depth": [5, 6],
  "classifier__gamma": [0.3, 0.4],
  "classifier__learning_rate": [0.1, 0.2],
  "classifier__n_estimators": [800, 1000],
  "classifier__subsample": [0.6, 0.7],
  "classifier__colsample_bytree": [0.7, 0.8],
  "classifier__reg_alpha": [0.1, 0.2],
  "classifier__reg_lambda": [1, 2],
  "classifier__scale_pos_weight": [6]   # keep since it helped recall
}
# Type of scoring used to compare parameter combinations
scorer = make_scorer(f1_score, average='macro')

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, scoring = scorer, cv = 5, n_jobs = -1)
    grid_search.fit(X_train, y_train)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with slow_call(mlflow.start_run, nested=True, delay=0.7):
          slow_call(mlflow.log_params, param_set, delay=0.5)
          slow_call(mlflow.log_metric, "mean_test_score", mean_score, delay=0.5)
          slow_call(mlflow.log_metric, "std_test_score", std_score, delay=0.5)

        #with mlflow.start_run(nested=True):
         #   mlflow.log_params(param_set)
         #   mlflow.log_metric("mean_test_score", mean_score)
         #   mlflow.log_metric("std_test_score", std_score)


    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.40

    y_pred_train_proba = best_model.predict_proba(X_train)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_val_proba = best_model.predict_proba(X_val)[:, 1]
    y_pred_val = (y_pred_val_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(y_train, y_pred_train, output_dict=True)
    val_report = classification_report(y_val, y_pred_val, output_dict=True)
    test_report = classification_report(y_test, y_pred_test, output_dict=True)

    safe_log_metrics({
    "train_accuracy": train_report['accuracy'],
    "train_precision": train_report['1']['precision'],
    "train_recall": train_report['1']['recall'],
    "train_f1_score": train_report['1']['f1-score'],

    "val_accuracy": val_report['accuracy'],
    "val_precision": val_report['1']['precision'],
    "val_recall": val_report['1']['recall'],
    "val_f1_score": val_report['1']['f1-score'],

    "test_accuracy": test_report['accuracy'],
    "test_precision": test_report['1']['precision'],
    "test_recall": test_report['1']['recall'],
    "test_f1_score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_tourism_purchase_prediction_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "samdurai102024/Tourism-Package-Prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

# create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_tourism_purchase_prediction_v1.joblib",
        path_in_repo="best_tourism_purchase_prediction_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
#Commit changes 
    api.create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        operations=[],
        commit_message="Force empty commit",
        )
