
# import os library
import os

# for data manipulation
import numpy as np
import pandas as pd
from pprint import pprint

import sklearn

# libraries for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

# libraries for encoding, used during pre-processing
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, LabelEncoder

# libraries for column transformer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer

# for data preprocessing and pipeline creation
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

# for model training, tuning, and evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

import xgboost as xgb

# Libraries for measuring scores
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix,
    accuracy_score
)

# for model serialization
import joblib

# import mlflow for expermemintation and logging
import mlflow
import mlflow.sklearn


# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError


# -------------------------
# 1. Configure parameters
# -------------------------
DATA_PATH = "tourism_project/data/tourism.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20   # final test split
MLFLOW_EXPERIMENT = "MLOps_Tourism_experiment"


# initialize mlflow - set tracking uri and experiment name
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(MLFLOW_EXPERIMENT)


# --------------------------------------------------
# 2. Load Train and test data from Hugging Face Space
# --------------------------------------------------

# Initialize Hiugging Face API
api = HfApi()

# update path variables with data paths for train and test data sets
Xtrain_path = "hf://datasets/harishsohani/MLOP-Project-Tourism/Xtrain.csv"
Xtest_path = "hf://datasets/harishsohani/MLOP-Project-Tourism/Xtest.csv"
ytrain_path = "hf://datasets/harishsohani/MLOP-Project-Tourism/ytrain.csv"
ytest_path = "hf://datasets/harishsohani/MLOP-Project-Tourism/ytest.csv"

# load train and test data
X_train = pd.read_csv(Xtrain_path)
X_test = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path)
y_test = pd.read_csv(ytest_path)

# print shape of train and test data (input variables)
print("Shapes: X_train", X_train.shape, "X_test", X_train.shape, "X_test", X_test.shape)

# --------------------------------------------------
# 3. Group t he features based on their nature
#
#    Here we separate the variavables into different
#    groups for preprocessing
#
# --------------------------------------------------

# define list with categorical variables
category_features  = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]

# define Ordinal features
ordinal_features = [
    "CityTier",                 # Defines Tier with values ranging from 1 to 3, where 1 > 2 > 3
    "PreferredPropertyStar",    # Defines Preferred Property Rating  with values ranging from 5 to 3, where 5 > 4 > 3
    "PitchSatisfactionScore"    # Defines Sales Pitch satisfaction score with ranging from 5 to 1, where 5 > 4 > 3 > 2 > 1
]

# umeric variables with binary values (0 and 1)
binary_numeric = ["Passport", "OwnCar"]

# numberic variables which are continuous in nature
continuous_numeric  = [
    "Age",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "MonthlyIncome",
    "NumberOfChildrenVisiting",
    "NumberOfTrips",
]

# following list combines all numeric features into single
numeric_cols = ordinal_features + binary_numeric + continuous_numeric

# Define target variable
target_col = 'ProdTaken'

# Ensure ordinals are proper dtype (int)
for col in ordinal_features:
    # If categorical strings exist, try to coerce numeric
    tourism_df[col] = pd.to_numeric(tourism_df[col], errors="coerce").astype("Int64")



# --------------------------------------------------
# 4. Compute scale_pos_weight for XGBoost
#    Note: data set is imbalanced
# --------------------------------------------------
# Corrected calculation: neg should be count of target's negative class
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print("scale_pos_weight:", scale_pos_weight)



# -------------------------
# 5. Build preprocessing
# -------------------------

# Explicit categories for OrdinalEncoder (must match the data values and order)
ordinal_categories = [
    [1, 2, 3],             # CityTier
    [3, 4, 5],             # PreferredPropertyStar (if only 3..5)
    [1, 2, 3, 4, 5]        # PitchSatisfactionScore
]

#define ordinal encoder
ordinal_encoder = OrdinalEncoder(categories=ordinal_categories, dtype=int)

#define category encoder
category_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# define preprocessor with columns transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("ordinal", ordinal_encoder, ordinal_features),
        ("categorical", category_encoder, category_features),
        ("binary", "passthrough", binary_numeric),
        ("continuous", "passthrough", continuous_numeric)
    ],
    remainder="drop",
    verbose_feature_names_out=False  # gives nicer feature names from transformers
)



# ----------------------------
# 6. Build model and pipeline
# ----------------------------
xgb_model  = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",   # recommended for newer xgboost versions
    scale_pos_weight=scale_pos_weight, # This will now be a single float
    random_state=RANDOM_STATE,
    tree_method="hist",      # faster for larger data; keep XGBoost deterministic-ish
)

pipeline = make_pipeline(preprocessor, xgb_model)



# ----------------------------------------
# 7. Define RandomizedSearchCV parameters
# ----------------------------------------
param_grid = {
    "xgbclassifier__n_estimators": [50, 100, 150],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__max_depth": [3, 4, 5],
    "xgbclassifier__min_child_weight": [1, 3, 5],
    "xgbclassifier__subsample": [0.7, 0.85, 1.0],
    "xgbclassifier__colsample_bytree": [0.6, 0.8, 1.0],
    "xgbclassifier__gamma": [0, 1],
    "xgbclassifier__reg_lambda": [0.5, 1.0, 2.0]
}

# CV strategy
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)


# --------------------------------------
# 8. Run RandomizedSearchCV with MLFlow
# --------------------------------------

with mlflow.start_run(run_name="random_search_xgb_pipeline"):
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=50,  # fewer iterations for faster runtime
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=RANDOM_STATE,
        return_train_score=True
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    mlflow.log_params(best_params)
    pprint(best_params)

    # -------------------------
    # Evaluate best model
    # -------------------------
    best_pipeline = random_search.best_estimator_
    THRESHOLD = 0.45

    def eval_and_log(X_, y_, dataset_name, threshold=THRESHOLD):
        y_proba = best_pipeline.predict_proba(X_)[:,1]
        y_pred = (y_proba >= threshold).astype(int)
        acc = accuracy_score(y_, y_pred)
        auc = roc_auc_score(y_, y_proba)
        report = classification_report(y_, y_pred, output_dict=True)
        cm = confusion_matrix(y_, y_pred)

        mlflow.log_metric(f"{dataset_name}_accuracy", float(acc))
        mlflow.log_metric(f"{dataset_name}_roc_auc", float(auc))
        mlflow.log_metric(f"{dataset_name}_threshold", float(threshold))

        if "1" in report:
            mlflow.log_metric(f"{dataset_name}_precision_pos", float(report["1"]["precision"]))
            mlflow.log_metric(f"{dataset_name}_recall_pos", float(report["1"]["recall"]))
            mlflow.log_metric(f"{dataset_name}_f1_pos", float(report["1"]["f1-score"]))

        print(f"\n{dataset_name} - acc: {acc:.4f} | roc_auc: {auc:.4f} | threshold={threshold}")
        print("confusion_matrix:\n", cm)
        print("classification_report:\n", classification_report(y_, y_pred))
        return {"acc": acc, "auc": auc, "cm": cm, "report": report}

    train_metrics = eval_and_log(X_train, y_train, "train")
    test_metrics = eval_and_log(X_test, y_test, "test")

    # Save the model locally
    model_path = "best_tourism_model.joblib"
    joblib.dump(best_pipeline, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "harishsohani/MLOP-Project-Tourism"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # Upload model
    api.upload_file(
        path_or_fileobj="best_tourism_model.joblib",
        path_in_repo="best_tourism_model.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
