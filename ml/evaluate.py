# This file evaluates the saved model on test data

import pickle  # load saved model
import numpy as np  # numerical operations
from sklearn.metrics import (
    accuracy_score,       # overall accuracy
    precision_score,      # precision score
    recall_score,         # recall score
    f1_score,             # f1 score
    confusion_matrix,     # confusion matrix
    roc_auc_score         # roc auc score
)
from dotenv import load_dotenv  # load env variables
import os  # access env variables
from preprocess import load_transformed_data, preprocess  # import preprocess functions
from sklearn.model_selection import train_test_split  # split data

load_dotenv()  # load .env file

def evaluate():
    # load saved model from disk
    with open(os.getenv('MODEL_PATH'), 'rb') as f:
        model = pickle.load(f)  # deserialize model
    print("Model loaded successfully")

    df = load_transformed_data()  # load transformed data from mysql
    X, y = preprocess(df)  # apply preprocessing

    # same split as train.py to get same test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)  # predict on test data
    y_prob = model.predict_proba(X_test)[:, 1]  # predict probabilities for roc auc

    # calculate all metrics
    acc = accuracy_score(y_test, y_pred)          # accuracy
    pre = precision_score(y_test, y_pred)          # precision
    rec = recall_score(y_test, y_pred)             # recall
    f1  = f1_score(y_test, y_pred)                 # f1 score
    auc = roc_auc_score(y_test, y_prob)            # roc auc
    cm  = confusion_matrix(y_test, y_pred)         # confusion matrix

    # print all metrics
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {pre:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return acc, pre, rec, f1, auc  # return metrics for mlflow tracking

if __name__ == "__main__":
    evaluate()  # run evaluation