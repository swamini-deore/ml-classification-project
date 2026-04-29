# This file handles feature scaling, selection and class imbalance before training

import pandas as pd  # data manipulation
import numpy as np  # numerical operations
from sqlalchemy import create_engine  # database connection
from sklearn.preprocessing import StandardScaler  # scale numerical features
from sklearn.feature_selection import SelectKBest, f_classif  # select top k important features
from imblearn.over_sampling import SMOTE  # handle class imbalance by oversampling minority class
import pickle  # save scaler and selector to disk
from dotenv import load_dotenv  # load env variables
import os  # access env variables

load_dotenv()  # load .env file

def get_pg_engine():
    # build postgresql connection string for data warehouse
    url = (
        f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}"
        f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_NAME')}"
    )
    return create_engine(url)  # return postgresql engine

def load_transformed_data():
    engine = get_pg_engine()  # get postgresql connection
    df = pd.read_sql("SELECT * FROM transformed_shoppers", engine)  # fetch transformed data from warehouse
    print(f"Loaded {len(df)} rows from PostgreSQL warehouse")  # log count
    return df  # return dataframe

def preprocess(df):
    X = df.drop('Revenue', axis=1)  # features
    y = df['Revenue']  # target

    # check class distribution before SMOTE
    print(f"Before SMOTE — Revenue 0: {(y==0).sum()}, Revenue 1: {(y==1).sum()}")

    # scale numerical features to same range (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # fit and transform features

    # select top 10 most important features using ANOVA f-test
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X_scaled, y)  # fit and select features

    # get selected feature names for reference
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Selected Features: {selected_features}")  # log selected features

    # apply SMOTE to balance classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)  # oversample minority class

    # check class distribution after SMOTE
    print(f"After SMOTE — Revenue 0: {(y_resampled==0).sum()}, Revenue 1: {(y_resampled==1).sum()}")

    # save scaler to disk for use in prediction
    with open('./ml/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)  # serialize scaler

    # save selector to disk for use in prediction
    with open('./ml/selector.pkl', 'wb') as f:
        pickle.dump(selector, f)  # serialize selector

    # save selected feature names for reference in prediction
    with open('./ml/selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)  # serialize feature names

    print("Scaler, Selector saved to ./ml/")  # log save
    return X_resampled, y_resampled  # return preprocessed data

if __name__ == "__main__":
    df = load_transformed_data()  # load data from postgresql
    X, y = preprocess(df)  # run preprocessing
    print(f"Final shape after preprocessing: {X.shape}")  # log final shape