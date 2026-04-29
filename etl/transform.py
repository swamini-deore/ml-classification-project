# This file cleans and preprocesses raw data for ML model

import pandas as pd  # data manipulation library
from sklearn.preprocessing import LabelEncoder  # encode categorical columns to numbers

def transform_data(df):
    # drop duplicate rows if any
    df = df.drop_duplicates()

    # drop rows where any value is null
    df = df.dropna()

    # encode Month column (Jan, Feb... → 1, 2...)
    le_month = LabelEncoder()
    df['Month'] = le_month.fit_transform(df['Month'])

    # encode VisitorType column (Returning_Visitor, New_Visitor... → 0, 1...)
    le_visitor = LabelEncoder()
    df['VisitorType'] = le_visitor.fit_transform(df['VisitorType'])

    # convert Weekend column to integer (True → 1, False → 0)
    df['Weekend'] = df['Weekend'].astype(int)

    # convert Revenue column to integer (True → 1, False → 0) — this is our target
    df['Revenue'] = df['Revenue'].astype(int)

    print(f"Transformed data shape: {df.shape}")  # log final shape
    return df  # return cleaned dataframe

if __name__ == "__main__":
    from extract import extract_data  # import extract function
    df = extract_data()  # fetch raw data
    df = transform_data(df)  # apply transformations
    print(df.head())  # print first 5 rows to verify
    print(df.dtypes)  # print column data types to verify encoding