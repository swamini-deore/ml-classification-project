# This file loads CSV data into MySQL with correct TRUE/FALSE conversion

import pandas as pd  # data manipulation
from sqlalchemy import create_engine  # database connection
from dotenv import load_dotenv  # load env variables
import os  # access env variables

load_dotenv()  # load .env file

def get_engine():
    # build mysql connection string
    url = (
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)  # return engine

def reload():
    engine = get_engine()  # get db connection
    df = pd.read_csv("data/online_shoppers_intention.csv")  # read csv file
    # convert Weekend and Revenue to int — handle multiple formats
    df['Weekend'] = df['Weekend'].astype(str).str.strip().str.upper().map({'TRUE': 1, 'FALSE': 0})  # convert Weekend to int
    df['Revenue'] = df['Revenue'].astype(str).str.strip().str.upper().map({'TRUE': 1, 'FALSE': 0})  # convert Revenue to int
    df.to_sql('online_shoppers', engine, if_exists='replace', index=False)  # load to mysql
    print(f"Loaded {len(df)} rows")  # log count
    print(df['Revenue'].value_counts())  # verify Revenue distribution

if __name__ == "__main__":
    reload()  # run reload