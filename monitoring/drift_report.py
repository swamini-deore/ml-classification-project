import pandas as pd
import warnings
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

warnings.filterwarnings("ignore")
load_dotenv()

def get_engine():
    url = (
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)

def generate_drift_report():
    engine = get_engine()

    df = pd.read_sql("SELECT * FROM online_shoppers", engine)

    # 🔥 Data cleaning
    df = df.fillna(0)
    df = df.loc[:, df.nunique() > 1]

    reference = df.iloc[:int(len(df) * 0.7)]
    current = df.iloc[int(len(df) * 0.7):]

    print(f"Reference size: {len(reference)} rows")
    print(f"Current size: {len(current)} rows")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    os.makedirs('./monitoring', exist_ok=True)
    report.save_html('./monitoring/drift_report.html')

    print("Drift report saved to ./monitoring/drift_report.html")

if __name__ == "__main__":
    generate_drift_report()