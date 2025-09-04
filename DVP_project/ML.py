import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

DATA_FILE = "data/gdp_data.csv"
MODEL_DIR = "models/"

def load_data():
    return pd.read_csv(DATA_FILE)

def train_model(country):
    df = load_data()
    df_country = df[df['Country'] == country]

    X = df_country[['Year']].values
    y = np.log(df_country['GDP'].values) 

    model = LinearRegression()
    model.fit(X, y)

    # Save model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(model, f"{MODEL_DIR}{country}_gdp_model.pkl")
    return model

# Predict future GDP for given years
def predict_future(country, years=5):
    model_path = f"{MODEL_DIR}{country}_gdp_model.pkl"

    if not os.path.exists(model_path):
        model = train_model(country)
    else:
        model = joblib.load(model_path)

    df = load_data()
    df_country = df[df['Country'] == country]
    last_year = df_country['Year'].max()

    future_years = np.array(range(last_year+1, last_year+years+1)).reshape(-1, 1)
    log_preds = model.predict(future_years)
    preds = np.exp(log_preds)  

    return list(zip(future_years.flatten(), preds))

# Get historical data
def get_historical_data(country):
    df = load_data()
    return df[df['Country'] == country][['Year', 'GDP']]


def evaluate_model(country):
    df = load_data()
    df_country = df[df['Country'] == country]

    X = df_country[['Year']].values
    y = np.log(df_country['GDP'].values)  

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log)   
    y_test_orig = np.exp(y_test)

    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# Test with accuracy, but available for DEV
if __name__ == "__main__":
    metrics = evaluate_model("India")
    print("Log-Linear Regression Accuracy:")
    print(f" MAE : {metrics['MAE']:.2e}")
    print(f" RMSE: {metrics['RMSE']:.2e}")
    print(f" R²  : {metrics['R2']:.2f}")
