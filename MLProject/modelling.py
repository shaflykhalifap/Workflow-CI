import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import os
import mlflow
import mlflow.sklearn

def train_model(data_path):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("Car_Price_Prediction")

    df = pd.read_csv(data_path)
    X = df.drop("Price", axis=1)
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    mlflow.sklearn.autolog()
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path ke dataset")
    args = parser.parse_args()
    train_model(args.data_path)

