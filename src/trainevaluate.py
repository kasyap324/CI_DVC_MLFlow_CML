import os
import yaml
import dvc.api
import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train_x = pd.read_csv('data1/prepared/train_x.csv')
test_x = pd.read_csv('data1/prepared/test_x.csv')
train_y = pd.read_csv('data1/prepared/train_y.csv')
test_y = pd.read_csv('data1/prepared/test_y.csv')

params = yaml.safe_load(open('params.yaml'))['parameters']
alpha = params['alpha']
l1_ratio = params['l1_ratio']
path = params['path']
repo = params['repo']
version = params['version']


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


print("hello")
mlflow.set_tracking_uri("sqlite:///mlruns.db")
run_name = "Quality Test"
with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_param('path', path)
    mlflow.log_param('data_version', version)
    mlflow.log_param('training_rows', train_x.shape[0])
    mlflow.log_param('training_columns', train_x.shape[1])
    print("input_rows: %s" % train_x.shape[0])
    print("  input_columns: %s" % train_x.shape[1])
    run_id = run.info.run_uuid
    experiment_id = run.info.experiment_id

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    mlflow.sklearn.log_model(lr, "model")

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
