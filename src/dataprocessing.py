import os
import yaml
import dvc.api
import pandas as pd
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open('params.yaml'))['parameters']
path = params['path']
repo = params['repo']
version = params['version']
data_url = dvc.api.get_url(
    path=path, 
    repo=repo,
    rev=version
)
data_path = os.path.join('data1', 'prepared')
os.makedirs(data_path, exist_ok=True)

data = pd.read_csv(data_url, sep=",")
train, test = train_test_split(data)
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]
pd.DataFrame(train_x).to_csv(os.path.join(data_path, "train_x.csv"))
pd.DataFrame(train_y).to_csv(os.path.join(data_path, "train_y.csv"))
pd.DataFrame(test_x).to_csv(os.path.join(data_path, "test_x.csv"))
pd.DataFrame(test_y).to_csv(os.path.join(data_path, "test_y.csv"))

