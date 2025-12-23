---
layout: post
title: "[MLOps] MLFlow 실습"
date: 2025-12-23 22:29 +0900
description: MLOps를 위한 MLFlow 실습
image:
  path: /assets/img/mlops/project/mlflow/mlflow-logo.png
  alt: MLFlow Logo
category: [MLOps, MLFlow]
tags: [MLOps, MLFlow]
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---
# MLFlow 구축


```python
# MLFlow 설치 및 환경 구축
!pip install mlflow
```

Terminal에 아래와 같이 작성하면 MLFlow 실행
```bash
mlflow ui
```
INFO:waitress:Serving on http://127.0.0.1:5000

위 주소 클릭 -> 실행

## 머신러닝


### 자동 로깅
Sklearn 머신러닝을 사용하는 경우 ```mlflow.sklearn.autolog()```를 사용할 수 있다. 해당 기능은 매개변수를 자동 로깅 통합 기능을 제공하다.

자세한 ```mlflow.autolog()```는 [MLflow Python API](https://mlflow.org/docs/2.21.3/api_reference/python_api/mlflow.html#mlflow.autolog)를 통해 확인할 수 있다.


```python
import mlflow
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# tracing저장 경로
mlflow.set_tracking_uri("file:./mlruns")

# 실험 환경 설정
mlflow.set_experiment("diabetes-rf")

# 자동 로깅
mlflow.sklearn.autolog()

data_diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data_diabetes.data, data_diabetes.target, test_size = 0.2, random_state = 1004)

reg = GridSearchCV(RandomForestRegressor(), {"n_estimators": [50,100], "max_depth": [2,5,10], "max_features": [5,8,10]})
reg.fit(X_train, y_train)

predictions = reg.predict(X_test)
```

    2025/12/22 17:27:09 INFO mlflow.utils.autologging_utils: Created MLflow autologging run with ID 'b854a7f39b034877969c0579904195f9', which will track hyperparameters, performance metrics, model artifacts, and lineage information for the current sklearn workflow
    2025/12/22 17:27:20 INFO mlflow.sklearn.utils: Logging the 5 best runs, 13 runs will be omitted.
    

### 수동 로깅

``mlflow.log_metric()``, ``mlflow.log_metrics()``: metric를 지정해서 로깅, metrics의 경우 dict형태로 저장
``mlflow.log_param()``, ``mlflow.log_params()``: param을 지정해서 로깅, params의 경우 dict형태로 저장


```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_model_with_hyperparameters(n_estimator, max_depth, max_feature):
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth, max_features=max_feature)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_param("n_estimators", n_estimator)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("max_features", max_feature)
        mlflow.log_metric("mae_on_test", mae)
        mlflow.log_metric("mse_on_test", mse)
        mlflow.log_metric("rmse_on_test", mse**0.5)
```


```python
n_estimators = [50,100]
max_depths = [2,5,10]
max_features = [5,8,10]

for n_estimator in n_estimators:
    for max_depth in max_depths:
        for max_feature in max_features:
            train_model_with_hyperparameters(n_estimator, max_depth, max_feature)
```

### Experiments

![log](/assets/img/mlflow/mlflow_prectice/mlflow_log.png)
- 만들어진 log를 확인하여 모델 선택

![register](/assets/img/mlflow/mlflow_prectice/register_model.png)
![create_register](/assets/img/mlflow/mlflow_prectice/create_register.png)
- model을 register에 저장

![models_page](/assets/img/mlflow/mlflow_prectice/model_select.png)
- version1으로 새로운 모델 생성

※ 구버전과 신버전의 차이가 있음

#### 신버전
- ``Tags``와 ``Aliases``로 모델 배포

#### 구버전
- ``stage``의 ``Archived``, ``Staging``, ``Production``으로 관리 및 배포
- ``Archived``: 저장 상태
- ``Staging``: QA 테스트 상태(준비 상태)
- ``Production``: 배포 상태

## Model Serving

- 배포된 모델이 예측 요청을 받고 처리하여 예측 결과를 반환하는 과정을 의미
- 요청 처리와 모델 추론

### 방법 1


```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
```


```python
model_url = "models:/RandomForestRegressor/1"

loaded_model = mlflow.sklearn.load_model(model_url)
```


```python
loaded_model
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(max_depth=5, max_features=8, n_estimators=50)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor(max_depth=5, max_features=8, n_estimators=50)</pre></div> </div></div></div></div>




```python
y_pred = loaded_model.predict(X_test)
mean_squared_error(y_pred, y_test)
```




    3041.9044368258315



### 방법 2

CLI를 통해 모델을 지정된 포트로 서버 통신

```bash
mlflow models serve -m models:/RandomForestRegressor/1 -p 5001 --no-conda
```


```python
import pandas as pd
import requests

host = "127.0.0.1"
url = f"http://{host}:5001/invocations"

X_test_df = pd.DataFrame(X_test, columns=data_diabetes.feature_names)
json_data = {"dataframe_split": X_test_df[:10].to_dict(orient="split")}

response = requests.post(url, json=json_data)
print(f"\nPyfunc 'predict_interval':\n${response.json()}")
```

    
    Pyfunc 'predict_interval':
    ${'predictions': [116.08085576971826, 100.81009133456911, 184.35325437343346, 104.71531717642706, 156.09652462733177, 174.07622265563091, 125.68893595239533, 140.11980756904265, 81.71646235903134, 200.24981584985514]}
    

### Cloud 배포 

#### AWS Sagemaker
```bash
mlflow sagemaker build-and-push-container -m models:/<model_id>
mlflow sagemaker deploy {parameters}
```

#### AzureML
```bash
mlflow azureml export -m {model' path} -o test-output
```

## 딥러닝

``model.pytorch.autolog()``가 가능하지만 수동으로 하는 것을 권장


```python
import mlflow

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import torch
import torch.nn as nn

import pandas as pd
import os
```


```python
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```


```python
mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment("FashionCNN")

# mlflow.pytorch.autolog()
```




    <Experiment: artifact_location=('file:///c:/Users/PC/Desktop/실습/mlruns/107038616652530463'), creation_time=1766485896982, experiment_id='107038616652530463', last_update_time=1766485896982, lifecycle_stage='active', name='FashionCNN', tags={'mlflow.experimentKind': 'custom_model_development'}>




```python
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)
```


```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```

    cuda:0
    


```python
class FashionDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=["file_name", "label"])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```


```python
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```


```python
a = next(iter(train_dataloader))
a[0].size()
```




    torch.Size([64, 1, 28, 28])




```python
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
```


```python
params = {
    "epochs": 5,
    "learning_rate": 0.001,
    "batch_size": 64
}
```


```python
model = FashionCNN()
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
print(model)

```

    FashionCNN(
      (layer1): Sequential(
        (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (layer2): Sequential(
        (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (fc1): Linear(in_features=2304, out_features=600, bias=True)
      (drop): Dropout(p=0.25, inplace=False)
      (fc2): Linear(in_features=600, out_features=120, bias=True)
      (fc3): Linear(in_features=120, out_features=10, bias=True)
    )
    


```python
@torch.no_grad()
def compute_accuracy_from_loader(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        logits = model(data)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return correct / max(total, 1)
```


```python
best_model_id = None
best_test_acc = -1.0

with mlflow.start_run() as run:
    mlflow.log_params(params)

    for epoch in range(params["epochs"]):
        model.train()
        for data, target in train_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                name=f"checkpoint-{epoch}",
                step=epoch,
                pip_requirements=[
                    "torch==2.6.0+cu124",
                    "torchvision==0.21.0+cu124",
                    "numpy",
                ],
            )
        
        train_acc = compute_accuracy_from_loader(model, train_dataloader, device)
        test_acc = compute_accuracy_from_loader(model, test_dataloader, device)

        mlflow.log_metric(
                key="train_accuracy",
                value=train_acc,
                step=epoch,
                model_id=model_info.model_id,
            )
        mlflow.log_metric(
            key="test_accuracy",
            value=test_acc,
            step=epoch,
            model_id=model_info.model_id,
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_id = model_info.model_id

    ranked_checkpoints = mlflow.search_logged_models(
        filter_string=f"source_run_id='{run.info.run_id}'",
        order_by=[{"field_name": "metrics.test_accuracy", "ascending": False}],
        output_format="list",
    )

    best_checkpoint = ranked_checkpoints[0]
    print(f"Best checkpoint: {best_checkpoint.name}")
    print(f"Accuracy: {best_checkpoint.metrics[0].value}")

    best_model_uri = f"models:/{best_model_id}"

    try:
        mv = mlflow.register_model(model_uri=best_model_uri, name="FasionCNN_Model")
        print("FashionCNN_model : ", mv.version)
    except Exception as e:
        print("error : ", e)
```

    Best checkpoint: checkpoint-4
    Accuracy: 0.9048
    FashionCNN_model :  2
    

    Registered model 'FasionCNN_Model' already exists. Creating a new version of this model...
    Created version '2' of model 'FasionCNN_Model'.
    
