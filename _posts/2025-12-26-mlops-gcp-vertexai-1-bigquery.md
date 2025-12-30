---
layout: post
title: "[MLOps] GCP VertexAI 1(BigQuery)"
date: 2025-12-26 15:38 +0900
description: VertexAI에서 BigQuery의 데이터 불러오기
image: 
  path: /assets/img/mlops/GCP/VertexAI1/bigquery-logo.png
  alt: BigQuery
category: [MLOps, GCP]
tags: [MLOps, GCP, VertexAI, BigQuery]
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---
![BigQuery Logo](/assets/img/mlops/GCP/VertexAI1/bigquery-logo.png)

# BigQuery

- 개요: Serverless로 구성된 대용량 데이터 웨어하우스 및 분석 플랫폼
- 특징
  - SQL 기반의 강력한 쿼리 기능 제공
  - 대용량 데이터셋을 실시간으로 분석할 수 있는 높은 성능
  - 서버리스 아키텍처로 인프라 관리 부담 없음
  - 외부 데이터 소스와 쉽게 통합 가능

## 데이터 분석의 변화에 적합한 구조

![BigQuery Architecture](/assets/img/mlops/GCP/VertexAI1/bigquery_architecture.svg)

- **전통적 데이터베이스의 한계 극복**
  - BigQuery는 **Serverless** 구조로 확장이 쉽고 실시간 데이터 분석 가능
  - 대용량 데이터셋에서도 **높은 성능** 제공

- **인프라 관리의 간소화**
  - 서버 관리 없이 자동 스케일링
  - 대용량 데이터셋에 대해서도 빠르고 효율적인 쿼리 수행 가능

- **다양한 데이터 소스 통합**
  - 여러 데이터 파이프라인 및 Google Cloud 서비스와 자연스럽게 연동
  - 데이터 수집부터 분석까지 하나의 흐름으로 처리 가능

---

## 쉬운 분석과 다양한 데이터 지원

![BigQuery Data Flow](/assets/img/mlops/GCP/VertexAI1/bigquery_flow.jpg)

- **시각화 도구와의 강력한 연동**
  - Google Data Studio, Looker 등과 연계 가능
  - SQL 기반으로 즉각적인 시각화 분석 수행

- **다양한 데이터 포맷 지원**
  - JSON, Avro, Parquet 등 다양한 형식 지원
  - 여러 데이터 소스로부터 유연하게 데이터 수집 가능

- **유연한 스키마 처리**
  - 스키마가 정의되지 않은 데이터도 쿼리 가능
  - 반정형 / 비정형 데이터 분석에 적합

## VertexAI에서 BigQuery 데이터 불러오기

### 방법 1
- SQL query를 통해 데이터를 전처리하고 불러오기

### 방법 2
- bpd 라이브러리(bigframes.pandas)을 사용하여 데이터 불러오기 -> pandas형식으로 전처리

해당 방법은 ``방법1``을 바탕으로 작성하였다.

데이터는 bigquery public에 있는 ``london_bicycles``의 ``cycle_stations``와 ``cycle_hire``를 사용하였다.


```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
import numpy as np
```


```python
from google.cloud.bigquery import Client, QueryJobConfig
client = Client()
```


```python
query="""WITH staging AS (
    SELECT
        STRUCT(
            start_stn.name,
            ST_GEOGPOINT(start_stn.longitude, start_stn.latitude) AS POINT,
            start_stn.docks_count,
            start_stn.install_date
        ) AS starting,
        STRUCT(
            end_stn.name,
            ST_GEOGPOINT(end_stn.longitude, end_stn.latitude) AS point,
            end_stn.docks_count,
            end_stn.install_date
        ) AS ending,
        STRUCT(
            rental_id,
            bike_id,
            duration, --seconds
            ST_DISTANCE(
                ST_GEOGPOINT(start_stn.longitude, start_stn.latitude),
                ST_GEOGPOINT(end_stn.longitude, end_stn.latitude)
            ) AS distance, --meters
            start_date,
            end_date
        ) AS bike
        FROM `bigquery-public-data.london_bicycles.cycle_stations` AS start_stn
        LEFT JOIN `bigquery-public-data.london_bicycles.cycle_hire` AS b
        ON start_stn.id = b.start_station_id
        LEFT JOIN `bigquery-public-data.london_bicycles.cycle_stations` AS end_stn
        on end_stn.id = b.end_station_id
        LIMIT 100000)

SELECT * FROM staging
"""
job = client.query(query)
df = job.to_dataframe()
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>starting</th>
      <th>ending</th>
      <th>bike</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'name': 'New Spring Gardens Walk, Vauxhall', ...</td>
      <td>{'name': 'Broadley Terrace, Marylebone', 'poin...</td>
      <td>{'rental_id': 118913498, 'bike_id': 21845, 'du...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'name': 'New Spring Gardens Walk, Vauxhall', ...</td>
      <td>{'name': 'Moor Street, Soho', 'point': 'POINT(...</td>
      <td>{'rental_id': 106029120, 'bike_id': 12260, 'du...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'name': 'New Spring Gardens Walk, Vauxhall', ...</td>
      <td>{'name': 'Ethelburga Estate, Battersea Park', ...</td>
      <td>{'rental_id': 79563236, 'bike_id': 10009, 'dur...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'name': 'New Spring Gardens Walk, Vauxhall', ...</td>
      <td>{'name': 'Northumberland Avenue, Strand', 'poi...</td>
      <td>{'rental_id': 114066766, 'bike_id': 15247, 'du...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'name': 'New Spring Gardens Walk, Vauxhall', ...</td>
      <td>{'name': 'Doddington Grove, Kennington', 'poin...</td>
      <td>{'rental_id': 78933280, 'bike_id': 9620, 'dura...</td>
    </tr>
  </tbody>
</table>
</div>




```python
values = df["bike"].values

duration = list(map(lambda x: x["duration"], values))
distance = list(map(lambda x: x["distance"], values))
dates = list(map(lambda x: x["start_date"], values))
data = pd.DataFrame(data = {"duration": duration, "distance": distance, "start_date": dates})
```


```python
data = data.dropna()
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration</th>
      <th>distance</th>
      <th>start_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1800.0</td>
      <td>5051.961504</td>
      <td>2022-04-13 17:09:00+00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>780.0</td>
      <td>2904.863984</td>
      <td>2021-03-14 20:37:00+00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>780.0</td>
      <td>3136.325002</td>
      <td>2018-08-22 18:33:00+00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>780.0</td>
      <td>2109.218317</td>
      <td>2021-10-25 08:46:00+00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>780.0</td>
      <td>1371.888510</td>
      <td>2018-08-05 13:55:00+00:00</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 94480 entries, 0 to 99999
    Data columns (total 3 columns):
     #   Column      Non-Null Count  Dtype              
    ---  ------      --------------  -----              
     0   duration    94480 non-null  float64            
     1   distance    94480 non-null  float64            
     2   start_date  94480 non-null  datetime64[ns, UTC]
    dtypes: datetime64[ns, UTC](1), float64(2)
    memory usage: 2.9 MB
    


```python
# start_date -> weekday, hour
# duration -> minute

data["weekday"] = data["start_date"].apply(lambda x: x.weekday())
data["hour"] = data["start_date"].apply(lambda x: x.time().hour)
data.drop(columns=["start_date"], inplace=True)
```


```python
data["duration"] = data["duration"].apply(lambda x: float(x/60))
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration</th>
      <th>distance</th>
      <th>weekday</th>
      <th>hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30.0</td>
      <td>5051.961504</td>
      <td>2</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.0</td>
      <td>2904.863984</td>
      <td>6</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.0</td>
      <td>3136.325002</td>
      <td>2</td>
      <td>18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.0</td>
      <td>2109.218317</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.0</td>
      <td>1371.888510</td>
      <td>6</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
# weekday, hour -> one-hot
# distance -> Normalization

data = pd.get_dummies(data, columns = ["weekday", "hour"], prefix = ["weekday", "hour"])
```


```python
X = data.drop(["duration"],axis=1).to_numpy()
y = data["duration"].to_numpy().reshape(-1,1)
```

## MLP 테스트

bigquery에서 추출한 데이터를 MLP모델에 학습하였다.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, random_state=1004)
```


```python
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)
```


```python
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size = 256, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size = 256, shuffle=False
)
```


```python
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layer(x)
```


```python
in_dim = X_train.size()[1]
model = MLP(in_dim)
```


```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```


```python
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    return total_loss / max(n, 1)
```


```python
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss, n = 0.0, 0

    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
