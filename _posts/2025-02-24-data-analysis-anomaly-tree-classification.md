---
layout: post
title: "[Data Analysis] 트리 & 분류 기반 이상 탐지"
date: 2025-02-24 18:37 +0900
description: 트리 및 분류 기반 이상탐지를 진행하였습니다.
category: [Data Analysis, Anomaly Detection]
tags: [Data Analysis, Anomaly Detection, Tree, Classification]
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---

## Isolation Forest 이상 탐지

임의의 변수에 임의의 값을 사용하여 split해 내가 isolation하고 싶은 객체가 존재하지 않는 부분을 버리는 방식



- 특정 한 개체가 isolation 되는 leaf 노드(terminal node)까지의 거리를 anomaly score로 정의
- 그 평균거리(depth)가 짧을 수록 anomaly score는 높아짐
<br>
<br>
- split이 적으면 anomaly score은 커짐
- split이 많으면 anomaly score은 작아짐


**Anomaly Score 계산**

Anomaly Score는 다음과 같은 수식으로 계산된다.:

![anomaly-score](/assets/img/data_analysis/anomaly_detection/tree-classification/anomaly-score.svg)

여기서 E(h(x))는 각 데이터 포인트의 평균 경로 길이, c(n)은 전체 데이터의 평균 경로 길이를 보정하는 상수


<img src = 'https://i.imgur.com/JVK1ZPt.png' width = 2000 >

<img src = 'https://i.imgur.com/rzP7siS.png'>

<img src = 'https://i.imgur.com/JRoRAA0.png'><br>
- 트리 높이의 제한을 엄격하게 줄때(
h
l
i
m
=
1
) 정상분포와 비정상분포의 스코어가 유사함
- 트리 높이의 제한을 덜 줄때(
h
l
i
m
=
6
) 정상분포(0.45)와 비정상분포(0.55)의 스코어가 적절히 구분되어 표기됨

- 장점
    - 군집기반 이상탐지 알고리즘에 비해 계산량이 매우 적음(smapling을 사용해서 트리를 생성)
    - 이상치가 포함되지 않아도 동작함
    - 비지도 학습이 가능하다
- 단점
    - 수직과 수평으로 분리하기 때문에 잘못된 scoring이 발생할 수 있음


```python
from sklearn.ensemble import IsolationForest

import pandas as pd
import numpy as np
```


```python
np.random.seed(42)
n_samples = 200
n_features = 2

X_normal = np.random.normal(loc=[0,0], scale=[1,1], size=(n_samples, n_features))

X_outlier = np.random.normal(loc=[5, 5], scale=[1, 1], size=(int(n_samples * 0.1), n_features))

X = np.vstack((X_normal, X_outlier))
```


```python
# contamination : 이상치의 비율 설정(default = 'auto')
iforest = IsolationForest(contamination=0.1, random_state=42)
iforest.fit(X)

# 예측
y_pred = iforest.predict(X)
```


```python
import matplotlib.pyplot as plt
%matplotlib inline

# 이상치와 정상 데이터를 색상으로 구분하여 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='blue', label='Inliers')
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', label='Outliers')
plt.title('Isolation Forest')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```


    
![png](/assets/img/data_analysis/anomaly_detection/tree-classification/isolation-forest-plot.png)
    


## Extended Isolation Forest 이상 탐지

Isolation Forest기반으로 확장된 알고리즘으로 수직과 평면으로 분기를 나누는 것을 넘어 초평면을 사용하여 데이터를 나누는 방식

![isolation-data](/assets/img/data_analysis/anomaly_detection/tree-classification/isolation-data.png)

위 데이터를 분할하는데 기존의 isolation은 수직과 수평으로 분할하면 아래와 같다.

![isolation-base-split](/assets/img/data_analysis/anomaly_detection/tree-classification/isolation-base-split.png)

이는 수직과 수평이 교차하여 데이터가 없는 곳에 낮은 이상 점수 영역을 만드는 문제가 발생하게 된다.

![extended-isolation-split](/assets/img/data_analysis/anomaly_detection/tree-classification/extended-isolation-split.png)

![extended-isolation-base](/assets/img/data_analysis/anomaly_detection/tree-classification/extended-isolation-base.png)

이에 Extended Isolation Forest는 분기 과정을 모든 방향으로 발생시키도록하여 지역을 훨씬 더 균일하게 나뉘게 할 수 있었으며, 데이터가 없는 곳에 이상 점수 영역을 만들지 않게 만들 수 있게 되었다.

- 장점
    - 복잡한 데이터 구조에서 isolation forest보다 잘 탐지함
    - isolation forest와 비슷하게 속도저하가 없음
- 단점
    - hyper parameter를 많이 조정해야 함.


```python
import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator

# H2O 초기화
h2o.init()
```

    Checking whether there is an H2O instance running at http://localhost:54321..... not found.
    Attempting to start a local H2O server...
      Java Version: openjdk version "11.0.26" 2025-01-21; OpenJDK Runtime Environment (build 11.0.26+4-post-Ubuntu-1ubuntu122.04); OpenJDK 64-Bit Server VM (build 11.0.26+4-post-Ubuntu-1ubuntu122.04, mixed mode, sharing)
      Starting server from /usr/local/lib/python3.11/dist-packages/h2o/backend/bin/h2o.jar
      Ice root: /tmp/tmp_xjhwin6
      JVM stdout: /tmp/tmp_xjhwin6/h2o_unknownUser_started_from_python.out
      JVM stderr: /tmp/tmp_xjhwin6/h2o_unknownUser_started_from_python.err
      Server is running at http://127.0.0.1:54321
    Connecting to H2O server at http://127.0.0.1:54321 ... successful.
    Warning: Your H2O cluster version is (3 months and 22 days) old.  There may be a newer version available.
    Please download and install the latest version from: https://h2o-release.s3.amazonaws.com/h2o/latest_stable.html
    



<style>

#h2o-table-1.h2o-container {
  overflow-x: auto;
}
#h2o-table-1 .h2o-table {
  /* width: 100%; */
  margin-top: 1em;
  margin-bottom: 1em;
}
#h2o-table-1 .h2o-table caption {
  white-space: nowrap;
  caption-side: top;
  text-align: left;
  /* margin-left: 1em; */
  margin: 0;
  font-size: larger;
}
#h2o-table-1 .h2o-table thead {
  white-space: nowrap; 
  position: sticky;
  top: 0;
  box-shadow: 0 -1px inset;
}
#h2o-table-1 .h2o-table tbody {
  overflow: auto;
}
#h2o-table-1 .h2o-table th,
#h2o-table-1 .h2o-table td {
  text-align: right;
  /* border: 1px solid; */
}
#h2o-table-1 .h2o-table tr:nth-child(even) {
  /* background: #F5F5F5 */
}

</style>      
<div id="h2o-table-1" class="h2o-container">
  <table class="h2o-table">
    <caption></caption>
    <thead></thead>
    <tbody><tr><td>H2O_cluster_uptime:</td>
<td>06 secs</td></tr>
<tr><td>H2O_cluster_timezone:</td>
<td>Etc/UTC</td></tr>
<tr><td>H2O_data_parsing_timezone:</td>
<td>UTC</td></tr>
<tr><td>H2O_cluster_version:</td>
<td>3.46.0.6</td></tr>
<tr><td>H2O_cluster_version_age:</td>
<td>3 months and 22 days</td></tr>
<tr><td>H2O_cluster_name:</td>
<td>H2O_from_python_unknownUser_vpnhj2</td></tr>
<tr><td>H2O_cluster_total_nodes:</td>
<td>1</td></tr>
<tr><td>H2O_cluster_free_memory:</td>
<td>3.170 Gb</td></tr>
<tr><td>H2O_cluster_total_cores:</td>
<td>2</td></tr>
<tr><td>H2O_cluster_allowed_cores:</td>
<td>2</td></tr>
<tr><td>H2O_cluster_status:</td>
<td>locked, healthy</td></tr>
<tr><td>H2O_connection_url:</td>
<td>http://127.0.0.1:54321</td></tr>
<tr><td>H2O_connection_proxy:</td>
<td>{"http": null, "https": null, "colab_language_server": "/usr/colab/bin/language_service"}</td></tr>
<tr><td>H2O_internal_security:</td>
<td>False</td></tr>
<tr><td>Python_version:</td>
<td>3.11.11 final</td></tr></tbody>
  </table>
</div>




```python
h2o_df = h2o.import_file("https://raw.github.com/h2oai/h2o/master/smalldata/logreg/prostate.csv")

predictors = ["AGE","RACE","DPROS","DCAPS","PSA","VOL","GLEASON"]

# Extended Isolation Forest 모델 생성 및 학습
eif = H2OExtendedIsolationForestEstimator(model_id = "eif.hex",
                                          ntrees = 100,
                                          sample_size = 256,
                                          extension_level = len(predictors) - 1)

eif.train(x=predictors, training_frame=h2o_df)

# 예측
y_pred = eif.predict(h2o_df).as_data_frame()
```

    Parse progress: |████████████████████████████████████████████████████████████████| (done) 100%
    extendedisolationforest Model Build progress: |██████████████████████████████████| (done) 100%
    extendedisolationforest prediction progress: |███████████████████████████████████| (done) 100%
    

    /usr/local/lib/python3.11/dist-packages/h2o/frame.py:1983: H2ODependencyWarning: Converting H2O frame to pandas dataframe using single-thread.  For faster conversion using multi-thread, install polars and pyarrow and use it as pandas_df = h2o_df.as_data_frame(use_multi_thread=True)
    
      warnings.warn("Converting H2O frame to pandas dataframe using single-thread.  For faster conversion using"
    


```python
anomaly_score = y_pred["anomaly_score"]

# 데이터셋과 이상치 점수 결합
prostate_df = h2o_df.as_data_frame()
prostate_df['anomaly_score'] = anomaly_score
```

    /usr/local/lib/python3.11/dist-packages/h2o/frame.py:1983: H2ODependencyWarning: Converting H2O frame to pandas dataframe using single-thread.  For faster conversion using multi-thread, install polars and pyarrow and use it as pandas_df = h2o_df.as_data_frame(use_multi_thread=True)
    
      warnings.warn("Converting H2O frame to pandas dataframe using single-thread.  For faster conversion using"
    


```python
prostate_df.head(5)
```





  <div id="df-30c74582-d613-45dd-b1e5-0d06059ec8b6" class="colab-df-container">
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
      <th>ID</th>
      <th>CAPSULE</th>
      <th>AGE</th>
      <th>RACE</th>
      <th>DPROS</th>
      <th>DCAPS</th>
      <th>PSA</th>
      <th>VOL</th>
      <th>GLEASON</th>
      <th>anomaly_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>65</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1.4</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.379427</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>72</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>6.7</td>
      <td>0.0</td>
      <td>7</td>
      <td>0.371727</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>70</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>4.9</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.380674</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>76</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>51.2</td>
      <td>20.0</td>
      <td>7</td>
      <td>0.492321</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>69</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>12.3</td>
      <td>55.9</td>
      <td>6</td>
      <td>0.449620</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-30c74582-d613-45dd-b1e5-0d06059ec8b6')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-30c74582-d613-45dd-b1e5-0d06059ec8b6 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-30c74582-d613-45dd-b1e5-0d06059ec8b6');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-4293bff0-c298-4b61-b796-4cc62fa91229">
  <button class="colab-df-quickchart" onclick="quickchart('df-4293bff0-c298-4b61-b796-4cc62fa91229')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-4293bff0-c298-4b61-b796-4cc62fa91229 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
plt.figure(figsize=(8, 6))
plt.scatter(prostate_df.loc[prostate_df['anomaly_score'] < 0.5, 'PSA'], prostate_df.loc[prostate_df['anomaly_score'] < 0.5, 'VOL'], c='blue', label='Inliers')
plt.scatter(prostate_df.loc[prostate_df['anomaly_score'] >= 0.5, 'PSA'], prostate_df.loc[prostate_df['anomaly_score'] >= 0.5, 'VOL'], c='red', label='Outliers')
plt.title('Extended Isolation Forest')
plt.xlabel('PSA')
plt.ylabel('VOL')
plt.legend()
plt.show()
```


    
![png](/assets/img/data_analysis/anomaly_detection/tree-classification/extended-isolation-plot.png)
    


## One-Class SVM 이상 탐지

데이터를 N차원 좌표측으로 표현하고, 원점과의 거리를 기준으로 초 평면을 그어 Classification하는 방법

1. 비지도 학습: One-Class SVM은 정상 데이터만을 사용하여 학습되며, 이상치의 라벨이 필요하지 않습니다.

2. 커널 사용: One-Class SVM은 다양한 커널을 사용하여 비선형 데이터를 선형적으로 변환할 수 있습니다. 주로 'rbf' 커널이 사용됩니다.

3. 서포트 벡터: One-Class SVM은 서포트 벡터를 사용하여 초평면을 정의합니다. 서포트 벡터는 초평면에 가장 가까운 데이터 포인트입니다.

4. 마진 최대화: One-Class SVM은 서포트 벡터와 초평면 사이의 거리, 즉 마진을 최대화하는 초평면을 찾습니다.

![one-class-svm](/assets/img/data_analysis/anomaly_detection/tree-classification/one-class-svm.png)

- 장점
    - 비지도 학습이 가능
    - 적은 데이터량으로 학습해도 일반화 능력이 좋음
- 단점
    - 데이터 량이 늘어날 수록 연산량이 크게 증가함
    - 데이터 스케일링에 민감함
    - Hyper parameter를 잘 조절해야 함



```python
from sklearn.svm import OneClassSVM

# kernel: 사용할 커널 타입을 지정합니다. ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
# nu: 훈련 오류의 비율 상한 및 서포트 벡터의 비율 하한입니다. (0, 1] 구간이어야 합니다.
# gamma: 'rbf', 'poly', 'sigmoid' 커널의 계수입니다. 'scale'은 1 / (n_features * X.var()), 'auto'는 1 / n_features를 사용합니다.
clf = OneClassSVM(kernel='rbf', nu=0.1, gamma='auto')
clf.fit(X)

y_pred = clf.predict(X)
```


```python
plt.figure(figsize=(8, 6))
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='blue', label='Inliers')
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', label='Outliers')
plt.title('One-Class SVM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```


    
![png](/assets/img/data_analysis/anomaly_detection/tree-classification/one-class-svm-plot.png)
    

&nbsp;

- 참고자료
    - [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
    - [https://en.wikipedia.org/wiki/Isolation_forest](https://en.wikipedia.org/wiki/Isolation_forest)
    - [https://github.com/sahandha/eif](https://github.com/sahandha/eif)
    - [https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/eif.html](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/eif.html)
    - [https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
    - [https://losskatsu.github.io/machine-learning/oneclass-svm/#2-one-class-svm%EC%9D%98-%EB%AA%A9%EC%A0%81](https://losskatsu.github.io/machine-learning/oneclass-svm/#2-one-class-svm%EC%9D%98-%EB%AA%A9%EC%A0%81)