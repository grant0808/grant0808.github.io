---
layout: post
title: "[Data Analysis] Autoencoder 이상 탐지"
date: 2025-03-05 20:10 +0900
description: Autoencoder로 이상탐지 실습
category: [Data Analysis, Anomaly Detection]
tags: [Data Analysis, Anomaly Detection, Autoencoder]
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---

## Autoencoder 이상 탐지
Autoencoder란 입력 데이터를 주요 특징으로 효율적으로 압축(인코딩)한 후 이 압축된 표현에서 원본 입력을 재구성(디코딩)하는 딥러닝 기술입니다.

 이를 통해 비정상적인 데이터가 입력되면 정상데이터만 학습한 Autoencoder는 noise부분은 제외하고 데이터를 출력할 것입니다. 출력된 데이터가 입력된 데이터와의 차이가 커지게 될 것이고 loss(=MSE)가 커져 threshold가 넘으면 비정상으로 판단할 것입니다.

![autoencoder](/assets/img/data_analysis/anomaly_detection/autoencoder/autoencoder.svg)

위 그림과 같이 정상적인 데이터만 학습한 모델에 정상적인 데이터를 입력하고 출력값을 비교하면 Error가 크지 않는 것을 볼 수 있습니다.

![autoencoder-anomaly](/assets/img/data_analysis/anomaly_detection/autoencoder/autoencoder-anomaly.svg)

반면, 비정상적인 데이터를 입력하고 출력값을 비교하면 Error가 크게 띄는 것을 확인할 수 있습니다.

### Autoencoder 구조 요소

- 인코더(Encoder) : 차원 감소를 통해 입력 데이터의 압축된 표현을 인코딩하는 레이어로 구성됩니다. 일반적인 오토인코더에서 신경망의 숨겨진 레이어는 입력 레이어보다 점점 더 적은 수의 노드를 포함하며 데이터가 인코더 레이어를 통과할 때 더 작은 차원으로 '압축'되는 과정을 통해 압축됩니다.

- 병목 현상(Bottleneck) : 인코더 네트워크의 출력 레이어이자 디코더 네트워크의 입력 레이어로, 입력을 가장 압축적으로 표현한 것입니다. 오코인코더의 설계 및 학습의 기본 목표는 입력 데이터를 효과적으로 재구성하는 데 필요한 최소한의 중요한 특징(또는 차원)을 발견하는 것입니다. 그런 다음 이 계층에서 나타나는 잠재 공간 표현, 즉 코드가 디코더에 입력됩니다.

- 디코더(Decoder) : 인코딩된 데이터 표현을 압축 해제(또는 디코딩)하여 궁극적으로 데이터를 인코딩 전의 원본 형태로 재구성하며, 점진적으로 더 많은 수의 노드가 있는 숨겨진 레이어로 구성됩니다. 그런 다음 이렇게 재구성된 출력을 '근거가 되는 진실(대부분의 경우 단순히 원본 입력)'과 비교하여 오토인코더의 효율성을 측정합니다. 출력과 근거 진실의 차이를 재구성 오류라고 합니다.

- 장점
    - 데이터의 Lable이 존재하지 않아도 사용 가능
    - 고차원 데이터의 특징 추출 가능
    - Autoencoder기반 다양한 알고리즘 존재(ex. 희소 오토인코더, 합성곱 오토인코더, 잡음 제거 오토인코더 등)

- 단점
    - Hyper parameter(hidden layer) 설정이 어려움
    - Loss(MSE)에 대한 threshold 설정이 어려움

## Autoencoder 실습


```python
import pandas as pd
import numpy as np

import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "colab"

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

from fastprogress import master_bar, progress_bar

from IPython.display import display
import random
```


```python
SEED = 7

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

np.random.seed(SEED)
random.seed(SEED)
```


```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
```

    cuda
    


```python
data = pd.read_csv("creditcard.csv")
data
```





  <div id="df-2ee9df68-d2b5-4001-99bd-2d1983cf6f66" class="colab-df-container">
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>0.090794</td>
      <td>-0.551600</td>
      <td>-0.617801</td>
      <td>-0.991390</td>
      <td>-0.311169</td>
      <td>1.468177</td>
      <td>-0.470401</td>
      <td>0.207971</td>
      <td>0.025791</td>
      <td>0.403993</td>
      <td>0.251412</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>-0.166974</td>
      <td>1.612727</td>
      <td>1.065235</td>
      <td>0.489095</td>
      <td>-0.143772</td>
      <td>0.635558</td>
      <td>0.463917</td>
      <td>-0.114805</td>
      <td>-0.183361</td>
      <td>-0.145783</td>
      <td>-0.069083</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>0.207643</td>
      <td>0.624501</td>
      <td>0.066084</td>
      <td>0.717293</td>
      <td>-0.165946</td>
      <td>2.345865</td>
      <td>-2.890083</td>
      <td>1.109969</td>
      <td>-0.121359</td>
      <td>-2.261857</td>
      <td>0.524980</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>-0.054952</td>
      <td>-0.226487</td>
      <td>0.178228</td>
      <td>0.507757</td>
      <td>-0.287924</td>
      <td>-0.631418</td>
      <td>-1.059647</td>
      <td>-0.684093</td>
      <td>1.965775</td>
      <td>-1.232622</td>
      <td>-0.208038</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>0.753074</td>
      <td>-0.822843</td>
      <td>0.538196</td>
      <td>1.345852</td>
      <td>-1.119670</td>
      <td>0.175121</td>
      <td>-0.451449</td>
      <td>-0.237033</td>
      <td>-0.038195</td>
      <td>0.803487</td>
      <td>0.408542</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>284802</th>
      <td>172786.0</td>
      <td>-11.881118</td>
      <td>10.071785</td>
      <td>-9.834783</td>
      <td>-2.066656</td>
      <td>-5.364473</td>
      <td>-2.606837</td>
      <td>-4.918215</td>
      <td>7.305334</td>
      <td>1.914428</td>
      <td>4.356170</td>
      <td>-1.593105</td>
      <td>2.711941</td>
      <td>-0.689256</td>
      <td>4.626942</td>
      <td>-0.924459</td>
      <td>1.107641</td>
      <td>1.991691</td>
      <td>0.510632</td>
      <td>-0.682920</td>
      <td>1.475829</td>
      <td>0.213454</td>
      <td>0.111864</td>
      <td>1.014480</td>
      <td>-0.509348</td>
      <td>1.436807</td>
      <td>0.250034</td>
      <td>0.943651</td>
      <td>0.823731</td>
      <td>0.77</td>
      <td>0</td>
    </tr>
    <tr>
      <th>284803</th>
      <td>172787.0</td>
      <td>-0.732789</td>
      <td>-0.055080</td>
      <td>2.035030</td>
      <td>-0.738589</td>
      <td>0.868229</td>
      <td>1.058415</td>
      <td>0.024330</td>
      <td>0.294869</td>
      <td>0.584800</td>
      <td>-0.975926</td>
      <td>-0.150189</td>
      <td>0.915802</td>
      <td>1.214756</td>
      <td>-0.675143</td>
      <td>1.164931</td>
      <td>-0.711757</td>
      <td>-0.025693</td>
      <td>-1.221179</td>
      <td>-1.545556</td>
      <td>0.059616</td>
      <td>0.214205</td>
      <td>0.924384</td>
      <td>0.012463</td>
      <td>-1.016226</td>
      <td>-0.606624</td>
      <td>-0.395255</td>
      <td>0.068472</td>
      <td>-0.053527</td>
      <td>24.79</td>
      <td>0</td>
    </tr>
    <tr>
      <th>284804</th>
      <td>172788.0</td>
      <td>1.919565</td>
      <td>-0.301254</td>
      <td>-3.249640</td>
      <td>-0.557828</td>
      <td>2.630515</td>
      <td>3.031260</td>
      <td>-0.296827</td>
      <td>0.708417</td>
      <td>0.432454</td>
      <td>-0.484782</td>
      <td>0.411614</td>
      <td>0.063119</td>
      <td>-0.183699</td>
      <td>-0.510602</td>
      <td>1.329284</td>
      <td>0.140716</td>
      <td>0.313502</td>
      <td>0.395652</td>
      <td>-0.577252</td>
      <td>0.001396</td>
      <td>0.232045</td>
      <td>0.578229</td>
      <td>-0.037501</td>
      <td>0.640134</td>
      <td>0.265745</td>
      <td>-0.087371</td>
      <td>0.004455</td>
      <td>-0.026561</td>
      <td>67.88</td>
      <td>0</td>
    </tr>
    <tr>
      <th>284805</th>
      <td>172788.0</td>
      <td>-0.240440</td>
      <td>0.530483</td>
      <td>0.702510</td>
      <td>0.689799</td>
      <td>-0.377961</td>
      <td>0.623708</td>
      <td>-0.686180</td>
      <td>0.679145</td>
      <td>0.392087</td>
      <td>-0.399126</td>
      <td>-1.933849</td>
      <td>-0.962886</td>
      <td>-1.042082</td>
      <td>0.449624</td>
      <td>1.962563</td>
      <td>-0.608577</td>
      <td>0.509928</td>
      <td>1.113981</td>
      <td>2.897849</td>
      <td>0.127434</td>
      <td>0.265245</td>
      <td>0.800049</td>
      <td>-0.163298</td>
      <td>0.123205</td>
      <td>-0.569159</td>
      <td>0.546668</td>
      <td>0.108821</td>
      <td>0.104533</td>
      <td>10.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>284806</th>
      <td>172792.0</td>
      <td>-0.533413</td>
      <td>-0.189733</td>
      <td>0.703337</td>
      <td>-0.506271</td>
      <td>-0.012546</td>
      <td>-0.649617</td>
      <td>1.577006</td>
      <td>-0.414650</td>
      <td>0.486180</td>
      <td>-0.915427</td>
      <td>-1.040458</td>
      <td>-0.031513</td>
      <td>-0.188093</td>
      <td>-0.084316</td>
      <td>0.041333</td>
      <td>-0.302620</td>
      <td>-0.660377</td>
      <td>0.167430</td>
      <td>-0.256117</td>
      <td>0.382948</td>
      <td>0.261057</td>
      <td>0.643078</td>
      <td>0.376777</td>
      <td>0.008797</td>
      <td>-0.473649</td>
      <td>-0.818267</td>
      <td>-0.002415</td>
      <td>0.013649</td>
      <td>217.00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>284807 rows × 31 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2ee9df68-d2b5-4001-99bd-2d1983cf6f66')"
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
        document.querySelector('#df-2ee9df68-d2b5-4001-99bd-2d1983cf6f66 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2ee9df68-d2b5-4001-99bd-2d1983cf6f66');
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


<div id="df-1cc8950f-628c-4b3d-a098-8abf13d08c6a">
  <button class="colab-df-quickchart" onclick="quickchart('df-1cc8950f-628c-4b3d-a098-8abf13d08c6a')"
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
        document.querySelector('#df-1cc8950f-628c-4b3d-a098-8abf13d08c6a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_25b1e40c-afb3-4f80-af59-1409a8ae94fe">
    <style>
      .colab-df-generate {
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

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('data')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_25b1e40c-afb3-4f80-af59-1409a8ae94fe button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('data');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
# 최소 전처리
data['Time'] = data['Time'] / 3600 % 24
```


```python
data['Class'].value_counts()
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
      <th>count</th>
    </tr>
    <tr>
      <th>Class</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>284315</td>
    </tr>
    <tr>
      <th>1</th>
      <td>492</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



### t-SNE 시각화


```python
# t-SNE을 위한 데이터 샘플링
fraud = data.loc[data['Class'] == 1]
non_fraud = data.loc[data['Class'] == 0].sample(3000, random_state=SEED)

new_data = pd.concat([fraud, non_fraud]).reset_index(drop=True)
y = new_data['Class'].values

tsne = TSNE(n_components=2, random_state=SEED)
tsne_data = tsne.fit_transform(new_data.drop(['Class'], axis=1))

traces = []
traces.append(go.Scatter(x=tsne_data[y==1, 0], y=tsne_data[y==1, 1], mode='markers', name='Fraud', marker=dict(color='red')))
traces.append(go.Scatter(x=tsne_data[y==0, 0], y=tsne_data[y==0, 1], mode='markers', name='Non-Fraud', marker=dict(color='blue')))

layout = go.Layout(title = "t-SNE Scatter Plot",
                   xaxis_title="component1",
                   yaxis_title="component2")

fig = go.Figure(data=traces, layout=layout)

fig.show()
```

<iframe src="/assets/img/data_analysis/anomaly_detection/autoencoder/tsne_plot.html" width="600" height="400"></iframe>


### Autoencoder 학습


```python
def get_dls(data, batch_sz, n_workers, valid_split=0.2):
    d_size = len(data)
    ixs = np.random.permutation(range(d_size))

    split = int(d_size * valid_split)
    train_ixs, valid_ixs = ixs[split:], ixs[:split]

    train_sampler = SubsetRandomSampler(train_ixs)
    valid_sampler = SubsetRandomSampler(valid_ixs)

    ds = TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(data).float())

    train_dl = DataLoader(ds, batch_sz, sampler=train_sampler, num_workers=n_workers)
    valid_dl = DataLoader(ds, batch_sz, sampler=valid_sampler, num_workers=n_workers)

    return train_dl, valid_dl
```


```python
# https://github.com/AnswerDotAI/fastprogress
def plot_loss_update(epoch, epochs, mb, train_loss, valid_loss):
    """ dynamically print the loss plot during the training/validation loop.
        expects epoch to start from 1.
    """
    x = range(1, epoch+2)
    y = np.concatenate((train_loss, valid_loss))
    graphs = [[x,train_loss], [x,valid_loss]]
    x_margin = 0.0001
    y_margin = 0.0005
    x_bounds = [1-x_margin, epochs+x_margin]
    y_bounds = [np.min(y)-y_margin, np.max(y)+y_margin]

    mb.update_graph(graphs, x_bounds, y_bounds)
```


```python
def train(epochs, model, train_dl, valid_dl, optimizer, criterion, device):
    model = model.to(device)

    mb = master_bar(range(epochs))
    mb.write(['epoch', 'train loss', 'valid loss'], table=True)
    train_loss_plot = []
    valid_loss_plot = []

    for epoch in mb:
        model.train()
        train_loss = 0.
        for train_X, train_y in progress_bar(train_dl, parent=mb):
            train_X, train_y = train_X.to(device), train_y.to(device)
            train_out = model(train_X)
            loss = criterion(train_out, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            mb.child.comment = f'{loss.item():.4f}'

        train_loss_plot.append(train_loss/len(train_dl))

        with torch.no_grad():
            model.eval()
            valid_loss = 0.
            for valid_X, valid_y in progress_bar(valid_dl, parent=mb):
                valid_X, valid_y = valid_X.to(device), valid_y.to(device)
                valid_out = model(valid_X)
                loss = criterion(valid_out, valid_y)
                valid_loss += loss.item()
                mb.child.comment = f'{loss.item():.4f}'

        valid_loss_plot.append(valid_loss/len(valid_dl))

        plot_loss_update(epoch, epochs, mb, train_loss_plot, valid_loss_plot)
        mb.write([f'{epoch+1}', f'{train_loss/len(train_dl):.6f}', f'{valid_loss/len(valid_dl):.6f}'], table=True)
```


```python
class AutoEncoder(nn.Module):
    def __init__(self, f_in):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(f_in, 100),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(100, 70),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(70, 40)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(40, 70),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(70, f_in)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
```


```python
EPOCHS = 10
BATCH_SIZE = 512
N_WORKERS = 0

model = AutoEncoder(30)
criterion = F.mse_loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```


```python
X = data.drop('Class', axis=1).values
y = data['Class'].values

X = MinMaxScaler().fit_transform(X)
X_nonfraud = X[y == 0]
X_fraud = X[y == 1]
train_dl, valid_dl = get_dls(X_nonfraud[:5000], BATCH_SIZE, N_WORKERS)
```


```python
train(EPOCHS, model, train_dl, valid_dl, optimizer, criterion, device)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train loss</th>
      <th>valid loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.270380</td>
      <td>0.194823</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.129282</td>
      <td>0.031349</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.044266</td>
      <td>0.011216</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.030186</td>
      <td>0.006618</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.024496</td>
      <td>0.003836</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.019749</td>
      <td>0.002219</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.017160</td>
      <td>0.002380</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.015620</td>
      <td>0.001863</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.013945</td>
      <td>0.002094</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.013202</td>
      <td>0.001787</td>
    </tr>
  </tbody>
</table>



    
![png](/assets/img/data_analysis/anomaly_detection/autoencoder/autoencoder_plot.png)
    


### 평가


```python
def print_metric(model, df, y, scaler=None):
    X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.2, shuffle=True, random_state=SEED, stratify=y)
    mets = [accuracy_score, precision_score, recall_score, f1_score]

    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    train_probs = model.predict_proba(X_train)[:, 1]
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]

    train_met = pd.Series({m.__name__: m(y_train, train_preds) for m in mets})
    train_met['roc_auc'] = roc_auc_score(y_train, train_probs)
    val_met = pd.Series({m.__name__: m(y_val, val_preds) for m in mets})
    val_met['roc_auc'] = roc_auc_score(y_val, val_probs)
    met_df = pd.DataFrame()
    met_df['train'] = train_met
    met_df['valid'] = val_met

    display(met_df)
```


```python
with torch.no_grad():
    model.eval()
    non_fraud_encoded = model.encoder(torch.from_numpy(X_nonfraud).float().to(device)).cpu().numpy()
    fraud_encoded = model.encoder(torch.from_numpy(X_fraud).float().to(device)).cpu().numpy()

encoded_X = np.append(non_fraud_encoded, fraud_encoded, axis=0)
encoded_y = np.append(np.zeros(len(non_fraud_encoded)), np.ones(len(fraud_encoded)))
```


```python
clf = LogisticRegression(random_state=SEED)
print('Metric scores for original data:')
print_metric(clf, X, y)
print('Metric score for encoded data:')
print_metric(clf, encoded_X, encoded_y)
```

    Metric scores for original data:
    



  <div id="df-964fcc6e-14fc-4c55-8825-f332d4f0a438" class="colab-df-container">
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
      <th>train</th>
      <th>valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>accuracy_score</th>
      <td>0.999008</td>
      <td>0.999017</td>
    </tr>
    <tr>
      <th>precision_score</th>
      <td>0.852941</td>
      <td>0.888889</td>
    </tr>
    <tr>
      <th>recall_score</th>
      <td>0.515228</td>
      <td>0.489796</td>
    </tr>
    <tr>
      <th>f1_score</th>
      <td>0.642405</td>
      <td>0.631579</td>
    </tr>
    <tr>
      <th>roc_auc</th>
      <td>0.968392</td>
      <td>0.960051</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-964fcc6e-14fc-4c55-8825-f332d4f0a438')"
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
        document.querySelector('#df-964fcc6e-14fc-4c55-8825-f332d4f0a438 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-964fcc6e-14fc-4c55-8825-f332d4f0a438');
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


<div id="df-002a766a-0bbc-411f-a4aa-366c44e16619">
  <button class="colab-df-quickchart" onclick="quickchart('df-002a766a-0bbc-411f-a4aa-366c44e16619')"
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
        document.querySelector('#df-002a766a-0bbc-411f-a4aa-366c44e16619 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>



    Metric score for encoded data:
    



  <div id="df-f8d67748-f99b-4781-a335-1e0142fcf47c" class="colab-df-container">
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
      <th>train</th>
      <th>valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>accuracy_score</th>
      <td>0.998503</td>
      <td>0.998490</td>
    </tr>
    <tr>
      <th>precision_score</th>
      <td>0.804598</td>
      <td>0.772727</td>
    </tr>
    <tr>
      <th>recall_score</th>
      <td>0.177665</td>
      <td>0.173469</td>
    </tr>
    <tr>
      <th>f1_score</th>
      <td>0.291060</td>
      <td>0.283333</td>
    </tr>
    <tr>
      <th>roc_auc</th>
      <td>0.972581</td>
      <td>0.983995</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f8d67748-f99b-4781-a335-1e0142fcf47c')"
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
        document.querySelector('#df-f8d67748-f99b-4781-a335-1e0142fcf47c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f8d67748-f99b-4781-a335-1e0142fcf47c');
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


<div id="df-42b665cd-ca49-4539-b0ae-ff4c175ef19f">
  <button class="colab-df-quickchart" onclick="quickchart('df-42b665cd-ca49-4539-b0ae-ff4c175ef19f')"
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
        document.querySelector('#df-42b665cd-ca49-4539-b0ae-ff4c175ef19f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>

&nbsp;

- 참고자료
    - [https://kr.mathworks.com/discovery/autoencoder.html](https://kr.mathworks.com/discovery/autoencoder.html)
    - [https://www.ibm.com/kr-ko/think/topics/autoencoder](https://www.ibm.com/kr-ko/think/topics/autoencoder)
    - [https://www.kaggle.com/code/rohitgr/autoencoders-tsne](https://www.kaggle.com/code/rohitgr/autoencoders-tsne)
    - [https://pebpung.github.io/autoencoder/2021/09/11/Auto-Encoder-1.html](https://pebpung.github.io/autoencoder/2021/09/11/Auto-Encoder-1.html)