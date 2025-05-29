---
layout: post
title: "[Data Analysis] 코호트 분석 실습"
date: 2025-05-29 13:00 +0900
description: 코호트 분석 실습해보기
category: [Data Analysis, Data]
tags: [Data Analysis, Data, Cohort Analysis]
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---

# 코호트 분석 실습

데이터 : [eCommerce Events History in Cosmetics Shop](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop)

쿼리를 작성하는 목표: 중형 화장품 온라인 스토어에서 5개월동안의 데이터에서 사용자의 관심 브랜드의 유지율을 확인하고자 한다. 이를 통해 각 브랜드의 관심도, 충성고객 등을 파악하고자 한다.

확인할 지표: 메인지표: 유지율(각 기간 브랜드별 사용자가 본 수 / 브랜드별 처음으로 사용자가 본 수), 가드레일 지표: 이탈률(기간별 AU/ 처음 기간 AU)

데이터의 기간: 2019.10 - 2020.02

사용할 테이블: 2019-Oct.csv, 2019-Nov.csv, 2019-Dec.csv, 2020-Jan.csv, 2020-Feb.csv

데이터 특징:
- 중형 화장품 온라인 스토어에서 5개월(2019.10 - 2020.02)동안의 행동 데이터로 Open CDP 프로젝트에서 수집한 데이터이다.
- 2019-Oct.csv: (1782439, 9), 482.54MB
- 2019-Nov.csv: (1810735, 9), 545.84MB
- 2019-Dec.csv: (1654771, 9), 415.3MB
- 2020-Jan.csv: (1811717, 9), 501.79MB
- 2020-Feb.csv: (1723228, 9), 488.8MB
- 공통:

| 컬럼 이름         | 설명 |
|------------------|------|
| `event_time`     | 이벤트가 발생한 시간 (UTC 기준) |
| `event_type`     | 이벤트 유형. 이 데이터셋에서는 `view`(보기), `cart`(장바구니 넣기), `remove_from_cart`(장바구니 삭제), `purchase`(구매) 존재 |
| `product_id`     | 상품의 고유 ID |
| `category_id`    | 상품의 카테고리 ID |
| `category_code`  | 상품 카테고리 분류 코드. 의미 있는 카테고리일 경우에만 제공되며, 액세서리처럼 다양한 종류에는 생략될 수 있음 |
| `brand`          | 소문자로 표기된 브랜드 이름. 누락될 수 있음 |
| `price`          | 상품의 가격 (소수점 포함) |
| `user_id`        | 영구적인 사용자 ID |
| `user_session`   | 임시 사용자 세션 ID. 동일한 세션 동안 유지되며, 사용자가 오랜 시간 후 다시 접속하면 변경됨 |

※위 실습은 [블로그](https://ud803.github.io/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D/2022/04/18/%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%87%E1%85%AE%E1%86%AB%E1%84%89%E1%85%A5%E1%86%A8-%E1%84%8F%E1%85%A9%E1%84%92%E1%85%A9%E1%84%90%E1%85%B3-%E1%84%87%E1%85%AE%E1%86%AB%E1%84%89%E1%85%A5%E1%86%A8-Cohort-Analysis/)의 실습을 참고하여 만들었습니다.

`일반적으로 코호트는 상호배타적(mutually exclusive)이다. 즉, 코호트끼리 겹치지 않는다. 메인지표에서는 번거로움을 피하고자 그런 과정을 거치지 않았는데, 이 점은 주의하기 바람!`


```python
import pandas as pd
import random

random.seed(1004)

whole_target_pool = set()
whole_data = pd.DataFrame()

for file in ['2019-Oct.csv', '2019-Nov.csv', '2019-Dec.csv', '2020-Jan.csv', '2020-Feb.csv']:
    data = pd.read_csv(file)
    sample_targets = random.sample(list(data.user_id.unique()), 1000)
    whole_target_pool.update(sample_targets)
    # 추출한 1000명 중 앞서 추출한 사용자가 있는 경우를 합하여 데이터를 추출한다.
    new_target = data[data.user_id.isin(sample_targets) | data.user_id.isin(whole_target_pool)]

    print(f"{file}: ",len(new_target.user_id.unique()), new_target.shape)
    whole_data = pd.concat([whole_data, new_target])

print(f"전체 데이터셋: {whole_data.shape}")
whole_data.head(3)
```

    2019-Oct.csv:  1000 (9432, 9)
    2019-Nov.csv:  1113 (17417, 9)
    2019-Dec.csv:  1247 (14018, 9)
    2020-Jan.csv:  1308 (20059, 9)
    2020-Feb.csv:  1392 (25459, 9)
    전체 데이터셋: (86385, 9)
    




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
      <th>event_time</th>
      <th>event_type</th>
      <th>product_id</th>
      <th>category_id</th>
      <th>category_code</th>
      <th>brand</th>
      <th>price</th>
      <th>user_id</th>
      <th>user_session</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5167</th>
      <td>2019-10-01 04:11:01 UTC</td>
      <td>view</td>
      <td>5649236</td>
      <td>1487580008246412266</td>
      <td>NaN</td>
      <td>concept</td>
      <td>8.65</td>
      <td>555480375</td>
      <td>ede986d2-417f-4c08-a61b-021c9d0007eb</td>
    </tr>
    <tr>
      <th>5253</th>
      <td>2019-10-01 04:13:15 UTC</td>
      <td>view</td>
      <td>5649236</td>
      <td>1487580008246412266</td>
      <td>NaN</td>
      <td>concept</td>
      <td>8.65</td>
      <td>555480375</td>
      <td>ede986d2-417f-4c08-a61b-021c9d0007eb</td>
    </tr>
    <tr>
      <th>6462</th>
      <td>2019-10-01 04:41:46 UTC</td>
      <td>view</td>
      <td>5649236</td>
      <td>1487580008246412266</td>
      <td>NaN</td>
      <td>concept</td>
      <td>8.65</td>
      <td>555484401</td>
      <td>bf933666-7fbb-4536-9f3a-102ceaee5859</td>
    </tr>
  </tbody>
</table>
</div>




```python
import sqlite3

con = sqlite3.connect('ecommerce.db')

cur = con.cursor()

cur.execute(
    """
    CREATE TABLE events
        (
            event_time text,
            event_type text,
            product_id text,
            category_id text,
            category_code text,
            brand text,
            price text,
            user_id text,
            user_session text
        )
    """
)

data_row = []
# 튜플형태로 row 저장
for idx, row in whole_data.iterrows():
    data_row.append(tuple([*row.values]))

# executemany를 사용하여 table에 insert
cur.executemany(
    """
    INSERT INTO events
    VALUES (
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?
    )
    """,
    data_row
)

con.commit()

```

## 메인지표: 브랜드별 리텐션

### 월별 브랜드 리텐션


```python
import sqlite3

con = sqlite3.connect('ecommerce.db')

cur = con.cursor()
```


```python
result = con.execute(
    """
    WITH
    base AS (
    SELECT
        user_id,
        -- '월'만 빼내는데, 연도가 바뀌면 계산이 틀어지기 때문에 현재 연도에서 가장 낮은 연도인 2019년을 뺀 만큼에 12를 곱한 값을 더해준다
        -- 즉, 2020년의 1월은 12 + (2020-2019)*12 = 13이 된다
        STRFTIME('%m', DATE(SUBSTR(event_time, 1, 10))) + (STRFTIME('%Y', DATE(SUBSTR(event_time, 1, 10))) - 2019) * 12 AS event_month,
        event_type,
        brand
    FROM events
    WHERE 
        brand IS NOT NULL
        AND event_type = 'view'
    ),
    first_view AS (
    SELECT
        user_id,
        brand AS cohort,
        MIN(event_month) AS cohort_time
    FROM base
    GROUP BY
        user_id,
        brand
    ),
    joinned AS (
    SELECT
        t1.user_id,
        t2.cohort,
        t1.event_month,
        t1.event_month - t2.cohort_time AS month_diff
    FROM base t1
    LEFT JOIN first_view t2
    ON t1.user_id = t2.user_id
    AND t1.brand = t2.cohort
    )

    SELECT
        cohort,
        month_diff,
        COUNT(DISTINCT user_id)
    FROM joinned
    GROUP BY 
        cohort,
        month_diff
    ORDER BY 
        cohort ASC,
        month_diff ASC
    """
).fetchall()
```


```python
# 데이터프레임으로 만들고
# 컬럼의 이름을 바꿔주고
# 피벗 기능을 이용해 코호트 테이블 형태로 만들어준다
# 빈 값은 0으로 채운다
pivot_table = pd.DataFrame(result)\
    .rename(columns={0: 'cohort', 1: 'duration', 2: 'value'})\
    .pivot(index='cohort', columns='duration', values='value')\
    .fillna(0)\
    .sort_values(by=[0], ascending=False)\
    .iloc[:10, :]

# 상위 10개만 잘랐다
pivot_table
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
      <th>duration</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
    <tr>
      <th>cohort</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>runail</th>
      <td>678.0</td>
      <td>67.0</td>
      <td>26.0</td>
      <td>15.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>irisk</th>
      <td>547.0</td>
      <td>54.0</td>
      <td>28.0</td>
      <td>12.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>grattol</th>
      <td>419.0</td>
      <td>48.0</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>masura</th>
      <td>311.0</td>
      <td>27.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>estel</th>
      <td>300.0</td>
      <td>17.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>kapous</th>
      <td>292.0</td>
      <td>19.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>jessnail</th>
      <td>280.0</td>
      <td>19.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>uno</th>
      <td>219.0</td>
      <td>12.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ingarden</th>
      <td>217.0</td>
      <td>28.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>concept</th>
      <td>211.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 첫 번째 기간으로 나누어 비율로 만들어주고
# %가 나오도록 포맷팅을 해주고
# 색을 입혀준다

round(pivot_table.div(pivot_table[0], axis='index'), 2)\
    .style.format({k: '{:,.0%}'.format for k in pivot_table})\
    .background_gradient(cmap ='Blues', axis=None, vmax=0.2) 
```




<style type="text/css">
#T_1cc66_row0_col0, #T_1cc66_row1_col0, #T_1cc66_row2_col0, #T_1cc66_row3_col0, #T_1cc66_row4_col0, #T_1cc66_row5_col0, #T_1cc66_row6_col0, #T_1cc66_row7_col0, #T_1cc66_row8_col0, #T_1cc66_row9_col0 {
  background-color: #08306b;
  color: #f1f1f1;
}
#T_1cc66_row0_col1, #T_1cc66_row1_col1 {
  background-color: #6aaed6;
  color: #f1f1f1;
}
#T_1cc66_row0_col2, #T_1cc66_row9_col1 {
  background-color: #d0e1f2;
  color: #000000;
}
#T_1cc66_row0_col3, #T_1cc66_row1_col3, #T_1cc66_row5_col2, #T_1cc66_row6_col2, #T_1cc66_row7_col2, #T_1cc66_row8_col3 {
  background-color: #e3eef9;
  color: #000000;
}
#T_1cc66_row0_col4, #T_1cc66_row1_col4, #T_1cc66_row2_col4, #T_1cc66_row3_col3, #T_1cc66_row4_col2, #T_1cc66_row4_col3, #T_1cc66_row5_col3, #T_1cc66_row6_col3, #T_1cc66_row8_col4, #T_1cc66_row9_col2 {
  background-color: #eef5fc;
  color: #000000;
}
#T_1cc66_row1_col2, #T_1cc66_row2_col2, #T_1cc66_row7_col1 {
  background-color: #c6dbef;
  color: #000000;
}
#T_1cc66_row2_col1 {
  background-color: #5ba3d0;
  color: #f1f1f1;
}
#T_1cc66_row2_col3, #T_1cc66_row3_col2, #T_1cc66_row8_col2 {
  background-color: #d9e8f5;
  color: #000000;
}
#T_1cc66_row3_col1 {
  background-color: #7fb9da;
  color: #000000;
}
#T_1cc66_row3_col4, #T_1cc66_row4_col4, #T_1cc66_row5_col4, #T_1cc66_row6_col4, #T_1cc66_row7_col3, #T_1cc66_row7_col4, #T_1cc66_row9_col3, #T_1cc66_row9_col4 {
  background-color: #f7fbff;
  color: #000000;
}
#T_1cc66_row4_col1 {
  background-color: #b7d4ea;
  color: #000000;
}
#T_1cc66_row5_col1, #T_1cc66_row6_col1 {
  background-color: #a6cee4;
  color: #000000;
}
#T_1cc66_row8_col1 {
  background-color: #3b8bc2;
  color: #f1f1f1;
}
</style>
<table id="T_1cc66">
  <thead>
    <tr>
      <th class="index_name level0" >duration</th>
      <th id="T_1cc66_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_1cc66_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_1cc66_level0_col2" class="col_heading level0 col2" >2</th>
      <th id="T_1cc66_level0_col3" class="col_heading level0 col3" >3</th>
      <th id="T_1cc66_level0_col4" class="col_heading level0 col4" >4</th>
    </tr>
    <tr>
      <th class="index_name level0" >cohort</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1cc66_level0_row0" class="row_heading level0 row0" >runail</th>
      <td id="T_1cc66_row0_col0" class="data row0 col0" >100%</td>
      <td id="T_1cc66_row0_col1" class="data row0 col1" >10%</td>
      <td id="T_1cc66_row0_col2" class="data row0 col2" >4%</td>
      <td id="T_1cc66_row0_col3" class="data row0 col3" >2%</td>
      <td id="T_1cc66_row0_col4" class="data row0 col4" >1%</td>
    </tr>
    <tr>
      <th id="T_1cc66_level0_row1" class="row_heading level0 row1" >irisk</th>
      <td id="T_1cc66_row1_col0" class="data row1 col0" >100%</td>
      <td id="T_1cc66_row1_col1" class="data row1 col1" >10%</td>
      <td id="T_1cc66_row1_col2" class="data row1 col2" >5%</td>
      <td id="T_1cc66_row1_col3" class="data row1 col3" >2%</td>
      <td id="T_1cc66_row1_col4" class="data row1 col4" >1%</td>
    </tr>
    <tr>
      <th id="T_1cc66_level0_row2" class="row_heading level0 row2" >grattol</th>
      <td id="T_1cc66_row2_col0" class="data row2 col0" >100%</td>
      <td id="T_1cc66_row2_col1" class="data row2 col1" >11%</td>
      <td id="T_1cc66_row2_col2" class="data row2 col2" >5%</td>
      <td id="T_1cc66_row2_col3" class="data row2 col3" >3%</td>
      <td id="T_1cc66_row2_col4" class="data row2 col4" >1%</td>
    </tr>
    <tr>
      <th id="T_1cc66_level0_row3" class="row_heading level0 row3" >masura</th>
      <td id="T_1cc66_row3_col0" class="data row3 col0" >100%</td>
      <td id="T_1cc66_row3_col1" class="data row3 col1" >9%</td>
      <td id="T_1cc66_row3_col2" class="data row3 col2" >3%</td>
      <td id="T_1cc66_row3_col3" class="data row3 col3" >1%</td>
      <td id="T_1cc66_row3_col4" class="data row3 col4" >0%</td>
    </tr>
    <tr>
      <th id="T_1cc66_level0_row4" class="row_heading level0 row4" >estel</th>
      <td id="T_1cc66_row4_col0" class="data row4 col0" >100%</td>
      <td id="T_1cc66_row4_col1" class="data row4 col1" >6%</td>
      <td id="T_1cc66_row4_col2" class="data row4 col2" >1%</td>
      <td id="T_1cc66_row4_col3" class="data row4 col3" >1%</td>
      <td id="T_1cc66_row4_col4" class="data row4 col4" >0%</td>
    </tr>
    <tr>
      <th id="T_1cc66_level0_row5" class="row_heading level0 row5" >kapous</th>
      <td id="T_1cc66_row5_col0" class="data row5 col0" >100%</td>
      <td id="T_1cc66_row5_col1" class="data row5 col1" >7%</td>
      <td id="T_1cc66_row5_col2" class="data row5 col2" >2%</td>
      <td id="T_1cc66_row5_col3" class="data row5 col3" >1%</td>
      <td id="T_1cc66_row5_col4" class="data row5 col4" >0%</td>
    </tr>
    <tr>
      <th id="T_1cc66_level0_row6" class="row_heading level0 row6" >jessnail</th>
      <td id="T_1cc66_row6_col0" class="data row6 col0" >100%</td>
      <td id="T_1cc66_row6_col1" class="data row6 col1" >7%</td>
      <td id="T_1cc66_row6_col2" class="data row6 col2" >2%</td>
      <td id="T_1cc66_row6_col3" class="data row6 col3" >1%</td>
      <td id="T_1cc66_row6_col4" class="data row6 col4" >0%</td>
    </tr>
    <tr>
      <th id="T_1cc66_level0_row7" class="row_heading level0 row7" >uno</th>
      <td id="T_1cc66_row7_col0" class="data row7 col0" >100%</td>
      <td id="T_1cc66_row7_col1" class="data row7 col1" >5%</td>
      <td id="T_1cc66_row7_col2" class="data row7 col2" >2%</td>
      <td id="T_1cc66_row7_col3" class="data row7 col3" >0%</td>
      <td id="T_1cc66_row7_col4" class="data row7 col4" >0%</td>
    </tr>
    <tr>
      <th id="T_1cc66_level0_row8" class="row_heading level0 row8" >ingarden</th>
      <td id="T_1cc66_row8_col0" class="data row8 col0" >100%</td>
      <td id="T_1cc66_row8_col1" class="data row8 col1" >13%</td>
      <td id="T_1cc66_row8_col2" class="data row8 col2" >3%</td>
      <td id="T_1cc66_row8_col3" class="data row8 col3" >2%</td>
      <td id="T_1cc66_row8_col4" class="data row8 col4" >1%</td>
    </tr>
    <tr>
      <th id="T_1cc66_level0_row9" class="row_heading level0 row9" >concept</th>
      <td id="T_1cc66_row9_col0" class="data row9 col0" >100%</td>
      <td id="T_1cc66_row9_col1" class="data row9 col1" >4%</td>
      <td id="T_1cc66_row9_col2" class="data row9 col2" >1%</td>
      <td id="T_1cc66_row9_col3" class="data row9 col3" >0%</td>
      <td id="T_1cc66_row9_col4" class="data row9 col4" >0%</td>
    </tr>
  </tbody>
</table>




### 주별 브랜드 리텐션(2019-10)


```python
result_2 = con.execute("""
WITH 
base AS (
-- 문자열을 날짜로 바꿔주기 위한 용도
SELECT
    user_id,
    -- '주간'만 빼내는데, 연도가 바뀌면 계산이 틀어지기 때문에 현재 연도에서 가장 낮은 연도인 2019년을 뺀 만큼에 52를 곱한 값을 더해준다
    -- 즉, 2019년 마지막 주는 52가 되고, 2020년의 첫 주는 1 + (2020-2019)*52 = 53이 된다
    STRFTIME('%W', DATE(SUBSTR(event_time, 1, 10))) + (STRFTIME('%Y', DATE(SUBSTR(event_time, 1, 10))) - 2019) * 52 AS event_week,
    event_type,
    brand
FROM events
-- 9개의 주간으로 나누기 위해 기간을 제한해준다
WHERE STRFTIME('%W', DATE(SUBSTR(event_time, 1, 10))) + (STRFTIME('%Y', DATE(SUBSTR(event_time, 1, 10))) - 2019) * 52 <= 47
AND brand IS NOT NULL
AND event_type = 'view'
AND DATE(SUBSTR(event_time, 1, 10)) >= '2019-10-01'
AND DATE(SUBSTR(event_time, 1, 10)) <= '2019-10-31'
)
,first_view AS (
SELECT
    user_id,
    brand AS cohort,
    MIN(event_week) AS cohort_time
FROM base
GROUP BY user_id, brand
)
,joinned AS (
SELECT
    t1.user_id,
    t2.cohort,
    t1.event_week,
    t1.event_week - t2.cohort_time AS week_diff
FROM base t1
LEFT JOIN first_view t2
ON t1.user_id = t2.user_id
AND t1.brand = t2.cohort
)

SELECT
    cohort,
    week_diff,
    COUNT(DISTINCT user_id)
FROM joinned
GROUP BY cohort, week_diff
ORDER BY cohort ASC, week_diff ASC
""").fetchall()
```


```python
# 데이터프레임으로 만들고
# 컬럼의 이름을 바꿔주고
# 피벗 기능을 이용해 코호트 테이블 형태로 만들어준다
# 빈 값은 0으로 채운다
pivot_table_2 = pd.DataFrame(result_2)\
    .rename(columns={0: 'cohort', 1: 'duration', 2: 'value'})\
    .pivot(index='cohort', columns='duration', values='value')\
    .fillna(0)\
    .sort_values(by=[0], ascending=False)\
    .iloc[:10, :]

# 상위 10개만 잘랐다
pivot_table_2
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
      <th>duration</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
    <tr>
      <th>cohort</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>runail</th>
      <td>100.0</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>irisk</th>
      <td>85.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>masura</th>
      <td>69.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>grattol</th>
      <td>52.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>estel</th>
      <td>45.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>jessnail</th>
      <td>41.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>kapous</th>
      <td>40.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>concept</th>
      <td>40.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ingarden</th>
      <td>33.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>uno</th>
      <td>32.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 첫 번째 기간으로 나누어 비율로 만들어주고
# %가 나오도록 포맷팅을 해주고
# 색을 입혀준다

round(pivot_table_2.div(pivot_table_2[0], axis='index'), 2)\
    .style.format({k: '{:,.0%}'.format for k in pivot_table_2})\
    .background_gradient(cmap ='Blues', axis=None, vmax=0.2) 
```




<style type="text/css">
#T_b150f_row0_col0, #T_b150f_row1_col0, #T_b150f_row2_col0, #T_b150f_row3_col0, #T_b150f_row4_col0, #T_b150f_row5_col0, #T_b150f_row6_col0, #T_b150f_row7_col0, #T_b150f_row8_col0, #T_b150f_row9_col0 {
  background-color: #08306b;
  color: #f1f1f1;
}
#T_b150f_row0_col1, #T_b150f_row5_col1 {
  background-color: #6aaed6;
  color: #f1f1f1;
}
#T_b150f_row0_col2 {
  background-color: #c6dbef;
  color: #000000;
}
#T_b150f_row0_col3, #T_b150f_row2_col4 {
  background-color: #eef5fc;
  color: #000000;
}
#T_b150f_row0_col4, #T_b150f_row1_col4, #T_b150f_row3_col4, #T_b150f_row4_col2, #T_b150f_row4_col3, #T_b150f_row5_col2, #T_b150f_row5_col3, #T_b150f_row5_col4, #T_b150f_row6_col2, #T_b150f_row6_col3, #T_b150f_row6_col4, #T_b150f_row7_col1, #T_b150f_row7_col2, #T_b150f_row7_col3, #T_b150f_row7_col4, #T_b150f_row8_col4, #T_b150f_row9_col3, #T_b150f_row9_col4 {
  background-color: #f7fbff;
  color: #000000;
}
#T_b150f_row1_col1 {
  background-color: #a6cee4;
  color: #000000;
}
#T_b150f_row1_col2, #T_b150f_row2_col1, #T_b150f_row3_col3, #T_b150f_row9_col1 {
  background-color: #b7d4ea;
  color: #000000;
}
#T_b150f_row1_col3, #T_b150f_row4_col1 {
  background-color: #e3eef9;
  color: #000000;
}
#T_b150f_row2_col2, #T_b150f_row4_col4 {
  background-color: #d0e1f2;
  color: #000000;
}
#T_b150f_row2_col3, #T_b150f_row8_col3, #T_b150f_row9_col2 {
  background-color: #d9e8f5;
  color: #000000;
}
#T_b150f_row3_col1 {
  background-color: #4a98c9;
  color: #f1f1f1;
}
#T_b150f_row3_col2, #T_b150f_row6_col1 {
  background-color: #94c4df;
  color: #000000;
}
#T_b150f_row8_col1, #T_b150f_row8_col2 {
  background-color: #7fb9da;
  color: #000000;
}
</style>
<table id="T_b150f">
  <thead>
    <tr>
      <th class="index_name level0" >duration</th>
      <th id="T_b150f_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_b150f_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_b150f_level0_col2" class="col_heading level0 col2" >2</th>
      <th id="T_b150f_level0_col3" class="col_heading level0 col3" >3</th>
      <th id="T_b150f_level0_col4" class="col_heading level0 col4" >4</th>
    </tr>
    <tr>
      <th class="index_name level0" >cohort</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b150f_level0_row0" class="row_heading level0 row0" >runail</th>
      <td id="T_b150f_row0_col0" class="data row0 col0" >100%</td>
      <td id="T_b150f_row0_col1" class="data row0 col1" >10%</td>
      <td id="T_b150f_row0_col2" class="data row0 col2" >5%</td>
      <td id="T_b150f_row0_col3" class="data row0 col3" >1%</td>
      <td id="T_b150f_row0_col4" class="data row0 col4" >0%</td>
    </tr>
    <tr>
      <th id="T_b150f_level0_row1" class="row_heading level0 row1" >irisk</th>
      <td id="T_b150f_row1_col0" class="data row1 col0" >100%</td>
      <td id="T_b150f_row1_col1" class="data row1 col1" >7%</td>
      <td id="T_b150f_row1_col2" class="data row1 col2" >6%</td>
      <td id="T_b150f_row1_col3" class="data row1 col3" >2%</td>
      <td id="T_b150f_row1_col4" class="data row1 col4" >0%</td>
    </tr>
    <tr>
      <th id="T_b150f_level0_row2" class="row_heading level0 row2" >masura</th>
      <td id="T_b150f_row2_col0" class="data row2 col0" >100%</td>
      <td id="T_b150f_row2_col1" class="data row2 col1" >6%</td>
      <td id="T_b150f_row2_col2" class="data row2 col2" >4%</td>
      <td id="T_b150f_row2_col3" class="data row2 col3" >3%</td>
      <td id="T_b150f_row2_col4" class="data row2 col4" >1%</td>
    </tr>
    <tr>
      <th id="T_b150f_level0_row3" class="row_heading level0 row3" >grattol</th>
      <td id="T_b150f_row3_col0" class="data row3 col0" >100%</td>
      <td id="T_b150f_row3_col1" class="data row3 col1" >12%</td>
      <td id="T_b150f_row3_col2" class="data row3 col2" >8%</td>
      <td id="T_b150f_row3_col3" class="data row3 col3" >6%</td>
      <td id="T_b150f_row3_col4" class="data row3 col4" >0%</td>
    </tr>
    <tr>
      <th id="T_b150f_level0_row4" class="row_heading level0 row4" >estel</th>
      <td id="T_b150f_row4_col0" class="data row4 col0" >100%</td>
      <td id="T_b150f_row4_col1" class="data row4 col1" >2%</td>
      <td id="T_b150f_row4_col2" class="data row4 col2" >0%</td>
      <td id="T_b150f_row4_col3" class="data row4 col3" >0%</td>
      <td id="T_b150f_row4_col4" class="data row4 col4" >4%</td>
    </tr>
    <tr>
      <th id="T_b150f_level0_row5" class="row_heading level0 row5" >jessnail</th>
      <td id="T_b150f_row5_col0" class="data row5 col0" >100%</td>
      <td id="T_b150f_row5_col1" class="data row5 col1" >10%</td>
      <td id="T_b150f_row5_col2" class="data row5 col2" >0%</td>
      <td id="T_b150f_row5_col3" class="data row5 col3" >0%</td>
      <td id="T_b150f_row5_col4" class="data row5 col4" >0%</td>
    </tr>
    <tr>
      <th id="T_b150f_level0_row6" class="row_heading level0 row6" >kapous</th>
      <td id="T_b150f_row6_col0" class="data row6 col0" >100%</td>
      <td id="T_b150f_row6_col1" class="data row6 col1" >8%</td>
      <td id="T_b150f_row6_col2" class="data row6 col2" >0%</td>
      <td id="T_b150f_row6_col3" class="data row6 col3" >0%</td>
      <td id="T_b150f_row6_col4" class="data row6 col4" >0%</td>
    </tr>
    <tr>
      <th id="T_b150f_level0_row7" class="row_heading level0 row7" >concept</th>
      <td id="T_b150f_row7_col0" class="data row7 col0" >100%</td>
      <td id="T_b150f_row7_col1" class="data row7 col1" >0%</td>
      <td id="T_b150f_row7_col2" class="data row7 col2" >0%</td>
      <td id="T_b150f_row7_col3" class="data row7 col3" >0%</td>
      <td id="T_b150f_row7_col4" class="data row7 col4" >0%</td>
    </tr>
    <tr>
      <th id="T_b150f_level0_row8" class="row_heading level0 row8" >ingarden</th>
      <td id="T_b150f_row8_col0" class="data row8 col0" >100%</td>
      <td id="T_b150f_row8_col1" class="data row8 col1" >9%</td>
      <td id="T_b150f_row8_col2" class="data row8 col2" >9%</td>
      <td id="T_b150f_row8_col3" class="data row8 col3" >3%</td>
      <td id="T_b150f_row8_col4" class="data row8 col4" >0%</td>
    </tr>
    <tr>
      <th id="T_b150f_level0_row9" class="row_heading level0 row9" >uno</th>
      <td id="T_b150f_row9_col0" class="data row9 col0" >100%</td>
      <td id="T_b150f_row9_col1" class="data row9 col1" >6%</td>
      <td id="T_b150f_row9_col2" class="data row9 col2" >3%</td>
      <td id="T_b150f_row9_col3" class="data row9 col3" >0%</td>
      <td id="T_b150f_row9_col4" class="data row9 col4" >0%</td>
    </tr>
  </tbody>
</table>




## 가드레일 지표: 이탈률


```python
guardrail = con.execute("""
WITH 
base AS (
-- 문자열을 날짜로 바꿔주기 위한 용도
SELECT
    user_id,
    STRFTIME('%m', DATE(SUBSTR(event_time, 1, 10))) + (STRFTIME('%Y', DATE(SUBSTR(event_time, 1, 10))) - 2019) * 12 AS event_month
FROM events
WHERE event_type = 'view'
)
,first_view AS (
-- 우선 사용자별로 최초 유입 월을 찾는다. 이게 코호트가 된다.
SELECT
    user_id,
    MIN(event_month) AS cohort
FROM base
GROUP BY user_id
)
,joinned AS (
-- 기존 데이터에 위에서 찾은 코호트를 조인해준다. 그리고 기존 이벤트 월과 코호트 월의 차이를 빼준다
SELECT
    t1.user_id,
    t2.cohort,
    t1.event_month,
    t1.event_month - t2.cohort AS month_diff
FROM base t1
LEFT JOIN first_view t2
ON t1.user_id = t2.user_id
)

-- (기준 코호트, 기준 코호트로부터의 경과주) 쌍을 만들어 고유한 사용자 수를 센다
SELECT
    cohort,
    month_diff,
    COUNT(DISTINCT user_id)
FROM joinned
GROUP BY cohort, month_diff
ORDER BY cohort ASC, month_diff ASC
""").fetchall()
```


```python
# 데이터프레임으로 만들고
# 컬럼의 이름을 바꿔주고
# 피벗 기능을 이용해 코호트 테이블 형태로 만들어준다
# 빈 값은 0으로 채운다
pivot_table_3 = pd.DataFrame(guardrail)\
    .rename(columns={0: 'cohort', 1: 'duration', 2: 'value'})\
    .pivot(index='cohort', columns='duration', values='value')\
    .fillna(0)

pivot_table_3
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
      <th>duration</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
    <tr>
      <th>cohort</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>978.0</td>
      <td>102.0</td>
      <td>78.0</td>
      <td>48.0</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>971.0</td>
      <td>158.0</td>
      <td>105.0</td>
      <td>99.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>961.0</td>
      <td>129.0</td>
      <td>86.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>977.0</td>
      <td>150.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>972.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 첫 번째 기간으로 나누어 비율로 만들어주고
# %가 나오도록 포맷팅을 해주고
# 색을 입혀준다

round(pivot_table_3.div(pivot_table_3[0], axis='index'), 2)\
    .style.format({k: '{:,.0%}'.format for k in pivot_table_3})\
    .background_gradient(cmap ='Blues', axis=None, vmax=0.2) 
```




<style type="text/css">
#T_cc109_row0_col0, #T_cc109_row1_col0, #T_cc109_row2_col0, #T_cc109_row3_col0, #T_cc109_row4_col0 {
  background-color: #08306b;
  color: #f1f1f1;
}
#T_cc109_row0_col1, #T_cc109_row1_col3 {
  background-color: #6aaed6;
  color: #f1f1f1;
}
#T_cc109_row0_col2 {
  background-color: #94c4df;
  color: #000000;
}
#T_cc109_row0_col3 {
  background-color: #c6dbef;
  color: #000000;
}
#T_cc109_row0_col4 {
  background-color: #d0e1f2;
  color: #000000;
}
#T_cc109_row1_col1 {
  background-color: #1764ab;
  color: #f1f1f1;
}
#T_cc109_row1_col2 {
  background-color: #5ba3d0;
  color: #f1f1f1;
}
#T_cc109_row1_col4, #T_cc109_row2_col3, #T_cc109_row2_col4, #T_cc109_row3_col2, #T_cc109_row3_col3, #T_cc109_row3_col4, #T_cc109_row4_col1, #T_cc109_row4_col2, #T_cc109_row4_col3, #T_cc109_row4_col4 {
  background-color: #f7fbff;
  color: #000000;
}
#T_cc109_row2_col1 {
  background-color: #3b8bc2;
  color: #f1f1f1;
}
#T_cc109_row2_col2 {
  background-color: #7fb9da;
  color: #000000;
}
#T_cc109_row3_col1 {
  background-color: #2171b5;
  color: #f1f1f1;
}
</style>
<table id="T_cc109">
  <thead>
    <tr>
      <th class="index_name level0" >duration</th>
      <th id="T_cc109_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_cc109_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_cc109_level0_col2" class="col_heading level0 col2" >2</th>
      <th id="T_cc109_level0_col3" class="col_heading level0 col3" >3</th>
      <th id="T_cc109_level0_col4" class="col_heading level0 col4" >4</th>
    </tr>
    <tr>
      <th class="index_name level0" >cohort</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_cc109_level0_row0" class="row_heading level0 row0" >10</th>
      <td id="T_cc109_row0_col0" class="data row0 col0" >100%</td>
      <td id="T_cc109_row0_col1" class="data row0 col1" >10%</td>
      <td id="T_cc109_row0_col2" class="data row0 col2" >8%</td>
      <td id="T_cc109_row0_col3" class="data row0 col3" >5%</td>
      <td id="T_cc109_row0_col4" class="data row0 col4" >4%</td>
    </tr>
    <tr>
      <th id="T_cc109_level0_row1" class="row_heading level0 row1" >11</th>
      <td id="T_cc109_row1_col0" class="data row1 col0" >100%</td>
      <td id="T_cc109_row1_col1" class="data row1 col1" >16%</td>
      <td id="T_cc109_row1_col2" class="data row1 col2" >11%</td>
      <td id="T_cc109_row1_col3" class="data row1 col3" >10%</td>
      <td id="T_cc109_row1_col4" class="data row1 col4" >0%</td>
    </tr>
    <tr>
      <th id="T_cc109_level0_row2" class="row_heading level0 row2" >12</th>
      <td id="T_cc109_row2_col0" class="data row2 col0" >100%</td>
      <td id="T_cc109_row2_col1" class="data row2 col1" >13%</td>
      <td id="T_cc109_row2_col2" class="data row2 col2" >9%</td>
      <td id="T_cc109_row2_col3" class="data row2 col3" >0%</td>
      <td id="T_cc109_row2_col4" class="data row2 col4" >0%</td>
    </tr>
    <tr>
      <th id="T_cc109_level0_row3" class="row_heading level0 row3" >13</th>
      <td id="T_cc109_row3_col0" class="data row3 col0" >100%</td>
      <td id="T_cc109_row3_col1" class="data row3 col1" >15%</td>
      <td id="T_cc109_row3_col2" class="data row3 col2" >0%</td>
      <td id="T_cc109_row3_col3" class="data row3 col3" >0%</td>
      <td id="T_cc109_row3_col4" class="data row3 col4" >0%</td>
    </tr>
    <tr>
      <th id="T_cc109_level0_row4" class="row_heading level0 row4" >14</th>
      <td id="T_cc109_row4_col0" class="data row4 col0" >100%</td>
      <td id="T_cc109_row4_col1" class="data row4 col1" >0%</td>
      <td id="T_cc109_row4_col2" class="data row4 col2" >0%</td>
      <td id="T_cc109_row4_col3" class="data row4 col3" >0%</td>
      <td id="T_cc109_row4_col4" class="data row4 col4" >0%</td>
    </tr>
  </tbody>
</table>





```python
con.close()
```

## 해석하기

### 메인지표
- `ingarden`가 두번째 달에 15%로 가장 높은 유지율을 가지고 있다. 하지만 세번째 달이 지나면 3%로 유지율이 줄어드는 것을 확인할 수 있다. 타 브랜드에 비하여 10%p로 줄어들었다.(확인 필요)
- 2기간이 지나면 유지율이 5%이하로 낮아진다.

**Action Plan** 
- 2기간동안 유지율을 높일 수 있는 방안 탐색. ex) 프로모션, 특정 제품 추가 분석 등
- `ingarden`가 유지율이 높다가 두번째 달에 급격히 낮아지는 이유 탐색

### 가드레일 지표
- 다음달 이탈률이 84% ~ 90%까지 나온다. 이탈률이 급격히 낮아짐으로 메인보다 가드레일에 먼저 초점을 맞출 필요가 있다.


## 회고
데이터에서 상품을 본(View)기준으로 분석을 진행하다보니 정확한 분석과 해석은 힘들지만 코호트 분석을 어떻게 만들고 나타낼 수 있을 지 이해할 수 있게 되었다.

해석에서 어떻게 해석을 해야할 지 고민을 많이 해보았으나, 아직 잘 와닿지 않는 거 같다. 좀더 다양한 해석을 찾아보고 이해해보는 시간을 가져야 겠다.
