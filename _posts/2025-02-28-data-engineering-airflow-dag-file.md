---
layout: post
title: "[Data Engineering] Airflow DAG파일 만들기"
date: 2025-02-28 17:59 +0900
description: DAG 구조를 파악하고자 합니다.
category: [Data Engineering, Airflow]
tags: [Data Engineering, Airflow]
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---

## ETL 만들기

ETL을 만들기 위해서는 데이터가 필요합니다. api를 통해 가져오면 좋겠지만 빠르게 만들어보기 위해서 Kaggle에서 데이터를 구하여서 사용하도록 하겠습니다.

pokemon_data_pokeapi.csv 출처

[https://www.kaggle.com/datasets/mohitbansal31s/pokemon-dataset?resource=download](https://www.kaggle.com/datasets/mohitbansal31s/pokemon-dataset?resource=download)

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
import pandas as pd
import pendulum
```

### Extract
pandas라이브러리를 통해 csv파일을 불러오도록하겠습니다.

```python
def extract() -> dict:
    data = pd.read_csv('pokemon_data_pokeapi.csv')

    return data.to_dict(orient="records")
```

to_dict을 사용하여 Json형식으로 파일을 보내도록 합니다.
- 직렬화하기 위함.
- sql로 저장하기 위함

### Transform
받아온 json을 다시 DataFrame형식으로 변환하고 Legendary Status가 있는 데이터를 사용하였습니다.
또한, null값을 Nan값으로 변환, ","을 "|"으로 변환하였습니다.

```python
def transform(ti: dict) -> dict:
    data = ti.xcom_pull(task_ids="extract")
    df = pd.DataFrame(data)
    
    legendary_data = df[df['Legendary Status'] == 'Yes']

    # NaN 값 변환
    legendary_data = legendary_data.fillna("NULL")
    
    # Abilities를 문자열로 변환
    legendary_data["Abilities"] = legendary_data["Abilities"].apply(lambda x: x.replace(",", "|"))

    return legendary_data.to_dict(orient='records')
```

### Load

```python
def load(ti: dict) -> None:
    data = ti.xcom_pull(task_ids="transform")
    mysql_hook = MySqlHook(mysql_conn_id="mysql_conn")
    conn = mysql_hook.get_conn()
    cursor = conn.cursor()
    
    # 테이블이 없으면 생성하는 SQL
    create_table_query = """
    CREATE TABLE IF NOT EXISTS pokemon_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100),
        pokedex_number INT,
        type1 VARCHAR(50),
        type2 VARCHAR(50) NULL,
        classification VARCHAR(100),
        height FLOAT,
        weight FLOAT,
        abilities TEXT,
        generation INT,
        legendary_status VARCHAR(10)
    );
    """
    cursor.execute(create_table_query)

    # 데이터 삽입 SQL
    insert_query = """
    INSERT INTO pokemon_data 
    (name, pokedex_number, type1, type2, classification, height, weight, abilities, generation, legendary_status)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    for row in data:
        cursor.execute(insert_query, (
            row["Name"], row["Pokedex Number"], row["Type1"], row["Type2"],
            row["Classification"], row["Height (m)"], row["Weight (kg)"],
            row["Abilities"], row["Generation"], row["Legendary Status"]
        ))

    conn.commit()
    cursor.close()
    conn.close()
```

만약 Table이 없을 경우 Table을 생성하도록 만들었습니다.
반복하여 데이터를 insert하고 끝나면 commit하고 종료합니다.

Airflow에서 MySQL에 연결하려면 airflow.cfg 또는 Web UI에서 Connection을 설정.

Airflow UI → Admin → Connections → Add Connection
- Conn Id: mysql_conn
- Conn Type: MySQL
- Host: your-mysql-host
- Schema: your-database
- Login: your-username
- Password: your-password
- Port: 3306


## Task 설정
```python
local_tz = pendulum.timezone("Asia/Seoul")

default_args = {
    'owner' : 'admin',
    'depends_on_past' : False,
    'start_date' : datetime(2025,2,28, tzinfo=local_tz),
    'email' : 'test@gmail.com',
    'email_on_failure' : False,
    'email_on_retry' : False
    'retries' : 1,
    'retry_delay' : timedelta(minutes=5)
}
```

시간은 한국 서울로 설정합니다.

- owner : 소유자
- depends_on_past : 이전 실행 여부
- start_date : 시작 시간
- email_on_failure : 실패시, 이메일 전송
- email_on_retry : 재실행 실패시, 이메일전송
- retries : 실패 시, 재시도 횟수
- retry_delay : 재시도 간격

## DAG 설정
```python
with DAG(
    "pokemon_etl_pipeline",
    default_args=default_args,
    description="ETL pipeline for Pokémon data",
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    extract_task = PythonOperator(
        task_id="extract",
        python_callable=extract
    )

    transform_task = PythonOperator(
        task_id="transform",
        python_callable=transform
    )

    load_task = PythonOperator(
        task_id="load_to_mysql",
        python_callable=load
    )

    extract_task >> transform_task >> load_task
```

- DAG의 이름
- default_args : 위에서 설정한 Task의 기본설정
- description : 설명
- schedule_interval : 실행주기
- catchup : 백필 여부

순서는 extract_task >> transform_task >> load_task 순으로 진행



-github Repository에 참고하시면 위 코드가 있습니다.
[https://github.com/grant0808/airflow-dag-k8s](https://github.com/grant0808/airflow-dag-k8s)

&nbsp;

참고자료
- [https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html#dynamic-dags](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html#dynamic-dags)