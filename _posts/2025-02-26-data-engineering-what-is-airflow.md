---
layout: post
title: "[Data Engineering] Airflow 개념 및 구성"
date: 2025-02-26 17:46 +0900
description: Airflow의 개념 및 구성
category: [Data Engineering, Airflow]
tags: [Data Engineering, Airflow]
image:
  path: /assets/img/data_engineering/airflow/airflow.png
  alt: Airflow Logo
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---

## Airflow란?
Airflow의 처음은 Airbnb 엔지니어링 팀에서 2014년 10월부터 개발한 워크플로우 오픈소스 프로젝트입니다.

- 워크플로우(Work Flow)란? : 의존성으로 연결된 작업(Task)들의 집합
(ex) ETL의 경우 Extractaction > Transformation > Loading 의 작업의 흐름 / ELT도 가능

(참고) 워크플로우 오케스트레이션(Work Flow Orchestration) : 작업 흐름의 자동화를 위한 시스템, Airflow도 이에 해당

- 단일 프로세스부터 대규모 워크플로를 지원하는 분산 설정까지 다양한 방식으로 배포 가능

- 비슷한 기업 서비스
  - AWS MWAA(Managed Workflows for Apache Airflow)
  - GCP Composer

## Airflow 특징 및 구성 요소

- Dynamic : Airflow파이프라인은 Python코드로 구성되어 동적 파이프라인 생성이 가능
- Extensible : Operator, Executor를 통해 사용자 환경에 적합하도록 확장해 사용이 가능
- Elegant : 파이프라인이 간결하고 명시적, 스크립트 Parameter는 Jinja템플릿 엔진을 사용
- Scalable : 분산구조와 메시지 큐를 이용해 scale out과 워커 간의 오케스트레이션을 지원원


1. DAG(Directed Acyclic Graph)
    - 단방향으로 실행 방향을 결정하고 순차적으로 실행
    - 만약 양방향성을 가진다면 명확한 실행 Path가 없게되어 교착상태(deadlock)으로 이어짐

2. Operator
    - Operator는 미리 정의된 Task에 대한 템플릿으로 DAG내부에서 선언적으로 정의
    - Action Operators
      - 기능이나 명령을 실행하는 Operator
      - 실제 연산을 수행, 데이터 추출 및 프로세싱
    - Transfer Operater
      - 하나의 시스템을 다른 시스템으로 옮김(데이터를 Source에서 Destination으로 전송 등)
      - 예를 들어, Presto에서 MySQL로 데이터를 전송하는데에 사용
    - Sensor Operators
      - 조건이 만족할 때까지 지다렸다가, 조건이 충족되면 다음 Task를 실행시킴
    - 기본적인 Airflow Operators
      - PythonOperator
      - BashOperator
    - AWS, GCP에서 활용할 Operators
      - BigqueryOperators
      - Amazon EMROperators
      - KubernetesPodOperator

2. Task & Task Instance
    - Task
      - Operator를 실행하면 Task가 됨
    - Task Instance
      - 데이터 파이프 라인이 Trigger되어 실행될 때 생성된 Task를 Task Instance라고 함

3. Airflow component
    - Airflow Webserver : Airflow의 로그를 보여주거나 스케줄러(Scheduler)에 의해 생성된 DAG목록, Task상태 등을 시각화해주는 UI
    - Airflow Scheduler : Airflow로 할당된 work들을 스케줄링 해주는 component. DAG를 분석하고 현재 시점에서 DAG의 스케줄이 지난 경우 Airflow worker에 DAG의 task를 예약
    - Airflow Workers : 스케줄링된 task를 가져와서 실행
    - Airflow Executor : 실행중인 task를 handling하는 component. default설치시에는 scheduler에 있는 모든 것들을 다 실행시키지만, production 수준에서의 executor는 worker에게 task를 push
    - Airflow DataBase : DAG, Task, 기타 Variables, Connections 등의 metadata를 저장하고 관리

![diagram-dag](/assets/img/data_engineering/airflow/diagram_dag_processor_airflow_architecture.png)

## Single Node Architecture
![single-node-architecture](/assets/img/data_engineering/airflow/single-node-architecture.png)

- 한 서버 내(마스터 노드)에 Airflow의 구성요소가 실행
- LocalExecutor 사용
- 운영 용으로 사용할 수 없음

## Multi Node Architecture
![multi-node-architeucture](/assets/img/data_engineering/airflow/multi-node-architecture.png)

- Celery Broker(Queue) : Queue에 Task들을 담고 각 Worker노드에 Task를 받아 실행
- worker에 auto scaling적용
- 작업 부하가 커지면 워커를 scale out처리, 부하가 줄어들면 scale in
- CeleryExecutor 사용

- Redis : In Memory 방식이며 key-value데이터 구조 스토어. 빠른 read, write성능을 보장
- RabbitMQ : 메시지 브로커로 메시지의 우선순위를 지원. 크고 복잡한 메시지를 다룰 때 적합
- 속도가 중요하다면 Redis, 복잡한 메시지 처리에 필요하면 RabbitMQ
- 최근 Redis를 더 많이 사용하는 추세

## Airflow 기본 동작 원리

1. 유저가 새로운 Dag를 작성 → Dags Foolder 안에 py 파일 배치
2. Web Server와 Scheuler가 파싱하여 읽어옴
3. Scheduler가 Metastore를 통해 DagRun 오브젝터를 생성함
    1. DagRun은 사용자가 작성한 Dag의 인스턴스임
        ``DagRun Status : Running``
4. 스케줄러는 Task Instance Object를 스케줄링함
    1. Dag Run object의 인스턴스임
5. 트리거가 상황이 맞으면 Scheduler가 Task Instance를 Executor로 보냄
6. Exeutor는 Task Instance를 실행시킴
7. 완료 후 → MetaStore에 완료했다고 보고함
    1. 완료된 Task Instance는 Dag Run에 업데이트됨
    2. Scheduler는 Dag 실행이 완료되었는지 Metastore를 통해 확인 후에 Dag Run의 생태를 완료로 바꿈
    
        ``DagRun Status : Completed``
8. Metastore가 Webserver에 업데이트해서 사용자도 확인

## Airflow 장단점
- 장점
  - Python을 통하여 다양하고 복잡한 파이프라인을 만들 수 있다.(확장성)
  - 개별 태스크들에 대해 자세히 확인할 수 있고 시간에 따른 파이프라인 상황 확인 가능
  - 데이터 인프라 관리, 데이터 웨어하우스 구축, 머신러닝/분석/실험에 데이터 환경 구성에 유용함

- 단점
  - 배치처리에 사용할 수 있지만, 스트리밍에는 사용이 불가능함(Kafka, Spark Streaming을 함께 사용해서 할 수는 있으나 추천X)
  - Python 경험이 없으면 DAG 작성이 어려움
  - 작은 환경 변화에도 작동에 오류가 날 수 있음(민감함)
  - 버전 업그레이드 시 DAG작성 법이 달라질 수 있음

&nbsp;

참고자료
  - [https://velog.io/@sophi_e/Airflow-%EA%B8%B0%EC%B4%88-%EA%B0%9C%EB%85%90-%EB%B0%8F-%EC%9E%A5%EB%8B%A8%EC%A0%90](https://velog.io/@sophi_e/Airflow-%EA%B8%B0%EC%B4%88-%EA%B0%9C%EB%85%90-%EB%B0%8F-%EC%9E%A5%EB%8B%A8%EC%A0%90)
  - [https://github.com/apache/airflow](https://github.com/apache/airflow)
  - [https://airflow.apache.org/docs/apache-airflow/stable/index.html](https://airflow.apache.org/docs/apache-airflow/stable/index.html)
  - [https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/overview.html](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/overview.html)
  - [https://lsjsj92.tistory.com/631](https://lsjsj92.tistory.com/631)
  - [https://limspace.tistory.com/56](https://limspace.tistory.com/56)
  - [https://apache.googlesource.com/airflow-on-k8s-operator/+/HEAD/docs/design.md](https://apache.googlesource.com/airflow-on-k8s-operator/+/HEAD/docs/design.md)