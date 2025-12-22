---
layout: post
title: "[MLOps] MLFlow 정리"
date: 2025-12-19 18:02 +0900
description: MLOps를 위한 MLFlow 정리
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

## MLflow란?
![MLflow](/assets/img/mlops/project/mlflow/mlflow-logo.png)

- 머신러닝 프로젝트를 관리하기 위한 오픈 소스 플랫폼
- Project Lifecycle의 모든 단계에서 적용할 수 있으며, 개발/학습/추론/배포에 이르는 모든 단계에서 사용할 수 있음
- MLFlow는 모델 학습에 집중한 플랫폼
- 단, MLFlow는 MLOps의 모든 기능을 제공하는 것은 아님. 추가적인 MLOps지원 기능과 integration할 필요가 있음. (MLOps 1단계에 사용할 것을 권장)

### 구성요소
- MLFlow Tracking : 머신러닝 모델 학습 결과를 추적하고,다양한 프레임워크에서 동작할 수 있는 학습코드의 재현성을 보장하는 기능
- MLFlow Projects : 머신러닝 프로젝트의 코드, 환경설정, 종속성 등을 관리
- MLFlow Models : 학습된 머신러닝 모델을 관리하고, 다양한 환경에서 모델을 배포할 수 있는 기능 제공
- MLFlow Registry : 모델 버전을 관리하고, 공동작업을 위한 모델저장소를 제공

### MLFlow Tracking
- 기계 학습 프로젝트에서 발생하는 모든 실험의 메타데이터를 추적하고 저장
- 데이터 과학자들이나 엔지니어들은 다양한 실험 간의 성능을 비교하고 분석 할 수 있음

주요 기능
- 메타 데이터 관리
- Backend Store

### MLFlow Projects
- ML모델을 포장하고 재현 가능한 방식으로 관리하여, 다양한 환경에서 일관된 결과를 얻을 수 있도록 지원

주요 기능
- 재생산 가능한 ML실행
- Dependency정의
- 실행 API

### MLFlow Registry
- ML모델의 다양한 버전과 스테이지를 중앙화된 방식으로 관리하고, 이러한 변화를 추적

주요 기능
- 중앙화된 Repository
- 모델 staging
- 변화 관리와 모니터링