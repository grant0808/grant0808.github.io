---
layout: post
title: "[Project] MLOps 구축"
date: 2025-03-25 17:04 +0900
description: MLOps 수준2로 만들어보기
category: [MLOps, Project]
tags: [MLOps, Project]
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---

## 개요
AI자동화라는 목표를 가지고 삼성청년 SW아카데미에서 백엔드와 프로젝트를 경험하였다. 수료 이후에는 MLOps에 대해서 공부하고 지속가능한 AI를 어떻게 구축하는지 약간이나마 이해할 수 있었다. 이제는 이론이 아닌 실전으로 내가 직접 아키텍처를 만들고 구동하는 것을 목표로 잡고 이 프로젝트를 진행하고자 한다.

![google-mlops2](/assets/img/mlops/project/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-4-ml-automation-ci-cd.svg)

google의 MLOps 수준 2: CI/CD파이프라인 자동화를 참고하였다.

## 시나리오
태양광 발전소에서 이상이 있는 장비를 파악하는 AI를 만들고 이를 MLOps를 통해 자동화를 시키는 것을 목표로 하고 있다. 모니터링을 통해 모델 예측에 이상이 있을 경우 다시 자동으로 재학습을 시키는 방법을 구현하고자 한다.


![architecture](/assets/img/mlops/project/mlops_architecture.png)


데이터 셋은 Kaggle에 있는 태양광 발전 시계열 데이터를 사용한다.

[데이터셋 링크](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data/data)

API를 통해 지속적으로 데이터가 들어오는 방식으로 하고싶었으나 시간과 비용적인 문제로 인하여 어느정도 경험을 하고 도전해보고자 한다. train과 test로 나누어 batch serving방식으로 시나리오를 구성한다.