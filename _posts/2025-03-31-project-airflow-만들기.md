---
layout: post
title: "[Project] Airflow 만들기 에러사항"
date: 2025-03-31 18:22 +0900
description: Kubernetes에 Airflow를 띄우면서 생긴 에러
category: [MLOps, Project]
tags: [MLOps, Project, Airflow]
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---


# 트러블이슈

## 문제 1번

1. ``The scheduler does not appear to be running. The DAGs list may not update, and new tasks will not be scheduled.``가 발생

scheduler를 확인해보면 실행되지가 않은 것을 발견

```bash
kubectl get pods -n airflow | grep scheduler

kubectl logs <airflow-scheduler-pod> -c git-sync -n airflow
```

```bash
kubectl describe pod <airflow-scheduler-pod> -n airflow
```

git sync가 CrashLoopBackOff상태인 것을 확인

https://velog.io/@jskim/-K8S-Pod-%EC%9E%A5%EC%95%A0-%EC%A7%84%EB%8B%A8-%EB%B0%A9%EB%B2%95

위는 kubernetes engine안에 ssh를 만들어서 다시 시작 -> 해결

2. 이번엔 실행 시 runing이 되지않고 log가 보이지 않은 경우

worker의 describe를 확인하면 

마지막 메시지에 
``running PreBind plugin "VolumeBinding": binding volumes: context deadline exceeded``가 있다.

```bash
kubectl get pvc -n airflow
```

logs-airflow-worker-0의 상태가 Pending이면 PV가 없다는 뜻

확인결과 ``values.yaml``에서 ``logs.persistence.enabled: false``로 설정했어도,
Helm chart가 기존 PVC인 logs-airflow-worker-0를 참조하고 있는 StatefulSet 설정을 유지하고 있는 상태


values.yaml파일 수정 및 업데이트
```bash
logs:
  persistence:
    enabled: false
workers:
  persistence:
    enabled: false

```



## 문제 2

log가 잘 나오다가 

처음에는 log가 잘나오다가

train-train-task-nyw2u3ru
*** Could not read served logs: HTTPConnectionPool(host='train-train-task-nyw2u3ru', port=8793): Max retries exceeded with url: /log/dag_id=train/run_id=manual__2025-04-02T15:24:07.231328+00:00/task_id=train_task/attempt=2.log (Caused by NameResolutionError("<urllib3.connection.HTTPConnection object at 0x7c74717319a0>: Failed to resolve 'train-train-task-nyw2u3ru' ([Errno -2] Name or service not known)"))

이런 에러가 발생했습니다.

원인 : Task pod이 실행 직후 삭제됨 ->  Webserver가 pod 존재 확인 전에 로그 요청