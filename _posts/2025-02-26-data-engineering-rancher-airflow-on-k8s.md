---
layout: post
title: "[Data Engineering] Rancher를 이용한 Airflow구축하기 (실패)"
date: 2025-03-02 15:33 +0900
description: Rancher를 이용한 Airflow구축 실습
category: [Data Engineering, Airflow]
tags: [Data Engineering, Airflow, Rancher]
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---

## Rancher란?
Rancher는 어디서나 모든 공급자에게 클러스터를 배포하고 실행할 수 있는 Kubernetes 관리 도구입니다.

Rancher는 쿠버네티스를 위한 완벽한 컨테이너 관리 플랫폼으로, 어디서나 쿠버네티스를 성공적으로 실행할 수 있는 도구를 제공합니다.

- Docker에서 실행하기
```bash
docker run --privileged -d --restart=unless-stopped -p 80:80 -p 443:443 rancher/rancher
```
``--privileged`` : 호스트 시스템의 전체 권한을 부여하여 컨테이너 내부에서 호스트 시스템에 접근

[http://localhost:80](http://localhost:80)접근

![rencher-login](/assets/img/data_engineering/rencher/rencher_login.png)

왼쪽 문구를 살펴보면 password를 찾는 명령어를 확인할 수 있습니다.

```bash
docker logs  container-id  2>&1 | grep "Bootstrap Password:"
```
명령어로 password를 찾고 입력하면 아래의 그림과 같이 나옵니다.

![welcome](/assets/img/data_engineering/rencher/rencher_welcome.png)

Set a specific password to use를 클릭하여 password를 변경합니다.

비밀번호를 설정할 때 12자리를 넘지않으면 ``Password must be at least 12 characters``가 나옵니다. 그러므로 12자리 이상으로 비밀번호를 변경합니다.

![나중에 없애기]```qlalfqjsgh1234```

![main-page](/assets/img/data_engineering/rencher/main-page.png)
들어가면 메인 페이지를 확인할 수 있습니다.
이때 Cluster 오른쪽 create를 클릭하여 Cluster를 만듭니다.

![cluster-page](/assets/img/data_engineering/rencher/cluster-page.png)
위에 Kubernetes를 만들 수 있는 페이지가 나오는데 위 중 자신에게 편한 것을 선택하면 됩니다.
저는 Google GKE를 선택합니다.

이때는 각 Cluster 마다 계정을 요구하기에 이에 맞춰서 진행하면 됩니다.

Google GKE는 다음을 요구합니다다.
1. Cluster name
2. Service Account(Json 파일)

위 단계만 맞추고 개인적으로 Cluster를 만들기위한 설정만 하면 10분에서 20분 사이에 Cluster가 생성됩니다.

## Error1
![error1](/assets/img/data_engineering/rencher/create-cluster-error.png)
생성은 되었으나 ready부분에서 cluster agent와 연결이 되어있지 않다는 것을 확인했습니다.

3일동안 검색을 하고 AWS로 바꿔 사용해보았으나 실패하여 나중에 다시 해보는 걸로 결정하였다.

고로 Local에서 진행해보도록 하겠습니다.

## Repository만들기
Cluster에 들어가 Apps에 Repositories에 들어갑니다.
Create를 눌러 [https://airflow-helm.github.io/charts](https://airflow-helm.github.io/charts)를 URL에 넣습니다.

그럼 Repositories에 등록이 된 것을 알 수 있습니다.

![repository](/assets/img/data_engineering/rencher/repository.png)

등록한 후 Apps에 Charts에 들어가 Airflow를 검색하면 Ariflow 가 나타나는 것을 확인할 수 있습니다. Airflow를 다운로드합니다.

## Error2

![pod-error](/assets/img/data_engineering/rencher/pod.png)
이번에 airflow-Postgresql가 pending되는 현상이 발생했다. 검색결과 PVC(Persistent Volume Claim)이 원인이라고 되어있어 pvc 상태를 확인하고 메모리 부족인지 확인했습니다.

Reason으로 FailedScheduling이 되어있어. Node를 늘리는 방법을 고려해야하지만, Local로 진행하였기에 따로 Custom을 만들어 진행하고자 하였으나 K3S에서는 Node를 늘릴 수 없습니다.

## 결과
Rancher에서 Kubernetes를 띄워 Airflow를 실행하고자 하였으나 시간이 너무 많이 지체하여서 여기까지만 하도록하겠습니다.
언제가 걸리지는 몰라도 Rancher가 아닌 다른 방법으로 구축을 해보고 시간이 남는다면 다시한번 도전해보겠습니다.

- 25.02-26 ~ 25.03.03


&nbsp;
참고자료
  - [https://velog.io/@leesjpr/Rancher-%EC%97%90%EC%84%9C-Kubernetes-%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0-%EA%B5%AC%EC%B6%95With-GKE](https://velog.io/@leesjpr/Rancher-%EC%97%90%EC%84%9C-Kubernetes-%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0-%EA%B5%AC%EC%B6%95With-GKE)