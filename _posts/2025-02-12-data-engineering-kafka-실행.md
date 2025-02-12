---
layout: post
title: "[Data Engineering] Kafka 설치 및 실행"
date: 2025-02-12 19:25 +0900
description: Kafka 직접 설치하고 실행해보기
category: [Data Engineering, Kafka]
tags: [Data Engineering, Kafka]
pin: false
math: true
mermaid: true
---

## Kafka

### Kafka 설치에 앞서
- Java 8버전 이상

### Kafka 설치하기
```bash
wget https://mirror.navercorp.com/apache/kafka/3.9.0/kafka_2.13-3.9.0.tgz
```

명령 프롬프트에서 미러 사이트의 Kafka를 설치

만약 버전이 오래되었을 때 아래 사이트에서 찾아서 사용

[naver mirror사이트](https://mirror.navercorp.com/)

[kakao mirror사이트트](https://mirror.kakao.com/)

### kafka 압축풀기
```bash
tar -xvzf kafka_2.13-3.9.0.tgz

cd kafka_2.13-3.9.0.tgz
```

### zookeeper 설정 및 실행

```bash
vim config/config/zookeeper.properties
```

```bash
dataDir=/tmp/zookeeper # zookeeper 기본 데이터 폴더
clientPort=2181  # zookeeper port
maxClientCnxns=0 # client 수 설정 0이면 unlimit
```

```bash
# zookeeper 실행
# Mac은 아래 코드를 사용(앞에 bash대신 ./사용)
./bin/zookeeper-server-start.sh -daemon config/zookeeper.properties

# 작성자는 Window를 사용함으로 sh파일을 실행하기 위해서는 git bash를 이용
bash bin/zookeeper-server-start.sh -daemon config/zookeeper.properties

# 종료
bash kafka/bin/zookeeper-server-stop.sh config/zookeeper.properties
```
-daemon 백그라운드 옵션

<br>

**만약 아래와 같은 에러가 발생한 경우**
![img](/assets/img/data_eigineering/kafka/zookeeper_error.png)
- 파일경로에 공백이 있는지 확인하고 공백이 있으면 제거하거나 대체한다.

### kafka 설정

```yaml
broker.id=0     # kafka broker id
log.dirs=/tmp/kafka-logs    # kafka broker log file 폴더
num.partitions=1    # kafka topic 만들 때 default partition 설정
log.retention.hours # kafka message 저장 기간
log.retention.bytes # partition의 크기 , 크기 초과되면 삭제됌
# log.retention.bytes x partition의 수는 topic의 최대크기. 이 값이 -1이면 크기는 unlimit

zookeeper.connect=localhost:2181 # 연동할 zookeeper host 및 port
```

카프카에 대한 설정 외에도 클러스터라면 broker.id를 지정해줘야 하고 zookeeper.connect에도 호스트와 포트를 지정해 줘야함

### Kafka 실행

```bash
bash bin/kafka-server-start.sh -daemon config/server.properties

#종료
bash kafka/bin/kafka-server-stop.sh config/server.properties
```

### Topic 생성
```bash
bash bin/kafka-topics.sh --create --bootstrap-server localhost:9092 -replication-factor 1 --partitions 3 --topic topic
```
- bootstrap-server : kafka 주소

- replication-factor : broker에 복사되는 갯수 (안정성 증가)

### Topic 목록 확인
```bash
bash bin/kafka-topics.sh --list --bootstrap-server localhost:9092
```

### Topci 상세정보
```bash
bash bin/kafka-topics.sh --describe --bootstrap-server localhost:9092 --topic topic
```

### Topic 삭제
```bash
bash bin/kafka-topics.sh --delete -topic topic --bootstrap-server localhost:9092
```