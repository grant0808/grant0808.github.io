---
layout: post
title: "[Data Engineering] UI for Kafka 사용"
date: 2025-02-13 15:50 +0900
description: UI for Kafka 사용해보기
category: [Data Engineering, Kafka]
tags: [Data Engineering, Kafka, UI for Kafka]
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---

## UI for Kafka
[UI for Kafka](https://github.com/provectus/kafka-ui) 의 주소에 들어가면 자세한 정보를 확인할 수 있다.
### Docker Compose 설정 파일로 zookeeper와 kafka 구동시키기
```yaml
services:
  zookeeper-1:
    image: confluentinc/cp-zookeeper:latest
    ports:
      - '2181:2181'
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka-1:
    image: confluentinc/cp-kafka:latest
    ports:
      - '9091:9091'
    depends_on:
      - zookeeper-1
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper-1:2181
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: INTERNAL
      KAFKA_ADVERTISED_LISTENERS: INTERNAL://kafka-1:29091,EXTERNAL://localhost:9091
      KAFKA_DEFAULT_REPLICATION_FACTOR: 3
      KAFKA_NUM_PARTITIONS: 3

  kafka-2:
    image: confluentinc/cp-kafka:latest
    ports:
      - '9092:9092'
    depends_on:
      - zookeeper-1
    environment:
      KAFKA_BROKER_ID: 2
      KAFKA_ZOOKEEPER_CONNECT: zookeeper-1:2181
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: INTERNAL
      KAFKA_ADVERTISED_LISTENERS: INTERNAL://kafka-2:29092,EXTERNAL://localhost:9092
      KAFKA_DEFAULT_REPLICATION_FACTOR: 3
      KAFKA_NUM_PARTITIONS: 3

  kafka-3:
    image: confluentinc/cp-kafka:latest
    ports:
      - '9093:9093'
    depends_on:
      - zookeeper-1
    environment:
      KAFKA_BROKER_ID: 3
      KAFKA_ZOOKEEPER_CONNECT: zookeeper-1:2181
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: INTERNAL
      KAFKA_ADVERTISED_LISTENERS: INTERNAL://kafka-3:29093,EXTERNAL://localhost:9093
      KAFKA_DEFAULT_REPLICATION_FACTOR: 3
      KAFKA_NUM_PARTITIONS: 3
```

**zookeeper, kafka 실행**
```bash
docker-compose -f doker-compose.yaml up -d
```
실행하면 아래와 같이 docker에 띄어진 것을 확인할 수 있다.

![img](/assets/img/data_eigineering/kafka/kafka_in_docker.png)


### UI for Kafka 실행
```yaml
services:
  kafka-ui:
    image: provectuslabs/kafka-ui
    container_name: kafka-ui
    ports:
      - "8989:8080"
    restart: always
    environment:
      - KAFKA_CLUSTERS_0_NAME=local
      - KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS=kafka-1:29091,kafka-2:29092,kafka-3:29093
      - KAFKA_CLUSTERS_0_ZOOKEEPER=zookeeper-1:2181
```
**ui for kafka 실행**
```bash
docker-compose -f doker-compose-kafka.yaml up -d
```
[http://localhost:8989/](http://localhost:8989/)에 접근하면 ui for kafka에 들어갈 수 있다.

![img](/assets/img/data_eigineering/kafka/ui_for_kafka.png)

## Kafka 내부 접속
```bash
docker exec -it [CONTAINER ID] bash
```
### Topic 생성
```bash
kafka-topics --create --bootstrap-server kafka-1:29091 --replication-factor 3 --partitions 3 --topic topic1
```

### 생성된 Topic 확인
```bash
kafka-topics --list --bootstrap-server kafka-1:29091
```

### Topic 상세 정보
```bash
kafka-topics --describe --bootstrap-server kafka-1:29091 --topic topic1
```
### 토픽 삭제
```bash
kafka-topics --delete --bootstrap-server kafka-1:29091 --topic topic1
```
### Producer 실행
```bash
kafka-console-producer --broker-list kafka-1:29091 --topic topic1
```
실행 후 직접 메시지를 입력하면 Kafka에 전송됨
![img](/assets//img/data_eigineering/kafka/producer.png)

### Consumer 실행
```bash
kafka-console-consumer --bootstrap-server kafka-1:29091 --topic topic1 --from-beginning
```
--from-beginning: 처음부터 모든 메시지 조회
![img](/assets/img/data_eigineering/kafka/consumer.png)

### Consumer 그룹 ID 지정
```bash
kafka-console-consumer --bootstrap-server kafka-1:29091 --topic topic1 --group my-group --from-beginning
```
특정 컨슈머 그룹(my-group)으로 메시지를 읽음

###  Consumer 그룹 목록 확인
```bash
kafka-consumer-groups --bootstrap-server kafka-1:29091 --list
```
###  특정 Consumer 그룹의 상태 확인
```bash
kafka-consumer-groups --bootstrap-server kafka-1:29091 --group my-group --describe
```
해당 그룹의 Offset, Lag(지연 메시지 개수), 파티션 정보 확인 가능
###  Consumer Offset 리셋
```bash
kafka-consumer-groups --bootstrap-server kafka-1:29091 --group my-group --reset-offsets --to-earliest --execute --topic topic1
```
--to-earliest: 가장 오래된 메시지부터 다시 읽도록 설정

&nbsp;

참고자료

- [https://devocean.sk.com/blog/techBoardDetail.do?ID=163980](https://devocean.sk.com/blog/techBoardDetail.do?ID=163980)
- [https://velog.io/@sangkyu-bae/Docker-Compose%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-Kafka%EB%A5%BC-%EC%8B%A4%ED%96%89%ED%95%B4%EB%B3%B4%EC%9E%90](https://velog.io/@sangkyu-bae/Docker-Compose%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-Kafka%EB%A5%BC-%EC%8B%A4%ED%96%89%ED%95%B4%EB%B3%B4%EC%9E%90)