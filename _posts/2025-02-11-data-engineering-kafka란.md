---
layout: post
title: "[Data Engineering] Kafka란?"
date: 2025-02-11 16:26 +0900
description: Kafka가 왜 사용되고 어떤 구조를 가지고 있는지 알아보았습니다.
image:
  path:
  alt:
category: [Data Enginnering, Kafka]
tags: [Data Enginnering, Kafka]
pin: false
math: true
mermaid: true
---
# Kafka 사용 배경
![img](/assets/img/data_eigineering/kafka/kafka.png)

기업의 MSA(Microservice Architecture)환경에서 하나의 기능에 다수의 서버가 운영되면서 데이터 흐름 복잡성과 운영유지비가 증가하고 있다.

이를 해결하고자 링크드인에서 2011년 Kafka라는 분산 스트리밍 플랫폼을 개발하였다. 이는 스트림 처리, 실시간 데이터 파이프라인 및 대규모 데이터 통합에 사용되었다. 모든 데이터는 로그 형식으로 실시간 기록되며 변경이 불가능 하다.

### Kafka의 장점
- 높은 처리량
- 높은 확장성
- 낮은 대기시간
- 데이터 보관
- 고가용성

### Kafka의 단점
- 높은 복잡도
- 서버 비용(기본적으로 3개의 브로커 필요)

## Kafka 용어
- 메시지(Message) : Kafka에서 주고받는 데이터 단위로 기본은 Byte의 배열
- 브로커(Broker) : Kafka에 설치되어 있는 서버 또는 노드드
- 토픽(Topic) : 데이터의 주제를 나타내며, 이름으로 분리된 로그. 서로 다른 요도의 메세지들이 섞이는 것을 방지하기 위해 세팅된 저장소
- 파티션(Partition) : 토픽은 하나 이상의 파티션으로 나누어질 수 있으며, 각 파티션은 순서가 있는 연속된 메시지의 로그. 파티션은 병렬 처리를 지원하고, 데이터의 분산 및 복제를 관리. 
(주의)파티션을 늘릴 수 있으나, 줄이는 것은 불가능.
- 레코드(Record) : 데이터의 기본 단위로 키와 값(key-value pair)구성. 파티션에 들어가 저장되며 FIFO 형식으로 처리됨.
- 오프셋(Offset) : 특정 파티션 내의 레코드 위치를 식별하는 값
- 프로듀서(Producer) : 토픽에서 데이터를 보내는 역할. 메시지를 생성하고 토픽에 보내는 서버 또는 애플리케이션
- 컨슈머(Consumer) : 토픽에서 데이터를 읽는 역할. 특정 토픽의 메시지를 가져와서 처리함. 컨슈머 그룹은 여러 개의 컨슈머 인스턴스를 그룹화하여 특정 토픽의 파티션을 공유하도록 구성. 이를 통해 데이터를 병렬로 처리하고 처리량을 증가시킬 수 있음.
- 카프카 커넥터(Connector) : 카프카와 외부 시스템을 연동 시 쉽게 연동 가능하도록 하는 프레임워크로 MySQL, S3 등 다양한 프로토콜과 연동을 지원
  - 소스 커넥터(source connector) : 메시지 발행과 관련 있는 커넥터
  - 싱크 커넥터(sink connector) : 메시지 소비와 관련 있는 커넥터
- 주키퍼(ZooKeeper) : 분산 코디네이션 서비스(분산 시스템에서 시스템 간의 정보 공유, 상태 체크, 서버들 간의 동기화를 위한 락 등을 처리해주는 서비스)를 제공하는 오픈소스 프로젝트. Leader와 Follower로 설정(Leader에 저장되면 Follower가 가져와 복제)

![img](/assets/img/data_eigineering/kafka/kafka-terminology.png)

## Kafka Segment-Record 구조
- Header
  - Key-Value 데이터를 추가할 수 있다.
  - 처리에 참고할 정보를 담을 수 있다.
- Timestamp
  - 시간을 저장(Unix timestamp 포함)
  - 기본적으로 ProductRecord 생성시간(Create Time)
  - BrokerLoading 시간으로도 설정 가능
  - Topic 단위 설정
- Key
  - Key를 분류하기 위한 목적
  - 메시지 키는 Partitioner에 의해 Topic의 Partition번호로 지정
  - 키 값이 없으면 `null`
  - Round Robin(우선순위를 두지 않고, 순서대로 시간단위로 CPU를 할당하는)방식
- Value
  - 실제 값이 들어가는 곳
  - Float ByteArray, String 지정 가능
  - 어떤 포맷으로 직렬화 되었는지 Consumer는 알지 못함. 역직렬화 포맷을 미리 알고 있어야 함
  - 보통 String으로 직렬화 역질화 또는 Json
- Offset
  - Producer가 생성한 Record에는 존재하지 않는다.
  - Broker에 적재되면서 Offset이 지정
  - Offset 기반으로 Consumer가 처리리


![img](/assets/img/data_eigineering/kafka/kafka-segment-record.png)   

&nbsp;

참고자료
- https://techblog.woowahan.com/17386/
- https://medium.com/@david-noxer-kor/%EC%8B%A4%EB%AC%B4%EC%97%90%EC%84%9C-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94-%EA%B8%B0%EC%88%A0-%EA%B0%9C%EB%85%90-%EC%A0%95%EB%A6%AC-case-1-kafka-6e74ec7ce042
- https://velog.io/@chan9708/Kafka-Kafka-Record-Segmenthttps://velog.io/@chan9708/Kafka-Kafka-Record-Segment