---
layout: post
title: "[Data Engineering] Apache Hadoop 개념 및 구조"
date: 2025-03-17 20:22 +0900
description: Hadoop의 개념 및 구조
category: [Data Engineering, Hadoop]
tags: [Data Engineering, Hadoop]
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---

## Apache Hodoop이란?
Apache Hadoop 소프트웨어 라이브러리는 간단한 프로그래밍 모델을 사용하여 컴퓨터 클러스터에서 대규모 데이터 세트를 분산 처리할 수 있는 프레임워크입니다. 단일 서버에서 수천 대의 머신으로 확장되도록 설계되었으며, 각각 로컬 계산 및 스토리지를 제공합니다. 고가용성을 제공하기 위해 하드웨어에 의존하는 대신 라이브러리 자체는 애플리케이션 계층에서 오류를 감지하고 처리하도록 설계되어 각각 오류가 발생하기 쉬운 컴퓨터 클러스터 위에 고가용성 서비스를 제공합니다.

## Hadoop 핵심 구성요소

### HDFS(Hadoop Distributed File System)
Hadoop 분산 파일 시스템(HDFS)은 상용 하드웨어에서 실행되도록 설계된 분산 파일 시스템입니다. 기존 분산 파일 시스템과 많은 유사점이 있습니다. 그러나 다른 분산 파일 시스템과의 차이점은 상당합니다. HDFS는 내결함성이 매우 뛰어나고 저비용 하드웨어에 배포되도록 설계되었습니다. HDFS는 애플리케이션 데이터에 대한 높은 처리량 액세스를 제공하며 대규모 데이터 세트가 있는 애플리케이션에 적합합니다. HDFS는 몇 가지 POSIX 요구 사항을 완화하여 파일 시스템 데이터에 대한 스트리밍 액세스를 가능하게 합니다. HDFS는 원래 Apache Nutch 웹 검색 엔진 프로젝트의 인프라로 구축되었습니다.

![hdfs architecture](/assets/img/data_engineering/hadoop/hdfsarchitecture.png)

HDFS는 마스터/슬레이브 아키텍처를 가지고 있습니다. HDFS 클러스터는 파일 시스템 네임스페이스를 관리하고 클라이언트의 파일 액세스를 조절하는 마스터 서버인 단일 NameNode로 구성됩니다. 또한 클러스터의 노드당 일반적으로 하나씩 여러 개의 DataNode가 있으며, 이는 실행되는 노드에 연결된 스토리지를 관리합니다. HDFS는 파일 시스템 네임스페이스를 노출하고 사용자 데이터를 파일에 저장할 수 있도록 합니다. 

내부적으로 파일은 하나 이상의 블록으로 분할되고 이러한 블록은 DataNode 세트에 저장됩니다. NameNode는 파일 및 디렉터리 열기, 닫기 및 이름 바꾸기와 같은 파일 시스템 네임스페이스 작업을 실행합니다. 또한 블록과 DataNode의 매핑을 결정합니다. DataNode는 파일 시스템 클라이언트의 읽기 및 쓰기 요청을 처리합니다. DataNode는 또한 NameNode의 지시에 따라 블록 생성, 삭제 및 복제를 수행합니다.

![hdfs node](/assets/img/data_engineering/hadoop/hdfsdatanodes.png)

HDFS는 대규모 클러스터의 머신에 걸쳐 매우 큰 파일을 안정적으로 저장하도록 설계되었습니다. 각 파일을 블록 시퀀스로 저장합니다. 파일의 블록은 장애 허용을 위해 복제됩니다. 블록 크기와 복제 계수는 파일별로 구성할 수 있습니다.

마지막 블록을 제외한 파일의 모든 블록은 크기가 동일하며, append 및 hsync에 가변 길이 블록 지원이 추가된 후에는 사용자는 구성된 블록 크기에 맞게 마지막 블록을 채우지 않고도 새 블록을 시작할 수 있습니다.

애플리케이션은 파일의 복제본 수를 지정할 수 있습니다. 복제 요소는 파일 생성 시 지정할 수 있으며 나중에 변경할 수 있습니다. HDFS의 파일은 한 번만 쓸 수 있으며(추가 및 잘라내기 제외) 항상 한 명의 작성자가 있습니다.

NameNode는 블록 복제와 관련된 모든 결정을 내립니다. 클러스터의 각 DataNode에서 주기적으로 Heartbeat와 Blockreport를 수신합니다. Heartbeat를 수신하면 DataNode가 제대로 작동하고 있음을 의미합니다. Blockreport에는 DataNode의 모든 블록 목록이 들어 있습니다.

### HDFS의 장점

1. **내결함성**HDFS는 오류를 탐지하고 자동으로 신속히 복구하도록 설계되어 있으므로 지속성과 안정성을 보장합니다.

2. **속도**클러스터 아키텍처이기 때문에 초당 2GB의 데이터를 유지관리할 수 있습니다.

3. **더 많은 유형의 데이터에 액세스**(특히, 스트리밍 데이터) 일괄 처리를 위해 대량의 데이터를 다루도록 설계되었기 때문에 높은 데이터 처리율을 달성하여 스트리밍 데이터 지원에 이상적입니다.

4. **호환성과 이동성** HDFS는 다양한 하드웨어 설정으로 이동이 가능하고 여러 기본 운영 체제와 호환되도록 설계되어 궁극적으로 사용자가 맞춤형 설정으로 HDFS를 사용할 수 있습니다. 이와 같은 장점은 특히 빅데이터를 다룰 때 중요한 의미가 있으며, HDFS가 데이터를 다루는 특별한 방식 때문에 가능한 일입니다.

5. **확장성** 파일 시스템 크기에 맞춰 리소스 규모를 조정할 수 있습니다. HDFS에는 수직 및 수평 확장성 메커니즘이 포함되어 있습니다.

6. **데이터 지역성** 하둡 파일 시스템은 데이터가 계산 단위가 있는 위치로 이동하는 것과 달리 데이터는 데이터 노드에 상주합니다. 이를 통해 데이터와 컴퓨팅 프로세스 사이의 거리를 단축하여 네트워크 혼잡을 줄이고 시스템을 보다 효과적이고 효율적으로 만듭니다.

7. **비용 효율** 처음에는 데이터를 생각할 때 값비싼 하드웨어와 대역폭 점유를 생각할 수 있습니다. 하드웨어 오류가 발생하면 수정하는 데 많은 비용이 발생할 수 있습니다. HDFS를 사용하면 데이터가 가상으로 저렴하게 저장되므로 파일 시스템 메타데이터 및 파일 시스템 네임스페이스 데이터 스토리지 비용을 크게 줄일 수 있습니다. 또한, HDFS는 오픈 소스이기 때문에 라이선스 비용에 대해 걱정할 필요가 없습니다.

8. **방대한 양의 데이터 저장** 데이터 스토리지는 HDFS가 개발된 이유이며 모든 종류와 크기의 데이터, 특히 데이터 저장에 어려움을 겪고 있는 기업의 방대한 데이터를 저장하는 데 유용합니다. 여기에는 정형 데이터와 비정형 데이터가 모두 포함됩니다.

9. **유연성** 다른 기존 스토리지 데이터베이스와 달리 수집된 데이터를 저장하기 전에 별도로 처리할 필요가 없습니다. 원하는 만큼 데이터를 저장할 수 있으며, 데이터로 무엇을 하고 싶은지, 나중에 어떻게 사용할지 정확히 결정할 수 있습니다. 여기에는 텍스트, 비디오 및 이미지와 같은 비정형 데이터도 포함됩니다.

### MapReduce
Hadoop MapReduce는 범용 하드웨어의 대규모 클러스터(수천 개의 노드)에서 엄청난 양의 데이터(수 테라바이트 규모의 데이터 세트)를 안정적이고 장애에 강한 방식으로 병렬로 처리하는 애플리케이션을 쉽게 작성할 수 있는 소프트웨어 프레임워크입니다.

MapReduce 작업은 일반적으로 입력 데이터 세트를 독립적인 청크로 분할하여 맵 작업 에서 완전히 병렬 방식으로 처리합니다. 프레임워크는 맵의 출력을 정렬한 다음 reduce 작업 에 입력합니다 . 일반적으로 작업의 입력과 출력은 모두 파일 시스템에 저장됩니다. 프레임워크는 작업 스케줄링, 모니터링 및 실패한 작업 재실행을 담당합니다.

일반적으로 컴퓨트 노드와 스토리지 노드는 동일합니다. 즉, MapReduce 프레임워크와 Hadoop 분산 파일 시스템이 동일한 노드 집합에서 실행됩니다. 이 구성을 통해 프레임워크는 데이터가 이미 있는 노드에서 작업을 효과적으로 스케줄링할 수 있어 클러스터 전체에서 매우 높은 집계 대역폭이 생성됩니다.

MapReduce 프레임워크는 단일 마스터 ResourceManager, NodeManager클러스터 노드당 하나의 워커, MRAppMaster애플리케이션당 하나의 워커로 구성됩니다

![mapreduce](/assets/img/data_engineering/hadoop/mapreduce.png)

### MapReduce단계

- Maping 단계
분할과 매핑이라는 두 단계가 있습니다. 입력 파일은 효율성을 위해 더 작고 동일한 청크로 나뉘며 이를 입력 분할이라고 합니다. 매퍼는 (키, 값) 쌍만 이해하므로 Hadoop은 TextInputFormat을 사용하여 입력 분할을 키-값 쌍으로 변환하는 RecordReader를 사용합니다. 

MapReduce에서 병렬성은 Mapper에 의해 달성됩니다. 각 입력 분할에 대해 매퍼의 새 인스턴스가 인스턴스화됩니다. 매핑 단계에는 이러한 데이터 블록에 적용되는 코딩 로직이 포함됩니다. 이 단계에서 매퍼는 키-값 쌍을 처리하고 동일한 형태(키-값 쌍)의 출력을 생성합니다.

- shuffle 및 sort 단계
Shuffle과 sort는 Mapper와 Reducer 사이의 MapReduce의 중간 단계로, Hadoop에서 처리하며 필요한 경우 재정의할 수 있습니다. Shuffle 프로세스는 Mapper 출력의 키 값을 그룹화하여 모든 Mapper 출력을 집계하고 값은 값 목록에 추가됩니다. 따라서 Shuffle 출력 형식은 map <key, List < list of values > >이 됩니다. Mapper 출력의 키는 통합되고 정렬됩니다.

- Reduce 단계
셔플 및 정렬 단계의 출력은 리듀서 단계의 입력으로 사용되고 리듀서는 값 목록을 처리합니다. 각 키는 다른 리듀서로 전송될 수 있습니다. 리듀서는 값을 설정할 수 있으며, 이는 MapReduce 작업의 최종 출력에 통합되고 값은 HDFS에 최종 출력으로 저장됩니다.


&nbsp;

참고자료
  - [https://hadoop.apache.org/docs/stable/](https://hadoop.apache.org/docs/stable/)
  - [https://www.ibm.com/kr-ko/topics/hadoop](https://www.ibm.com/kr-ko/topics/hadoop)
  - [https://aws.amazon.com/ko/compare/the-difference-between-hadoop-vs-spark/](https://aws.amazon.com/ko/compare/the-difference-between-hadoop-vs-spark/)
  - [https://www.databricks.com/kr/glossary/hadoop-distributed-file-system-hdfs](https://www.databricks.com/kr/glossary/hadoop-distributed-file-system-hdfs)
  - [https://www.whizlabs.com/blog/understanding-mapreduce-in-hadoop-know-how-to-get-started/](https://www.whizlabs.com/blog/understanding-mapreduce-in-hadoop-know-how-to-get-started/)