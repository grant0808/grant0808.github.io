---
layout: post
title: "[Data Engineering] ELK란?"
date: 2025-02-17 17:32 +0900
description: ELK가 무엇인지 정리해보았다
category: [Data Engineering, Elastic Search]
tags: [Data Engineering, Elastic Search, ELK]
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---
# ELK Stack

![elk](/assets/img/data_eigineering/elastic_search/elk.webp)

- E (Elasticsearch) : 검색 및 분석 엔진
- L (Logstash) : Data 처리 Pipeline 역할
- K (Kibana) : 데이터 시각화 

# Logstash
![logstash](/assets/img/data_eigineering/elastic_search/logstash.png)

Logstash는 실시간 파이프라인 기능을 가진 오픈소스 데이터 수집 엔진입니다. Logstash는 서로 다른 소스의 데이터를 탄력적으로 통합하고 사용자가 선택한 목적지로 데이터를 정규화할 수 있습니다. 다양한 고급 다운스트림 분석 및 시각화 활용 사례를 위해 모든 데이터를 정리하고 대중화(democratization)합니다.

**강점**
- 다양한 소스에서 데이터를 수집하여 변환한 후 자주 사용하는 저장소로 전달하는 기능
- 플러그형 파이프라인 아키텍처
  - 파이프라인을 구성하는 각 요소들 전부 플러그인 형태
  - 다양한 입력, 필터, 출력을 믹스매치하고 조정하면서 파이프라인에서 조화롭게 운용
- 커뮤니티에서 확장 가능하고 개발자에게 편리한 플러그인 에코시스템
  - 200여 개 플러그인 사용 가능, 또한 직접 플러그인을 만들어 제공할 수 있는 유연성
- 성능
  - 자체적 내장 메모리, 파일 기반의 큐를 통해 안전성이 높고 처리속도가 빠름
  - 벌크 인덱싱 및 파이프라인 배치 크기 조정을 통한 병목현상을 방지, 성능 최적화 가능
- 안정성
  - Elasticsearch의 장애 상황에 대응하기 위한 재시도 로직이나 오류가 발생한 Document를 따로 보관하는 Dead-Letter-Queue를 내장

## Pipeline
![pipeline](/assets/img/data_eigineering/elastic_search/basic_logstash_pipeline.png)

- Input, Output은 필수, Filter는 Option.
- 입력 플러그인은 소스의 데이터를 사용하고, 필터 플러그인은 사용자가 지정한 대로 데이터를 수정하며, 출력 플러그인은 목적지에 데이터를 기록

### input
![input](/assets/img/data_eigineering/elastic_search/logstash_input.svg)
- 외부에서 데이터를 받아오는 영역
- 사용하는 플러그인
  - file : UNIX명령어인 tail -0F 처럼 파일시스템에서 파일을 읽음
  - syslog : 514포트를 통해 syslog 메시지를 읽어오고 RFC3164 포맷에 따라 구분
  - redis : redis 채널과 redis 목록을 모두 사용하여 redis 서버로부터 읽음
  - beats : Beats에서 보낸 이벤트를 처리
  - jdbc : JDBC 데이터를 통해 이벤트를 생성

*Input plugin Ref. https://www.elastic.co/guide/en/logstash/current/input-plugins.html

### Filter
![filter](/assets/img/data_eigineering/elastic_search/logstash_filter.svg)
- 입력 받은 데이터를 가공, 조건에 대한 가공
- 사용하는 플러그인
  - grok : grok패턴을 사용해 메세지를 구조화된 형태로 분석
  - mutate : 필드 이름 변경, 제거, 수정 등을 변환
    - gsub : 가장 상단에서 grok에 보낼 메세지를 미리 전처리할 작업이 있을 떄 사용
  - date : 문자열을 지정한 패턴의 날짜형으로 변경경

*Filter plugin Ref. https://www.elastic.co/guide/en/logstash/current/filter-plugins.html

### Output
![output](/assets/img/data_eigineering/elastic_search/logstash_output.svg)
- 입력과 필터를 거친 데이터를 Target 대상으로 보내는 단계
- 사용하는 플러그인
  - elasticsearch : 시계열, 비시계열 데이터 세트 모두 전송 가능
  - file : 파일에 output데이터를 저장
  - Kafka : Kafka Topic에 데이터를 전송

*Output plugin Ref. https://www.elastic.co/guide/en/logstash/current/output-plugins.html

## Kibana
![kibana](/assets/img/data_eigineering/elastic_search/kinaba.png)
- Kibana는 Elasticsearch 데이터를 시각화하고 탐색할 수 있도록 설계된 분석 및 시각화 플랫폼
- 주요 기능
  - 대시보드 및 시각화: 차트, 그래프, 지도 등 다양한 형식으로 데이터를 시각화
  - Elastic Stack 통합: Elasticsearch와 긴밀하게 연동되며, 로그 및 메트릭 데이터를 분석
  - 보안 및 관리: 데이터 접근 권한을 제어하고, 사용자 관리 기능을 제공합니다.
  - 개발자 도구: API 및 개발자 도구를 활용하여 맞춤형 분석 및 확장
  - 알림 및 머신러닝: 데이터 이상 탐지 및 알림 기능을 지원하여 운영 문제를 신속하게 해결


&nbsp;

참고자료
- [https://medium.com/day34/elk-stack%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EB%A1%9C%EA%B7%B8-%EA%B4%80%EC%A0%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-%EB%A7%8C%EB%93%A4%EA%B8%B0-ca199502beab](https://medium.com/day34/elk-stack%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EB%A1%9C%EA%B7%B8-%EA%B4%80%EC%A0%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-%EB%A7%8C%EB%93%A4%EA%B8%B0-ca199502beab)
- [https://www.elastic.co/guide/kr/logstash/5.4/introduction.html](https://www.elastic.co/guide/kr/logstash/5.4/introduction.html)
- [https://www.elastic.co/guide/en/kibana/current/introduction.html](https://www.elastic.co/guide/en/kibana/current/introduction.html)
- [https://idkim97.github.io/2024-04-17-LogStash(%EB%A1%9C%EA%B7%B8%EC%8A%A4%ED%83%9C%EC%8B%9C)%EB%9E%80/](https://idkim97.github.io/2024-04-17-LogStash(%EB%A1%9C%EA%B7%B8%EC%8A%A4%ED%83%9C%EC%8B%9C)%EB%9E%80/)
- [https://blog.bizspring.co.kr/%ED%85%8C%ED%81%AC/logstash-%EC%8B%A4%EC%8B%9C%EA%B0%84-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%88%98%EC%A7%91-%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8/](https://blog.bizspring.co.kr/%ED%85%8C%ED%81%AC/logstash-%EC%8B%A4%EC%8B%9C%EA%B0%84-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%88%98%EC%A7%91-%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8/)