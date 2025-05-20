---
layout: post
title: "[Data Analysis] Funnel분석 & AARRR"
date: 2025-05-19 22:25 +0900
description: 그로스 해킹 이해해보기
category: [Data Analysis, Data]
tags: [Data Analysis, Funnel, AARRR]
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---

# 그로스 해킹

내가 데이터 분석을 다시한번 고민하게 도움을 준 라이브 초청 강의에서 멘토분이 추천해 준 책이 있다. 그로스 해킹이라는 책으로 비즈니스 마케팅에 대해 분석의 방향성을 제시한 전에는 생각지도 못한 책이었다.

그로스 해킹(Growth hacking)은 창의성, 분석적인 사고, 소셜 망을 이용하여 제품을 팔고, 노출시키는 마케팅 방법으로 스타트업회사들에 의해 개발되었다. 성장을 뜻하는 growth와 해킹(hacking)이 결합된 단어로 고객의 반응에 따라 제품 및 서비스를 수정해 제품과 시장의 궁합(Product-Market Fit)을 높이는 것을 의미한다. 여기서 말하는 고객의 반응은 정량 데이터와 정성 데이터로 산출된다.

아래는 그로스 해킹이라는 것이 무엇이고 창시자는 어떤 목적으로 만들었는지 이해하기 위해 본 영상이다.


[![그로스 해킹](http://img.youtube.com/vi/TMYaTkOyV4Y/0.jpg)](https://youtu.be/TMYaTkOyV4Y)

[https://www.youtube.com/watch?v=TMYaTkOyV4Y](https://www.youtube.com/watch?v=TMYaTkOyV4Y)

위 동영상을 통해 ``Activation(활성화)``에 집중할 생각이다.


## Funnel 분석

Funnel(깔데기)분석이란 고객들이 우리가 설계한 유저 경험 루트를 따라 잘 도착하고 있는지 확인해보기 위해 최초 유입부터 최종 목적지까지 단계를 나누어서 살펴보는 분석 기법이다. 

``tip``: 퍼널은 앞에서 뒤로 개선하는 것보다 뒤에서 앞으로 개선하는 것이 좋음. 즉, 결제할 사람을 확실히 결제하게 만들고, 나중에 유입을 늘리기

![Funnel](/assets/img/data_analysis/aarrr/funnel-analysis-sub-funnel.png)

[https://chartio.com/learn/product-analytics/what-is-a-funnel-analysis/](https://chartio.com/learn/product-analytics/what-is-a-funnel-analysis/)

&nbsp;

아래로 내려가며 ``CTR(클릭율, 클릭율/페이지 노출 수)`` 또는 ``CVR(전환율, 전환수/특정 행동을 한 수)``지표를 새운다. 목표는 아래에서 부터 설정한 지표를 높이기 위한 정의를 세운다.

## AARRR

AARRR분석 방식은 Funnel분석 중 디지털 마케팅과 온라인 마케팅에서 자주 사용되는 분석 방식이다. 마케팅에서 사용자의 여정을 단계별로 구분한 것이다. 미국의 스타트업 엑셀러레이터인 500 Startsup(now 500 Global)의 설립자 데이브 맥클루어(Dave McClure)에 의해 개발된 분석 프레임 워크로 진정한 성장에 집중해야 할 스타트업 기업들이 소셜 미디어의 ‘좋아요’ 와 같은 피상적인 지표에 현혹되어 성장의 본질에 집중하지 못하는 것을 보고 어쩌면 가이드라인과 같은 스타트업 성장의 모니터링 지표를 고안하였다

![AARRR](/assets/img/data_analysis/aarrr/aarrr.png)

[https://blog.martinee.io/post/growthmarketing-aarrr-funnel-analytics](https://blog.martinee.io/post/growthmarketing-aarrr-funnel-analytics)

- Acquisition(획득) : 얼마나 제품에 접근하는가?
- Activation(활성화) : 고객이 최초의 좋은 경험을 하는가?
- Retention(유지) : 다시 제품을 사용하는가?
- Revenue(매출) : 얼마나 돈을 버는가?
- Referral(추천) : 다른 사람에게 공유하는가?


대표적인 지표

- ``Acquisition(획득)``
  - 고객 획득 비용(CAC, Customer Acquistion Cost) : 한 명의 고객을 획득하기 위해 지출되는 비용
  - 일간/월간 활성 유저(DAU, Daliy Active User / MAU, Monthly Active User) : 특정 기간동안 서비스를 이용하는 순수 이용자
- ``Activation(활성화)``
  - 페이지 뷰(PV, Page View) : 유저가 둘러본 페이지 수
  - 체류 시간(DT, Duration Time) : 고객이 서비스에 방문해서 나가기까지 머무른 시간
  - 아하 모먼트(Aha Moment) : 제품의 핵심 가치를 고객이 경험하는 결정적인 순간
- ``Retention(유지)``
  - 리텐션율(Retention Rate) : 고객이 유지되는 비율
  - 이탈률(Churn Rate) : 고객이 이탈하는 비율
- ``Revenue(매출)``
  - 고객 생애 가치(LTV, Life Time Value) : 고객 한 명이 자신의 일생동안 기업에게 가져다주는 이익의 종합
  - 결제 유저(PU, Paying User) : 실제로 결제하는 고객
  - 유저 당 평균 수익(ARPU, Average Revenue Per User) : 전체 수익을 전체 회원 수로 나눈 값
- ``Referral(추천)``
  - 사용자 언급 댓글 수
  - SNS 공유된 횟수
  - 레퍼럴 트래픽(Referral Traffic) : 다양한 채널에서 사이트로 유입된 트래픽

``Tip`` : 상황에 따라 RARRA로 변경 가능

- **RARRA**
  - Retention : 다시 제품을 사용하는가?
  - Activation : 고객이 최초의 좋은 경험을 하는가?
  - Referral : 다른 사람에게 공유하는가?
  - Revenue : 얼마나 돈을 버는가?
  - Acquisition : 얼마나 제품에 접근하는가?

&nbsp;
- 참고자료
  - [https://www.inflearn.com/course/pm-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%A6%AC%ED%84%B0%EB%9F%AC%EC%8B%9C](https://www.inflearn.com/course/pm-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%A6%AC%ED%84%B0%EB%9F%AC%EC%8B%9C)
  - [https://datarian.io/blog/funnel-analysis](https://datarian.io/blog/funnel-analysis)
  - [https://chartio.com/learn/product-analytics/what-is-a-funnel-analysis/](https://chartio.com/learn/product-analytics/what-is-a-funnel-analysis/)
  - [https://yozm.wishket.com/magazine/detail/1071/](https://yozm.wishket.com/magazine/detail/1071/)
  - [https://blog.martinee.io/post/growthmarketing-aarrr-funnel-analytics](https://blog.martinee.io/post/growthmarketing-aarrr-funnel-analytics)
  - [https://medium.com/wisetracker/%EC%8A%A4%ED%83%80%ED%8A%B8%EC%97%85%EC%9D%84-%EC%9C%84%ED%95%9C-aarrr-%ED%95%B4%EC%A0%81%EC%A7%80%ED%91%9C-%EA%B0%9C%EB%85%90%EC%9E%A1%EA%B8%B0-d8b79a6024c](https://medium.com/wisetracker/%EC%8A%A4%ED%83%80%ED%8A%B8%EC%97%85%EC%9D%84-%EC%9C%84%ED%95%9C-aarrr-%ED%95%B4%EC%A0%81%EC%A7%80%ED%91%9C-%EA%B0%9C%EB%85%90%EC%9E%A1%EA%B8%B0-d8b79a6024c)
