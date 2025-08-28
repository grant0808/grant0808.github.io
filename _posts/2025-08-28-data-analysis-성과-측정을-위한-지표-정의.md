---
layout: post
title: "[Data Analysis] 성과 측정을 위한 지표용어 정의"
date: 2025-08-28 12:33 +0900
description: 데이터 분석에서 대시보드에 정의하는 지표용어 정리하기
category: [Data Analysis, Metric]
tags: [Data Analysis, Metric]
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---

# 목적
데이터 분석에서 대시보드를 제작사는데 있어 지표정의는 중요하다. 취업을 위해서 대표젹인 지표를 숙지하고 기억이 안나면 다시한번 보기 위해서 이 글을 작성하고자 한다.

지표가 중요한 이유는 비즈니스 성과를 측정하고, 전략적인 방향을 설정하는 데 있어 이정표 역할을 한다. 

# 지표
## KPI(Key Performance Indecator)
- 핵심성과 지표로 비즈니스 목표를 달성하고 있는지를 측정하는 정량적인 지표

## NSM(North Star Metric)
- 북극성 지표로 고객에게 제공하는 서비스를 통해 얻게되는 행심가치를 정확하게 대변하는 지표

## AU(Active User)
- 서비스에서 (특정)활동하는 사용자
- DAU(Daily), WAU(Weekly), MAU(Monthly)등 사용
- 방문 DAU, 거래 DAU 등
- ex) <span style="color: #0000FF">Distinct</span> daily active user Id = 하루 활동한 사용자 수(중복제거)

## RU(Resistered User)
- 현재까지 가입한 총 사용자의 수

## NRU(New Resistered User)
- 신규 사용자의 수

## Retention(리텐션)
- 서비스를 사용한 사람이 다시 사용하는 비율
- Corhort(코호트)
  - 통계적으로 동일한 특색이나 행동 양식을 공유하는 집단
  - 가입일자 기준으로 많이 파악

## Stickness(고착도)
- 월간/주간 순 사용자 중 특정 일자에 접속한 사용자의 비율
- ex) DAU/MAU or DAU/WAU

## PU(Paying User)
- 결제를 한번이라도 한 사용자

## PV(Page View)
- 특정 페이지를 본 수

## UV(Unique View, Unique Visitor)
- 특정 페이지를 본 순 사용자, 일정 기간동안 페이지를 본 순 사용자

## DT(Duration Time)
- 체류시간으로 특정 페이지에서 머무른 시간
- ex) (Mean, Median) 체류 시간

## Session(세션)
- 정의된 기간 동안 유저가 앱/웹에서 활동하는 묶음
- 하루에 여러번 사용하는 서비스는 세션 기준으로 분석하기도 함
- Google Analytics는 기본적으로 30분 동안 아무 활동이 없으면 세션을 종료하는 것으로 간주

## CTR(Click Throught Rate)
- 클릭율로 페이지에서 특정 Component(버튼 등)을 클릭한 비율
- ex) 클릭수/페이지 노출 수

## VTR(View Through Rate)
- 시청율로 광고가 노출되었을 때 시청하는 비율
- ex) 광고시청 수/광고 노출 수

## CVR(Conversion Rate)
- 전환율로 특정 행동을 한 후, 전환된 비율
- 지표를 이해하기 위해 앞에 구체적인 단어 사용 필요(XX전환율)
- ex) 전환 수/특정 행동을 한 수

## Reach(도달률)
- 특정 마케팅 캠페인이 목표로 하는 사용자에게 얼마나 도달했는지를 나타내는 지표

## Requests
- 광고 요청 횟수

## Coverage
- 1. 광고를 확인할 수 있는 범위
- 2. 코드 커버리지 = 테스트 시 실행된 코드 라인 수 / 전체 라인 수

## bounce rate
- 이탈율로 사이트에 방문한 사람 중 한 페이지만 보고 이탈하는 수의 비율
- ex) 한 페이지만 본 사용자 수 / 전체 사용자 수

## ROAS(Return On Ads Spending)
- 광고비 대비 매출 비율
- ex) 광고로 인한 매출 / 광고비

## ARPU(Average Revenue Per User)
- 사용자 1명당 평균 수익
- ex)

## LTV(Life Time Value)
- 고객생애가치로 사용자 특정 비즈니스와 관계를 유지하는 동안 해당 기업에 창출할 것으로 예상되는 총 이익 또는 총수익
- game: 사용자 1인당 게임에서 이탈하기까지 지불하는 비용
- 기본 공식: (1인당 평균 구매액) × (구매 빈도) × (고객 유지 기간) 
- 좀 더 구체적인 계산: (매출액 – 매출 원가) ÷ (구매자 수) 
- 마케팅 비용 고려: LTV = (평균 구매 단가 × 구매 빈도 × 계속 구매 기간) - (고객 획득 비용 + 고객 유지 비용)

## CRR(Customer Retention Rate)
- 고객 유지율

## CCR(Customer Churn Rate)
- 고객 이탈률

## Funnel(퍼널)
- Funnel(깔데기)분석이란 고객들이 우리가 설계한 유저 경험 루트를 따라 잘 도착하고 있는지 확인해보기 위해 최초 유입부터 최종 목적지까지 단계를 나누어서 살펴보는 분석 기법

## AARRR
- Acquisition(획득) : 얼마나 제품에 접근하는가?
- Activation(활성화) : 고객이 최초의 좋은 경험을 하는가?
- Retention(유지) : 다시 제품을 사용하는가?
- Revenue(매출) : 얼마나 돈을 버는가?
- Referral(추천) : 다른 사람에게 공유하는가?
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

# 정리

## 핵심 지표 (매우 높음)

| 지표명 | 정의 | 수식 |
|--------|------|------|
| **KPI (Key Performance Indicator)** | 핵심성과 지표로 비즈니스 목표 달성을 측정하는 정량적 지표 | - |
| **NSM (North Star Metric)** | 고객에게 제공하는 서비스를 통해 얻게되는 핵심가치를 정확하게 대변하는 지표 | - |
| **AU (Active User)** | 서비스에서 활동하는 사용자 (DAU, WAU, MAU 등) | `Distinct daily active user Id` = 하루 활동한 사용자 수(중복제거) |
| **Retention (리텐션)** | 서비스를 사용한 사람이 다시 사용하는 비율 | `재방문 사용자 수 / 전체 사용자 수` |
| **LTV (Life Time Value)** | 고객생애가치로 사용자가 특정 비즈니스와 관계를 유지하는 동안 창출할 것으로 예상되는 총 이익 | `(1인당 평균 구매액) × (구매 빈도) × (고객 유지 기간)` |
| **ARPU (Average Revenue Per User)** | 사용자 1명당 평균 수익 | `총 수익 / 전체 사용자 수` |

## 중요 지표 (높음)

| 지표명 | 정의 | 수식 |
|--------|------|------|
| **RU (Registered User)** | 현재까지 가입한 총 사용자의 수 | - |
| **NRU (New Registered User)** | 신규 사용자의 수 | - |
| **Stickiness (고착도)** | 월간/주간 순 사용자 중 특정 일자에 접속한 사용자의 비율 | `DAU/MAU` 또는 `DAU/WAU` |
| **PU (Paying User)** | 결제를 한번이라도 한 사용자 | - |
| **CTR (Click Through Rate)** | 클릭율로 페이지에서 특정 Component를 클릭한 비율 | `클릭수 / 페이지 노출 수` |
| **CVR (Conversion Rate)** | 전환율로 특정 행동을 한 후, 전환된 비율 | `전환 수 / 특정 행동을 한 수` |
| **ROAS (Return On Ads Spending)** | 광고비 대비 매출 비율 | `광고로 인한 매출 / 광고비` |
| **CRR (Customer Retention Rate)** | 고객 유지율 | - |
| **CCR (Customer Churn Rate)** | 고객 이탈률 | - |
| **CAC (Customer Acquisition Cost)** | 한 명의 고객을 획득하기 위해 지출되는 비용 | `총 획득 비용 / 획득한 고객 수` |

## 표준 지표 (보통)

| 지표명 | 정의 | 수식 |
|--------|------|------|
| **PV (Page View)** | 특정 페이지를 본 수 | - |
| **UV (Unique View/Visitor)** | 특정 페이지를 본 순 사용자, 일정 기간동안 페이지를 본 순 사용자 | - |
| **DT (Duration Time)** | 체류시간으로 특정 페이지에서 머무른 시간 | `(Mean, Median) 체류 시간` |
| **Session (세션)** | 정의된 기간 동안 유저가 앱/웹에서 활동하는 묶음 (Google Analytics 기본: 30분 비활성시 세션 종료) | - |
| **VTR (View Through Rate)** | 시청율로 광고가 노출되었을 때 시청하는 비율 | `광고시청 수 / 광고 노출 수` |
| **Reach (도달률)** | 특정 마케팅 캠페인이 목표로 하는 사용자에게 얼마나 도달했는지를 나타내는 지표 | - |
| **Requests** | 광고 요청 횟수 | - |
| **Coverage** | 1. 광고를 확인할 수 있는 범위<br>2. 코드 커버리지 | `테스트 시 실행된 코드 라인 수 / 전체 라인 수` |
| **Bounce Rate (이탈율)** | 사이트에 방문한 사람 중 한 페이지만 보고 이탈하는 수의 비율 | `한 페이지만 본 사용자 수 / 전체 사용자 수` |

## 분석 프레임워크

### AARRR 프레임워크

| 단계 | 의미 | 주요 지표 | 설명 |
|------|------|----------|------|
| **Acquisition (획득)** | 얼마나 제품에 접근하는가? | CAC, DAU/MAU | 고객 획득 비용, 일간/월간 활성 유저 |
| **Activation (활성화)** | 고객이 최초의 좋은 경험을 하는가? | PV, DT, Aha Moment | 페이지 뷰, 체류 시간, 아하 모먼트 |
| **Retention (유지)** | 다시 제품을 사용하는가? | Retention Rate, Churn Rate | 리텐션율, 이탈률 |
| **Revenue (매출)** | 얼마나 돈을 버는가? | LTV, PU, ARPU | 고객 생애 가치, 결제 유저, 유저 당 평균 수익 |
| **Referral (추천)** | 다른 사람에게 공유하는가? | 사용자 언급 댓글 수, SNS 공유 횟수, Referral Traffic | 레퍼럴 트래픽 |

### RARRA 프레임워크 (상황에 따라 변경 가능)

| 순서 | 단계 | 의미 |
|------|------|------|
| 1 | **Retention (유지)** | 다시 제품을 사용하는가? |
| 2 | **Activation (활성화)** | 고객이 최초의 좋은 경험을 하는가? |
| 3 | **Referral (추천)** | 다른 사람에게 공유하는가? |
| 4 | **Revenue (매출)** | 얼마나 돈을 버는가? |
| 5 | **Acquisition (획득)** | 얼마나 제품에 접근하는가? |

## 추가 개념 및 용어

| 개념 | 정의 | 설명 |
|------|------|------|
| **Funnel (퍼널)** | 깔대기 분석 | 고객들이 우리가 설계한 유저 경험 루트를 따라 잘 도착하고 있는지 확인해보기 위해 최초 유입부터 최종 목적지까지 단계를 나누어서 살펴보는 분석 기법 |
| **Cohort (코호트)** | 동질 집단 | 통계적으로 동일한 특색이나 행동 양식을 공유하는 집단 (가입일자 기준으로 많이 파악) |
| **Aha Moment (아하 모먼트)** | 핵심 가치 경험 순간 | 제품의 핵심 가치를 고객이 경험하는 결정적인 순간 |

## LTV 계산 공식 상세

| 계산 방법 | 공식 |
|-----------|------|
| **기본 공식** | `(1인당 평균 구매액) × (구매 빈도) × (고객 유지 기간)` |
| **구체적인 계산** | `(매출액 – 매출 원가) ÷ (구매자 수)` |
| **마케팅 비용 고려** | `LTV = (평균 구매 단가 × 구매 빈도 × 계속 구매 기간) - (고객 획득 비용 + 고객 유지 비용)` |
| **게임 업계** | 사용자 1인당 게임에서 이탈하기까지 지불하는 비용 |

&nbsp;
- 참고자료
  - [https://zzsza.github.io/data-for-pm/metric/examples.html#session-%E1%84%89%E1%85%A6%E1%84%89%E1%85%A7%E1%86%AB](https://zzsza.github.io/data-for-pm/metric/examples.html#session-%E1%84%89%E1%85%A6%E1%84%89%E1%85%A7%E1%86%AB)
  - [https://ourkofe.tistory.com/82](https://ourkofe.tistory.com/82)
  - [https://blog.blux.ai/crm-%EB%A7%88%EC%BC%80%ED%84%B0%EB%9D%BC%EB%A9%B4-%EA%BC%AD-%EC%95%8C%EC%95%84%EC%95%BC%ED%95%A0-crm-%EB%A7%88%EC%BC%80%ED%8C%85%EC%9D%98-%EA%B8%B0%EB%B3%B8-%EA%B0%9C%EB%85%90-%EC%B4%9D%EC%A0%95%EB%A6%AC-%EC%84%B1%EA%B3%BC%EC%A7%80%ED%91%9C%ED%8E%B8-20896](https://blog.blux.ai/crm-%EB%A7%88%EC%BC%80%ED%84%B0%EB%9D%BC%EB%A9%B4-%EA%BC%AD-%EC%95%8C%EC%95%84%EC%95%BC%ED%95%A0-crm-%EB%A7%88%EC%BC%80%ED%8C%85%EC%9D%98-%EA%B8%B0%EB%B3%B8-%EA%B0%9C%EB%85%90-%EC%B4%9D%EC%A0%95%EB%A6%AC-%EC%84%B1%EA%B3%BC%EC%A7%80%ED%91%9C%ED%8E%B8-20896)