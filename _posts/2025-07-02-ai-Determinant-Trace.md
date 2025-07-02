---
layout: post
title: "[AI] Determinant and Trace"
date: 2025-07-02 12:43 +0900
description: Matrix Decomposition이해하기
category: [Data Analysis, AI]
tags: [AI, ML, Matix Decomposition, Determinant, Trace]
pin: false
math: true
mermaid: true
sitemap:
  changefreq: daily
  priority: 1.0
---

## Determinant: Motivation(행렬식: 동기부여)

**행렬의 역행렬과 행렬식 (Determinant)**

- 2x2 행렬의 정의:
  $$
  \mathbf{A} = \begin{pmatrix}
    a_{11} & a_{12} \\
    a_{21} & a_{22}
  \end{pmatrix}
  $$

- 역행렬:
  $$
  \mathbf{A}^{-1} = \frac{1}{a_{11}a_{22} - a_{12}a_{21}}
  \begin{pmatrix}
    a_{22} & -a_{12} \\
    -a_{21} & a_{11}
  \end{pmatrix}
  $$

- 가역 조건:
  $$
  \mathbf{A} \text{ is invertible iff } a_{11}a_{22} - a_{12}a_{21} \neq 0
  $$

- 행렬식 정의:
  $$
  \det(\mathbf{A}) = a_{11}a_{22} - a_{12}a_{21}
  $$

- 표기법:
  $$
  \det(\mathbf{A}) \text{ or } |\text{whole matrix}|
  $$

---

- 3x3 행렬의 행렬식(가우시안 소거법):
  -  연립일차방정식을 풀이하는 알고리즘이다. 풀이 과정에서, 일부 미지수가 차츰 소거되어 결국 남은 미지수에 대한 선형 결합으로 표현되면서 풀이가 완성된다.

  $$
  \begin{vmatrix}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23} \\
    a_{31} & a_{32} & a_{33}
  \end{vmatrix}
  = a_{11}a_{22}a_{33} + a_{21}a_{32}a_{13} + a_{31}a_{12}a_{23}
  - a_{31}a_{22}a_{13} - a_{11}a_{32}a_{23} - a_{21}a_{12}a_{33}
  $$

## Laplace Expansion을 이용한 3x3 행렬식 계산

3x3 행렬의 행렬식(determinant)은 아래와 같이 **Laplace 전개**(Laplace expansion)를 통해 계산할 수 있다.

$$
\begin{align*}
\det(\mathbf{A}) =\ 
& a_{11}a_{22}a_{33} + a_{21}a_{32}a_{13} + a_{31}a_{12}a_{23} \\
& - a_{31}a_{22}a_{13} - a_{11}a_{32}a_{23} - a_{21}a_{12}a_{33} \\
=\
& a_{11}(-1)^{1+1} \det(\mathbf{A}_{1,1}) \\
& + a_{12}(-1)^{1+2} \det(\mathbf{A}_{1,2}) \\
& + a_{13}(-1)^{1+3} \det(\mathbf{A}_{1,3})
\end{align*}
$$

여기서  
- $$a_{1j}$$는 첫 번째 행의 $$j$$번째 원소
- $$\mathbf{A}_{1,j}$$는 **첫 번째 행과 $$j$$번째 열을 제거한 부분행렬(submatrix)**
- $$\det(\mathbf{A}_{1,j})$$는 해당 부분행렬의 행렬식

> **Laplace expansion**: 행렬의 행 또는 열을 기준으로, 각 원소에 그 원소를 제외한 부분행렬의 행렬식(소행렬식, minor)을 곱하고, 부호를 번갈아가며 더하는 방식으로 전체 행렬의 행렬식을 계산하는 방법입니다. 이때 곱해지는 $$(-1)^{i+j}$$는 부호를 결정한다.  
> [Laplace expansion - Wikipedia][2]

> **Submatrix (부분행렬)**: 원래 행렬에서 특정 행과 열을 제거하여 얻는 더 작은 행렬이다.  
> [Matrix (mathematics) - Wikipedia][3]

> **Determinant (행렬식)**: 정사각행렬에서 정의되는 값으로, 선형 변환의 부피 변화율, 가역성(invertibility) 등 다양한 성질을 나타내는 스칼라 값.  
> [Determinant - Wikipedia][4]

---

### 예시 (Laplace 전개의 시각화)

아래 그림처럼, 첫 번째 행의 각 원소에 대해 해당 원소가 속한 행과 열을 제거한 2x2 부분행렬의 행렬식을 곱해 더하고, 부호를 번갈아 적용한다.

$$a_{11}$$: 첫 번째 행, 첫 번째 열 제거 → 

$$\left|\begin{matrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{matrix}\right|$$

$$a_{12}$$: 첫 번째 행, 두 번째 열 제거 → 

$$\left|\begin{matrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{matrix}\right|$$

$$a_{13}$$: 첫 번째 행, 세 번째 열 제거 →  

$$\left|\begin{matrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{matrix}\right|$$

따라서,

$$
\det(\mathbf{A}) = a_{11} \left| \begin{matrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{matrix} \right|
- a_{12} \left| \begin{matrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{matrix} \right|
+ a_{13} \left| \begin{matrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{matrix} \right|
$$

> **Laplace expansion**은 $$n \times n$$ 행렬의 행렬식을 $$n-1 \times n-1$$ 부분행렬의 행렬식으로 재귀적으로 표현하는 방법.  
> [Laplace expansion - Wikipedia][2]

## Determinant의 Laplace 전개와 가역성

임의의 정방행렬 $$\mathbf{A} \in \mathbb{R}^{n \times n}$$에 대해, 행 또는 열을 기준으로 Laplace 전개를 통해 행렬식을 계산.

- **열 j를 따라 전개:**
  $$
  \det(\mathbf{A}) = \sum_{k=1}^n (-1)^{k+j} a_{kj} \det(\mathbf{A}_{k,j})
  $$
- **행 j를 따라 전개:**
  $$
  \det(\mathbf{A}) = \sum_{k=1}^n (-1)^{k+j} a_{jk} \det(\mathbf{A}_{j,k})
  $$

> **Laplace 전개(Laplace expansion)**: 행렬의 임의의 행 또는 열에 대해 각 원소와 그 원소를 제외한 부분행렬(submatrix)의 행렬식을 곱하고, 교호부호 $$(-1)^{i+j}$$를 곱해 모두 더하는 방식이다. 이 방법을 여인자 전개(cofactor expansion)라고도 한다.  
> [Laplace expansion - Wikipedia][2], [라플라스 전개 - 위키백과][5]

> **부분행렬(submatrix)**: 원래 행렬에서 특정 행과 열을 제거하여 얻는 더 작은 행렬이다.  
> [Definition:Submatrix - ProofWiki][3]

모든 전개 방식(임의의 행 또는 열 기준)은 동일한 값을 가지므로, 정의상의 문제는 발생하지 않는다.

---

### 행렬식과 가역성

- **정리**  
  $$
  \det(\mathbf{A}) \neq 0 \iff \operatorname{rk}(\mathbf{A}) = n \iff \mathbf{A} \text{는 가역이다}
  $$

> **랭크(rank)**: 행렬에서 선형독립인 행 또는 열의 최대 개수로 정의된다. 랭크가 $$n$$이면 모든 행(또는 열)이 선형독립임을 의미한다.

## Determinant의 주요 성질

1. **곱셈에 대한 성질**
   $$
   \det(AB) = \det(A)\det(B)
   $$
   > 두 정사각행렬의 곱의 행렬식은 각 행렬의 행렬식의 곱과 같다.  
   > [Determinant - Wikipedia][6]

2. **전치행렬에 대한 성질**
   $$
   \det(A) = \det(A^T)
   $$
   > 행렬을 전치(transpose)해도 행렬식은 변하지 않는다.  
   > [Transpose - Wikipedia][6]

3. **역행렬에 대한 성질**
   $$
   \det(A^{-1}) = \frac{1}{\det(A)}
   $$
   > 가역(invertible) 행렬의 역행렬의 행렬식은 원래 행렬식의 역수이다.  
   > [Invertible matrix - Wikipedia][6]

4. **닮음 행렬에 대한 성질**
   $$
   \text{If } A' = S^{-1}AS, \quad \det(A) = \det(A')
   $$
   > 닮음(similar) 행렬은 행렬식이 같다. 닮음 변환은 기저 변환에 해당하며, 선형 변환의 고유한 특성을 보존한다.  
   > [Similarity (linear algebra) - Wikipedia][6]

5. **삼각행렬의 행렬식**
   $$
   \text{If } T \text{ is triangular,}\quad \det(T) = \prod_{i=1}^n T_{ii}
   $$
   > 삼각행렬(upper/lower triangular matrix)은 대각 원소의 곱이 행렬식이다.  
   > [Triangular matrix - Wikipedia][6]

6. **행/열에 다른 행/열의 배수를 더해도 행렬식은 변하지 않는다**
   > 한 행(또는 열)에 다른 행(또는 열)의 상수배를 더해도 행렬식은 변하지 않는다.  
   > [Elementary row operation - Wikipedia][6]

7. **행/열의 스칼라배**
   $$
   \det(\lambda A) = \lambda^n \det(A)
   $$
   > 행렬의 모든 행(또는 열)에 $$\lambda$$를 곱하면, 행렬식은 $$\lambda^n$$배가 된다 ($$n$$은 행렬의 크기).  
   > [Determinant - Wikipedia][6]

8. **행/열 맞바꿈**
   > 두 행(또는 두 열)을 맞바꾸면 행렬식의 부호가 바뀐다.  
   > [Determinant - Wikipedia][6]

---

### 추가 설명

- (5)~(8)번 성질을 활용하면 **가우스 소거법(Gaussian elimination)**을 통해 행렬을 삼각행렬로 만들고, 대각 원소의 곱으로 행렬식을 계산할 수 있다.

> **전치(transpose)**: 행렬의 행과 열을 맞바꾼 행렬을 의미한다.  
> [Transpose - Wikipedia][6]

> **동치(similarity)**: $$A$$와 $$A'$$가 $$A' = S^{-1}AS$$ 꼴로 표현될 때 두 행렬은 닮음 관계에 있다. 선형 변환의 본질적 구조(고유값 등)를 보존한다.  
> [Similarity (linear algebra) - Wikipedia][6]

> **삼각행렬(triangular matrix)**: 모든 원소가 주대각선 위 또는 아래에만 존재하는 행렬을 말한다.  
> [Triangular matrix - Wikipedia][6]

> **가우스 소거법(Gaussian elimination)**: 연립방정식 해법 및 행렬식 계산에 사용되는 행 연산 알고리즘이다.  
> [Gaussian elimination - Wikipedia][6]

## Trace(트레이스)의 정의와 성질

정사각행렬 $$\mathbf{A} \in \mathbb{R}^{n \times n}$$의 **트레이스(trace)**는 다음과 같이 정의한다.

$$
\operatorname{tr}(\mathbf{A}) := \sum_{i=1}^{n} a_{ii}
$$

> **트레이스(trace)**: 정사각행렬의 주대각선 원소(왼쪽 위에서 오른쪽 아래로 이어지는 원소)의 합을 의미한다. 트레이스는 선형대수학에서 행렬의 특성을 나타내는 중요한 값 중 하나다.  
> [Trace (linear algebra) - Wikipedia]

---

### 트레이스의 주요 성질

- 두 행렬의 합의 트레이스는 각 행렬의 트레이스의 합과 같다.
  $$
  \operatorname{tr}(\mathbf{A} + \mathbf{B}) = \operatorname{tr}(\mathbf{A}) + \operatorname{tr}(\mathbf{B})
  $$

- 스칼라 $$\alpha$$와 행렬의 곱의 트레이스는 스칼라와 트레이스의 곱과 같다.
  $$
  \operatorname{tr}(\alpha \mathbf{A}) = \alpha \operatorname{tr}(\mathbf{A})
  $$

- 단위행렬 $$\mathbf{I}_n$$의 트레이스는 $$n$$이다.
  $$
  \operatorname{tr}(\mathbf{I}_n) = n
  $$

> **단위행렬(identity matrix)**: 주대각선 원소가 모두 1이고, 나머지 원소가 모두 0인 정사각행렬을 의미한다.  
> [Identity matrix - Wikipedia]
