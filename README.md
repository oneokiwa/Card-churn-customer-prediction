# 📌 프로젝트 개요
본 프로젝트는 카드사 고객 데이터를 기반으로,
VIP 고객의 이탈(Churn) 여부를 예측하고
주요 이탈 요인을 분석하는 것을 목표로 합니다.

### 주요 작업 흐름:
* 데이터 병합 및 정제
* 결측치 처리 및 인코딩
* 데이터 분리 (7-9월 독립변수, 10월 이탈 여부)
* 로지스틱 회귀, 결정 트리 모델링, MLP 설계
* 20대 고객군에 대한 심층 분석
* SHAP, 특성 중요도 등 시각적 분석

---

# 📂 데이터 파이프라인
### 1. 데이터 병합 및 전처리
* 월별(7~9월) 데이터 병합 → 회원번호 기준 종합 데이터 생성
* VIP 고객 필터링 (VIP 등급코드 없는 고객 제외)
* 10월 이탈 여부(EXIT) 변수 추가

### 2. 결측치 처리 및 인코딩
* '직장시도명' 결측치는 "없음"으로 대체
* 원-핫 인코딩(거주시도명, 직장시도명)
* 레이블 인코딩(Life Stage, 연령, 이용금액대)
* 중복 변수 제거 및 영문 변수명 통일

### 3. 데이터셋 분리
* 7~9월 데이터(X), 10월 EXIT 값(y) 분리
* 훈련 60% / 검증 20% / 테스트 20%로 분할

---

# 🤖 모델링 및 예측
### 1. 로지스틱 회귀 (Logistic Regression)
* 데이터 표준화(StandardScaler) 후 학습
* 성능 평가
  
### 2. 결정 트리 (Decision Tree)
* 엔트로피 지수 기반(criterion='entropy'), 최대 깊이 제한(max_depth=5)
* 트리 구조 시각화
* 주요 변수 (Feature Importance) 추출

---

# 🔎 심층 분석 항목
![Image](https://github.com/user-attachments/assets/43f11e48-37be-4aac-baab-a18c1489c3b4)
### 1. 20대 고객군 집중 분석 (Age=1)
* 연령별 이탈률 분석
* 이용금액대(UsageAmountCategory)별 이탈률
* VIP 등급별 이탈률
* 지역(거주/근무지)별 이탈률
* 거주지/근무지가 일치 여부에 따른 이탈률
* 사용량 감소 패턴에 따른 이탈률
* 성별(Gender)에 따른 이탈률

### 2. 잔액(Balance) 중심 이탈 분석
* 월별 잔액(201809_Balance_B0M) 분포와 EXIT 비교
* 잔액 수준별(Balance Category) 이탈률 분석
* 사용금액대(UsageAmountCategory)와 잔액 간 관계 분석

---


# 📈 프로젝트 주요 성과

* 전체 고객군의 평균 이탈률 약 50%
* 20대 고객군(20s) 이탈률 약 57%로 타 연령 대비 높음
* 잔액이 낮거나 이용금액대가 낮은 고객군에서 이탈률 증가
![Image](https://github.com/user-attachments/assets/47717a40-7cca-4ca4-be81-ef132146c8bd)
* VIP 등급 6~7 그룹 고객 이탈 위험 증가
![Image](https://github.com/user-attachments/assets/a81496e4-7767-4ab0-976e-6ed95a4f1e1e)
