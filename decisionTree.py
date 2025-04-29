# ===================================================================================
# 📌 결정 트리 모델 학습 및 잔액 이탈 분석 파이프라인 요약
# 1️⃣ 데이터 준비 및 전처리
# 2️⃣ 결정 트리 모델 학습 및 평가
# 3️⃣ 트리 구조 시각화
# 4️⃣ 주요 특성(Feature Importance) 분석
# 5️⃣ 201809_Balance_B0M 특성 심층 분석
# 6️⃣ 사용금액대(UsageAmountCategory)와 잔액(Balance_B0M) 간 관계 분석
# ===================================================================================

import pandas as pd
import numpy as np
import shap
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
X_train = pd.read_csv('train/X_train.csv')
X_val = pd.read_csv('train/X_val.csv')
X_test = pd.read_csv('train/X_test.csv')
y_train = pd.read_csv('train/y_train.csv').squeeze()
y_val = pd.read_csv('train/y_val.csv').squeeze()
y_test = pd.read_csv('train/y_test.csv').squeeze()

# 문자열 및 범주형 데이터 처리
def preprocess_features(df):
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:  # 문자열 데이터 열 확인
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])  # 문자열 데이터를 숫자로 변환
        label_encoders[col] = le  # 각 열의 인코더 저장
    return df, label_encoders

X_train, label_encoders = preprocess_features(X_train)
X_val, _ = preprocess_features(X_val)  # 검증 데이터도 동일하게 처리
X_test, _ = preprocess_features(X_test)  # 테스트 데이터도 동일하게 처리

# DecisionTreeClassifier 모델 생성
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
model.fit(X_train, y_train)  # 모델 학습

# 검증 데이터로 성능 확인
y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nConfusion Matrix (Validation):\n", confusion_matrix(y_val, y_val_pred))
print("\nClassification Report (Validation):\n", classification_report(y_val, y_val_pred))

# 테스트 데이터로 최종 평가
y_test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nConfusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))

# 6. 트리 시각화
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X_train.columns, class_names=['Retained', 'Churned'], filled=True, fontsize=10)
plt.title('Decision Tree Visualization')
plt.show()

############### 심층 분석 및 시각화 ###############
# 1. Feature Importance 분석
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 상위 5개 변수 시각화
plt.figure(figsize=(20, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(), palette='viridis', hue='Importance')
plt.title('Top 5 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# 1. 분포 분석: 잔액 분포와 이탈 여부에 따른 차이
plt.figure(figsize=(20, 6))
sns.histplot(X_train['201809_Balance_B0M'], kde=True, color='blue', label='Overall')
sns.histplot(X_train[y_train == 1]['201809_Balance_B0M'], kde=True, color='red', label='Churned')
sns.histplot(X_train[y_train == 0]['201809_Balance_B0M'], kde=True, color='green', label='Retained')
plt.title('Distribution of 201809_Balance_B0M by EXIT')
plt.xlabel('201809_Balance_B0M')
plt.ylabel('Density')
plt.legend()
plt.show()

# 2. 상관관계 분석: 주요 독립변수와 EXIT 간 상관성
correlation = X_train['201809_Balance_B0M'].corr(y_train)
print(f"Correlation between 201809_Balance_B0M and EXIT: {correlation:.4f}")

# 3. 잔액 구간별 이탈률 분석 - 잔액이 적을수록 이탈률이 높다.
X_train['EXIT'] = y_train

# 잔액 구간 정의 (low: 낮음, medium: 중간, high: 높음)
def categorize_balance(balance):
    if balance < 500000:  # 잔액이 50만 원 미만
        return 'Low'
    elif balance < 2000000:  # 잔액이 50만 원 이상 200만 원 미만
        return 'Medium'
    else:  # 잔액이 200만 원 이상
        return 'High'

# Balance_B0M 구간 추가
X_train['Balance_Category'] = X_train['201807_Balance_B0M'].apply(categorize_balance)

# 잔액 구간별 이탈률 계산
balance_exit_rate = X_train.groupby('Balance_Category')['EXIT'].mean().reset_index()

# 시각화
plt.figure(figsize=(8, 6))
sns.barplot(
    data=balance_exit_rate,
    x='Balance_Category',
    y='EXIT',
    palette='coolwarm'
)
plt.title('Churn Rate by Balance Category', fontsize=16)
plt.xlabel('Balance Category', fontsize=12)
plt.ylabel('Churn Rate', fontsize=12)
plt.ylim(0, 1)
plt.show()

print("Churn Rate by Balance Category:")
print(balance_exit_rate)


# 4. UsageAmountCategory와 Balance_B0M 변수 간 상관관계 계산
correlation = X_train[['201809_UsageAmountCategory', '201809_Balance_B0M']].corr().iloc[0, 1]
print(f"Correlation between UsageAmountCategory and Balance_B0M: {correlation:.4f}")

# i. 산점도를 이용한 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='201809_UsageAmountCategory',
    y='201809_Balance_B0M',
    data=X_train,
    alpha=0.7
)
plt.title('Scatter Plot: UsageAmountCategory vs. Balance_B0M')
plt.xlabel('UsageAmountCategory')
plt.ylabel('Balance_B0M')
plt.show()

# ii. 히트맵을 이용한 상관관계 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(
    X_train[['201809_UsageAmountCategory', '201809_Balance_B0M']].corr(),
    annot=True,
    cmap='coolwarm',
    cbar=True,
    fmt='.2f'
)
plt.title('Correlation Heatmap: UsageAmountCategory vs. Balance_B0M')
plt.show()