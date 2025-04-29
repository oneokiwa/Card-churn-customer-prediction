# ===================================================================================
# 📌 로지스틱 회귀 모델 학습 및 20대 고객군 이탈 분석 파이프라인 요약
# 1️⃣ 데이터 준비 및 전처리
# 2️⃣ 로지스틱 회귀 모델 학습 및 평가
# 3️⃣ 주요 특성 분석 및 시각화
# 4️⃣ 20대(Age=1) 고객 중점 분석
# 5️⃣ - 지역(ResidenceCity, WorkCity)별 이탈률 분석
# 6️⃣ - 사용 패턴별 이탈 분석
# 7️⃣ - 성별(Gender)별 이탈률 비교
# ===================================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures

# 데이터 불러오기
X_train = pd.read_csv('train/X_train.csv')
X_val = pd.read_csv('train/X_val.csv')
X_test = pd.read_csv('train/X_test.csv')
y_train = pd.read_csv('train/y_train.csv').squeeze()
y_val = pd.read_csv('train/y_val.csv').squeeze()
y_test = pd.read_csv('train/y_test.csv').squeeze()

# 문자열 데이터 확인 및 변환 (발급회원번호)
def preprocess_features(df):
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

X_train = preprocess_features(X_train)
X_val = preprocess_features(X_val)
X_test = preprocess_features(X_test)

# 데이터 정규화 (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 로지스틱 회귀 모델 학습
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# 검증 데이터 평가
y_val_pred = model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)

print("Validation Accuracy:", val_accuracy)
print("\nConfusion Matrix (Validation):\n", confusion_matrix(y_val, y_val_pred))
print("\nClassification Report (Validation):\n", classification_report(y_val, y_val_pred))

# 테스트 데이터 평가
y_test_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Test Accuracy:", test_accuracy)
print("\nConfusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))

print("Logistic Regression model trained and evaluated successfully.")

########### 상세 분석 및 시각화 ##############
# 1. 특성 중요도 분석
feature_importance = np.abs(model.coef_[0])
features = X_train.columns
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(30, 20))
plt.bar(range(len(features)), feature_importance[sorted_idx], align="center")
plt.xticks(range(len(features)), features[sorted_idx], rotation=90)
plt.title("Feature Importance (Logistic Regression)")
plt.show()

# 2. ROC-AUC 분석
y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_test_prob)
fpr, tpr, _ = roc_curve(y_test, y_test_prob)

# 3. ROC Curve 시각화
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 5. 클래스 분포 확인 - 균형
# print(y_train.value_counts(normalize=True))
# 결과
# EXIT
# 1    0.500102
# 0    0.499898
# Name: proportion, dtype: float64


# 6. 연령별 이탈률 분석
# X_train에 y_train 추가하여 연령별 분석하기
X_train['EXIT'] = y_train

age_exit_rate = X_train.groupby('Age')['EXIT'].agg(['mean', 'count']).reset_index()
age_exit_rate.columns = ['Age', 'ExitRate', 'CustomerCount']

# 그래프 시각화
plt.figure(figsize=(12, 6))
sns.barplot(x='Age', y='ExitRate', data=age_exit_rate, palette='Blues_d', hue='Age', dodge=False, legend=False)

plt.title('Exit Rate by Age Group', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Exit Rate', fontsize=12)
plt.ylim(0, 1)  # 이탈률 범위 (0~1)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend([], [], frameon=False)  # 범례 제거
plt.show()

# 전체 고객 평균 이탈률 계산 - 50%
overall_exit_rate = X_train['EXIT'].mean()
print(f"Overall Exit Rate: {overall_exit_rate:.2f}")

# 20대(연령=1) 고객 평균 이탈률 계산 - 57%
age_20s_exit_rate = X_train[X_train['Age'] == 1]['EXIT'].mean()
print(f"Exit Rate for Age Group 20s (Age=1): {age_20s_exit_rate:.2f}")


############ 20대 고객 중점 분석 ###############
age_20s_data = X_train[X_train['Age'] == 1]  # Age가 1인 데이터를 필터링하여 20대 고객 데이터 생성
age_20s_data['EXIT'] = y_train  # 20대 고객의 이탈 여부를 데이터에 추가

# i. 20대 고객의 이탈 여부에 따른 UsageAmountCategory 분포 비교
usage_columns = ['201809_UsageAmountCategory', '201808_UsageAmountCategory', '201807_UsageAmountCategory']  # 분석 대상 열 지정

for col in usage_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=age_20s_data,  # 데이터
        x=col,  # X축 UsageAmountCategory 지정
        hue='EXIT',  # 이탈 여부에 따라 색상 구분
        palette='Set2',  # 시각화를 위한 색상 팔레트 설정
        order=sorted(age_20s_data[col].unique())  # UsageAmountCategory 값의 순서를 지정하여 정렬
    )
    plt.title(f"Distribution of {col} by EXIT for Age 20s")  # 제목
    plt.xlabel("Usage Amount Category")  # X축 레이블
    plt.ylabel("Count")  # Y축 레이블
    plt.xticks(
        ticks=range(len(age_20s_data[col].unique())),  # 카테고리 수에 맞게 틱 설정
        labels=sorted(age_20s_data[col].unique())  # 카테고리 이름으로 틱 레이블 설정
    )
    plt.legend(title="EXIT", labels=["Retained (0)", "Churned (1)"])  # 범례 설정
    plt.show()

# ii. 이용금액대별 이탈률 계산 및 시각화
for col in usage_columns:
    churn_rate = age_20s_data.groupby(col)['EXIT'].mean().reset_index()  # UsageAmountCategory 별 평균 이탈률 계산

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=churn_rate,  # 데이터 소스
        x=col,  # X축 UsageAmountCategory 열 지정
        y='EXIT',  # Y축에 이탈률 지정
        palette='coolwarm'  # 색상 팔레트 설정
    )
    plt.title(f"Churn Rate by {col} for Age 20s")  # 제목
    plt.xlabel("Usage Amount Category")  # X축 레이블
    plt.ylabel("Churn Rate")  # Y축 레이블
    plt.ylim(0, 1)  # Y축 범위 설정
    plt.show()


# iii. 20대 고객 주요 변수와 이탈 여부 간 상관관계
correlations = age_20s_data.corr()['EXIT'].sort_values(ascending=False)
print("Correlations with EXIT for Age Group 20s:")
print(correlations)
# 201809_UsageAmountCategory    0.870465
# 201808_UsageAmountCategory    0.856354
# 201807_UsageAmountCategory    0.832290
# WorkCity_None                 0.100457

# iv. VIP 등급에 따른 20대 고객의 이탈률 시각화 - 6, 7의 이탈률 높음
plt.figure(figsize=(10, 6))
vip_exit_rate = age_20s_data.groupby('VIPGradeCode')['EXIT'].mean().reset_index()
#VIP 등급별로 데이터를 나누고, 각 그룹 내에서 EXIT 열의 평균값을 계산 (해당 그룹의 이탈률)
sns.barplot(x='VIPGradeCode', y='EXIT', data=vip_exit_rate, palette='Blues_d', hue='VIPGradeCode')
plt.title('Exit Rate by VIP Grade for Age 20s')
plt.xlabel('VIP Grade')
plt.ylabel('Exit Rate')
plt.ylim(0, 1)
plt.show()

# ---------------------------------------------------------------------------------------------

# i. 지역 변수와 이탈률 관계 분석

# 1. ResidenceCity와 EXIT의 관계 분석
residence_columns = [col for col in age_20s_data.columns if col.startswith('ResidenceCity')]
residence_exit_rate = pd.DataFrame()

# ResidenceCity 별 이탈률 계산
for col in residence_columns:
    city_name = col.replace('ResidenceCity_', '')  # 변수명에서 지역 이름 추출
    city_data = age_20s_data[age_20s_data[col] == 1]  # 해당 지역의 데이터 필터링
    exit_rate = city_data['EXIT'].mean() if not city_data.empty else 0  # 이탈률 계산
    residence_exit_rate = pd.concat(
        [residence_exit_rate, pd.DataFrame({'City': [city_name], 'ExitRate': [exit_rate]})]
    )

# ResidenceCity 이탈률 시각화
plt.figure(figsize=(12, 8))
sns.barplot(data=residence_exit_rate, x='City', y='ExitRate', palette='coolwarm', hue='City')
plt.title('Exit Rate by Residence City for Age 20s')
plt.xlabel('Residence City')
plt.ylabel('Exit Rate')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()

# 2. WorkCity와 EXIT의 관계 분석
work_columns = [col for col in age_20s_data.columns if col.startswith('WorkCity')]
work_exit_rate = pd.DataFrame()

# WorkCity별 이탈률 계산
for col in work_columns:
    city_name = col.replace('WorkCity_', '')  # 변수명에서 지역 이름 추출
    city_data = age_20s_data[age_20s_data[col] == 1]  # 해당 지역의 데이터 필터링
    exit_rate = city_data['EXIT'].mean() if not city_data.empty else 0  # 이탈률 계산
    work_exit_rate = pd.concat(
        [work_exit_rate, pd.DataFrame({'City': [city_name], 'ExitRate': [exit_rate]})]
    )

# WorkCity 이탈률 시각화
plt.figure(figsize=(12, 8))
sns.barplot(data=work_exit_rate, x='City', y='ExitRate', palette='coolwarm', hue='City')
plt.title('Exit Rate by Work City for Age 20s')
plt.xlabel('Work City')
plt.ylabel('Exit Rate')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()

# ii. 거주지 근무지 다른 경우의 이탈률 분석

# 거주지와 근무지 변수 필터링
residence_columns = [col for col in X_train.columns if col.startswith('ResidenceCity_')]
work_columns = [col for col in X_train.columns if col.startswith('WorkCity_')]

# 20대 고객 데이터 필터링
age_20s_data = X_train[X_train['Age'] == 1].copy()
age_20s_data['EXIT'] = y_train  # 이탈 여부 추가

# 거주지와 근무지 비교: 동일 여부 확인
def residence_work_match(row):
    for res_col, work_col in zip(residence_columns, work_columns):
        if row[res_col] == 1 and row[work_col] == 1:
            return 'Same'
    return 'Different'

# "Same"과 "Different" 레이블 추가
age_20s_data['Residence_Work_Match'] = age_20s_data.apply(residence_work_match, axis=1)

# 이탈률 계산
exit_rate = age_20s_data.groupby('Residence_Work_Match')['EXIT'].mean().reset_index()
exit_rate.columns = ['Residence_Work_Match', 'Exit_Rate']

# 시각화 - 별 차이 없음
plt.figure(figsize=(8, 6))
sns.barplot(data=exit_rate, x='Residence_Work_Match', y='Exit_Rate', palette='coolwarm')
plt.title('Exit Rate by Residence and Work Match for Age 20s')
plt.xlabel('Residence and Work Match')
plt.ylabel('Exit Rate')
plt.ylim(0, 1)
plt.show()

# 상세 데이터 출력
same_match_exit = age_20s_data[age_20s_data['Residence_Work_Match'] == 'Same']['EXIT'].mean()
different_match_exit = age_20s_data[age_20s_data['Residence_Work_Match'] == 'Different']['EXIT'].mean()

print("\nDetailed Exit Rates:")
print(f"Same Residence and Work: {same_match_exit:.2f}")
print(f"Different Residence and Work: {different_match_exit:.2f}")


# 7. 20대 고객 중, 연속적으로 사용량이 감소하는 고객군이 이탈할 위험이 높은지 확인

# 20대 고객 데이터 필터링
age_20s_data = X_train[X_train['Age'] == 1]
age_20s_data['EXIT'] = y_train  # 20대 고객의 이탈 여부를 추가

# UsageAmountCategory 열 정의
usage_columns = ['201809_UsageAmountCategory', '201808_UsageAmountCategory', '201807_UsageAmountCategory']

# 월별 사용량 변화 패턴 분석
# 연속적으로 사용량이 감소하는 고객 수를 계산
def count_consecutive_decreases(row):
    decreases = 0
    for i in range(len(usage_columns) - 1):
        if row[usage_columns[i]] > row[usage_columns[i + 1]]:  # 감소 여부
            decreases += 1
    return decreases

age_20s_data['Consecutive_Decreases'] = age_20s_data.apply(count_consecutive_decreases, axis=1)

# 연속적으로 사용량이 감소하는 고객군 분리
decrease_group = age_20s_data[age_20s_data['Consecutive_Decreases'] == len(usage_columns) - 1]  # 모든 월에서 감소
non_decrease_group = age_20s_data[age_20s_data['Consecutive_Decreases'] < len(usage_columns) - 1]  # 그렇지 않은 경우

# 이탈률 계산
churn_rate_decrease = decrease_group['EXIT'].mean()  # 감소 그룹의 이탈률
churn_rate_non_decrease = non_decrease_group['EXIT'].mean()  # 비감소 그룹의 이탈률

print(f"Churn Rate for Decrease Group: {churn_rate_decrease:.2f}")
print(f"Churn Rate for Non-Decrease Group: {churn_rate_non_decrease:.2f}")

# 시각화
plt.figure(figsize=(10, 6))
sns.barplot(
    x=['Decrease Group', 'Non-Decrease Group'],
    y=[churn_rate_decrease, churn_rate_non_decrease],
    palette='coolwarm'
)
plt.title('Churn Rate by Usage Decrease Pattern for Age 20s')
plt.ylabel('Churn Rate')
plt.ylim(0, 1)
plt.show()

# 8. 20대 - 성별에 따른 이탈율 - 거의 차이 없음

# 성별에 따른 이탈률 분석
gender_exit_rate = age_20s_data.groupby('GenderCode')['EXIT'].mean().reset_index()  # 성별별 평균 이탈률 계산
gender_exit_rate['Gender'] = gender_exit_rate['GenderCode'].map({1: 'Male', 2: 'Female'})  # 성별 코드 매핑

# 이탈률 시각화
plt.figure(figsize=(8, 5))
sns.barplot(x='Gender', y='EXIT', data=gender_exit_rate, palette='Set2')
plt.title('Exit Rate by Gender for Age 20s')
plt.xlabel('Gender')
plt.ylabel('Exit Rate')
plt.ylim(0, 1)  # Y축 범위 설정 (0에서 1 사이)
plt.show()

# 성별별 이탈률 출력
print("Gender-wise Exit Rates:")
print(gender_exit_rate[['Gender', 'EXIT']])