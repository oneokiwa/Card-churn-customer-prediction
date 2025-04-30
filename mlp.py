#===================================================================================
# 📌  MLP 파이프라인 요약
# 1️⃣. 데이터 로딩 및 전처리
# 2️⃣. MLP 모델 구축 및 학습
# 3️⃣. 모델 평가 및 지표 출력
# 4️⃣. Permutation Importance 분석
# 5️⃣. K-Means 클러스터링
# 6️⃣. 클러스터별 이탈률 분석
# 7️⃣. SHAP 분석
# 8️⃣. 잔액 구간별 이탈률 분석

# 데이터 분류, 중요 변수 분석, 고객 군집화, 설명력 분석까지 종합적으로 수행
#===================================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import shap
import matplotlib.pyplot as plt

# 모델 개선을 위한 추가 조치 (과적합 방지)
# 1. Early Stopping 추가
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

from tensorflow.keras.regularizers import l2


# 데이터 불러오기
X_train = pd.read_csv('train/X_train.csv')
X_val = pd.read_csv('train/X_val.csv')
X_test = pd.read_csv('train/X_test.csv')
y_train = pd.read_csv('train/y_train.csv').squeeze()
y_val = pd.read_csv('train/y_val.csv').squeeze()
y_test = pd.read_csv('train/y_test.csv').squeeze()

# 범주형 데이터 인코딩 (Label Encoding)
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


###  MLP(다계층 퍼셉트론) 모델 구축 ###
# 입력층: 특성 개수만큼 뉴런 포함
# 첫 번째 은닉층: 64개 뉴런, ReLU 활성화 함수, Dropout(0.3) 적용
# 두 번째 은닉층: 32개 뉴런, ReLU 활성화 함수, Dropout(0.3) 적용
# 출력층: 1개 뉴런, sigmoid 활성화 함수 (이진 분류)
# 인공신경망 모델 정의 (MLP)

# 모델 성능 개선
model = Sequential([     # Sequential 모델 사용
    keras.Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),  # L2 정규화 추가, ReLU 활성화 함수 :  비선형성을 유지
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),  # L2 정규화 추가
    Dropout(0.3),
    Dense(1, activation='sigmoid') # sigmoid 사용, 출력값을 0~1 사이의 확률로 변환
                                   # 0.5 이상이면 이탈 (1)
])

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

###  모델 학습  ###
# Adam 옵티마이저, binary_crossentropy 손실 함수 사용
# epochs=50, batch_size=32
# 학습 과정에서 손실(Loss) 및 정확도(Accuracy) 그래프 출력

history = model.fit(X_train_scaled, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val_scaled, y_val),
                    verbose=1,
                    callbacks=[early_stopping])  # Early Stopping 적용

############ 학습 결과 시각화 (Loss & Accuracy) ##############

# 손실(Loss) 그래프
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# 정확도(Accuracy) 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# 모델 평가 (테스트 데이터)
y_test_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

# 성능 평가 지표 출력
accuracy = accuracy_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, model.predict(X_test_scaled))
classification = classification_report(y_test, y_test_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {auc:.4f}")
print("Classification Report:")
print(classification)


# # 📌 Permutation Importance를 위한 래퍼 클래스 생성
class KerasWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        """Sklearn 인터페이스를 위해 fit 메서드 추가 (실제로는 사용되지 않음)"""
        pass

    def predict(self, X):
        """MLP 모델의 예측 결과를 0 또는 1로 변환"""
        return (self.model.predict(X, verbose=0) > 0.5).astype(int).flatten()

# 📌 Sklearn 인터페이스를 따르는 MLP 래퍼 모델 생성
wrapped_model = KerasWrapper(model)

# 📌 Permutation Importance 계산
result = permutation_importance(
    estimator=wrapped_model,  # MLP 모델 래퍼 전달
    X=X_test_scaled,
    y=y_test,
    scoring="accuracy",
    n_repeats=10,
    random_state=42
)

# 📌 변수 중요도 시각화
feature_importances = result.importances_mean
sorted_idx = np.argsort(feature_importances)[::-1]
top_n = 10  # 상위 10개 변수만 표시

plt.figure(figsize=(20, 10))
plt.barh(X_test.columns[sorted_idx][:top_n], feature_importances[sorted_idx][:top_n])
plt.xlabel("Permutation Importance")
plt.title("MLP Feature Importance")
plt.gca().invert_yaxis()  # 큰 값이 위로 가도록 설정
plt.show()

# 중요한 변수 선택
important_features = [
    "201809_ConsecutiveNonPerformanceMonths_Basic_24M_Card",
    "201809_UsageAmount_Credit_B0M",
    "201809_Balance_B0M",
    "201809_BillingAmount_B0",
    "201809_BillingAmount_R3M"
]

########### K-Means 분석 ###########
# 군집화를 위한 데이터 선택 (중요 변수만 사용)
cluster_data = X_test_scaled[:, [X_test.columns.get_loc(col) for col in important_features]]

# 최적의 K 찾기 (Elbow Method)
inertia = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(cluster_data)
    inertia.append(kmeans.inertia_)

# Elbow Method 그래프 출력
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# 최적의 K=3으로 K-Means 적용
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(cluster_data)

# PCA를 사용하여 시각화 (2D 변환)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(cluster_data)

# 군집화 결과 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette='viridis')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Customer Segmentation using K-Means")
plt.legend(title="Cluster")
plt.show()


######### 추가분석 #########
# K-Means 결과를 데이터프레임에 추가
X_test_clustered = pd.DataFrame(X_test_scaled, columns=X_test.columns)
X_test_clustered['Cluster'] = clusters
X_test_clustered['EXIT'] = y_test.values  # 실제 이탈 여부 추가

# 클러스터별 평균 이탈율 계산
cluster_exit_rates = X_test_clustered.groupby('Cluster')['EXIT'].mean()

# 시각화
plt.figure(figsize=(8, 5))
sns.barplot(x=cluster_exit_rates.index, y=cluster_exit_rates.values, palette='viridis')
plt.xlabel("Cluster")
plt.ylabel("EXIT Rate")
plt.title("EXIT Rate per Cluster")
plt.show()

######## shap 분석 #########
# Cluster 0의 데이터만 필터링
cluster_0_data = X_test_clustered[X_test_clustered['Cluster'] == 0]
cluster_0_features = cluster_0_data.drop(columns=['Cluster', 'EXIT'])  # 독립변수만 사용

# SHAP 분석을 위한 Explainer 생성
explainer = shap.Explainer(model, cluster_0_features)
shap_values = explainer(cluster_0_features)

# SHAP Summary Plot (변수 중요도 분석)
shap.summary_plot(shap_values, cluster_0_features, feature_names=cluster_0_features.columns)

######### 잔액 구간 분석 ##########
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 클러스터 0 데이터 필터링
cluster_0_data = X_test_clustered[X_test_clustered['Cluster'] == 0]

# 잔액 변수 선택
balance_var = "201809_Balance_B0M"

# 📌 잔액 구간 나누기 (3개 구간: 낮음, 중간, 높음)
bins = [0, 7500000, 15000000, 22588073]  # 3개의 구간 (조정 가능)
labels = ["Low (0 ~ 7.5M)", "Medium (7.5M ~ 15M)", "High (15M ~ 22.5M)"]

cluster_0_data['Balance_Bin'] = pd.cut(cluster_0_data[balance_var], bins=bins, labels=labels, include_lowest=True)

# 📌 각 구간별 이탈율(Exit Rate) 계산
balance_exit_rates = cluster_0_data.groupby('Balance_Bin')['EXIT'].mean()

# 📌 이탈율 시각화 (Bar Plot)
plt.figure(figsize=(12, 7))
sns.barplot(x=balance_exit_rates.index, y=balance_exit_rates.values, palette='coolwarm')
plt.xlabel("Balance Range")
plt.ylabel("EXIT Rate")
plt.title(f"EXIT Rate by {balance_var} in Cluster 0")
plt.ylim(0, 1)  # 0~100% 비율 표현
plt.show()