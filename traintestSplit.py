# ===================================================================================
# 📌 데이터셋 분리 및 저장 파이프라인 요약
# 1️⃣ 10월 데이터 분리
# 2️⃣ 7월~9월 데이터 분리
# 3️⃣ 학습용/검증용/테스트용 데이터셋 분할
# ===================================================================================

# 10월만 따로 빼기
import pandas as pd

# 파일 경로 설정
input_file = "data/열분리_영어변환.csv"
output_file = "data/열분리_10월정보만.csv"

# 데이터 불러오기
df = pd.read_csv(input_file)

# '발급회원번호' 열과 '201810_' 접두어가 포함된 열, 'EXIT' 열을 선택
columns_to_save = ["MemberID"] + ['GenderCode'] + ['Age'] + ['VIPGradeCode'] + [col for col in df.columns if col.startswith("WorkCity")] + [col for col in df.columns if col.startswith("ResidenceCity")] + [col for col in df.columns if col.startswith("201810_")] + ["EXIT"]

# 해당 열들로 데이터프레임 생성
df_filtered = df[columns_to_save]

# 결과 저장
df_filtered.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"발급회원번호와, 남녀구분코드, 연령, VIP, 거주, 직장시도명, 201810_ 접두어가 포함된 열, EXIT 열을 '{output_file}'로 저장했습니다.")

print(f"필터링된 데이터의 크기: {df_filtered.shape}") #(49174, 27)

######## 3개월치 종합정보만 따로 저장 #########
import pandas as pd

# 파일 경로 설정
input_file = "data/열분리_영어변환.csv"
output_file = "data/열분리_3개월만_EXIT포함.csv"

# 데이터 불러오기
df = pd.read_csv(input_file)

# '201810_' 접두어가 포함된 열과 'EXIT' 열 제외
columns_to_keep = [col for col in df.columns if not col.startswith("201810_")]

# 해당 열들로 데이터프레임 생성
df_filtered = df[columns_to_keep]

# 결과 저장
df_filtered.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"201810_ 접두어가 제외된 데이터를 '{output_file}'로 저장했습니다.")
print(f"필터링된 데이터의 크기: {df_filtered.shape}") #(49174, 99)

# 훈련, 테스트, 검증 데이터셋 나누기
######## 훈련 검증 데이터셋 분리 ########
import pandas as pd
from sklearn.model_selection import train_test_split

# 파일 경로 설정
input_file_features = "data/열분리_3개월만.csv"
input_file_target = "data/열분리_10월정보만.csv"

# 데이터 불러오기
df_features = pd.read_csv(input_file_features)
df_target = pd.read_csv(input_file_target)

# 종속 변수 (EXIT 열)와 독립 변수 (그 외 열들) 분리
X = df_features
y = df_target["EXIT"]

# 훈련(60%), 검증(20%), 테스트(20%)로 나누기
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 각 데이터셋을 CSV 파일로 저장
X_train.to_csv("train/X_train.csv", index=False, encoding="utf-8-sig")
y_train.to_csv("train/y_train.csv", index=False, encoding="utf-8-sig")
X_val.to_csv("train/X_val.csv", index=False, encoding="utf-8-sig")
y_val.to_csv("train/y_val.csv", index=False, encoding="utf-8-sig")
X_test.to_csv("train/X_test.csv", index=False, encoding="utf-8-sig")
y_test.to_csv("train/y_test.csv", index=False, encoding="utf-8-sig")

print("훈련, 검증, 테스트 데이터셋이 저장되었습니다.")
