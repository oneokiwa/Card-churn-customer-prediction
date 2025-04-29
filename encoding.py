# ===================================================================================
# 📌 데이터 전처리 파이프라인 요약
# 1️⃣ 결측치 처리
# 2️⃣ 인코딩 작업
# 3️⃣ 열 정리
# 4️⃣ 변수명 영어로 변환
# 최종 결과 파일: "data/열분리_영어변환.csv"
# ===================================================================================

######## 1️⃣. 결측치 대체 ########
import pandas as pd

# 데이터 불러오기
data = pd.read_csv("data/종합정보_회원번호기준_10월포함_이탈고객_균형.csv")

# 결측치를 "없음"으로 대체할 열 목록
columns_to_replace = [
    "201807_직장시도명",
    "201808_직장시도명",
    "201809_직장시도명",
    "201810_직장시도명"
]

# 해당 열들의 결측치를 "없음"으로 대체
data[columns_to_replace] = data[columns_to_replace].fillna("없음")

# 변경된 데이터를 새 파일로 저장
data.to_csv("data/종합정보_10월포함_결측치처리.csv", index=False)

# 확인 메시지
print("결측치가 '없음'으로 대체된 파일이 저장되었습니다.")


###### 2️⃣. 원-핫 인코딩, 레이블 인코딩 수행 #########
# "data/종합정보_10월포함_결측치처리.csv"

### 1. 거주시도, 직장시도 18개로 인코딩 ###

# 파일 경로
input_file = "data/종합정보_10월포함_결측치처리.csv"
output_file = "data/거주시도_직장시도_인코딩.csv"

# 데이터 읽기
df = pd.read_csv(input_file)

# 원-핫 인코딩을 수행할 열 이름들
columns_to_encode = [
    '201807_거주시도명', '201808_거주시도명', '201809_거주시도명', '201810_거주시도명',
    '201807_직장시도명', '201808_직장시도명', '201809_직장시도명', '201810_직장시도명'
]

# 거주시도명/직장시도명 값 리스트 (없음 포함)
city_names = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
              '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주', '없음']

# 기존 열을 원-핫 인코딩하여 덮어쓰기
for col in columns_to_encode:
    # 해당 열을 원-핫 인코딩
    df[col] = df[col].apply(lambda x: [1 if city == x else 0 for city in city_names])

# 결과를 새로운 CSV 파일로 저장
df.to_csv(output_file, index=False)

print(f"원-핫 인코딩 결과가 '{output_file}'에 저장되었습니다.")

### 2. Life Stage 레이블 인코딩 ###

# 파일 경로
input_file = "data/거주시도_직장시도_인코딩.csv"
output_file = "data/Life_Stage_인코딩.csv"

# 데이터 읽기
df = pd.read_csv(input_file)

# 레이블 인코딩을 수행할 열 이름들
columns_to_encode = ['201807_Life_Stage', '201808_Life_Stage', '201809_Life_Stage', '201810_Life_Stage']

# Life Stage 변수 순서 정의
life_stage_order = {
    "1.Single": 1,
    "2.가족형성기": 2,
    "3.자녀출산기": 3,
    "4.자녀성장기(1)": 4,
    "5.자녀성장기(2)": 5,
    "6.자녀출가기": 6,
    "7.노령": 7
}

# 레이블 인코딩 수행
for col in columns_to_encode:
    df[col] = df[col].map(life_stage_order)

# 결과를 새로운 CSV 파일로 저장
df.to_csv(output_file, index=False)

print(f"레이블 인코딩 결과가 '{output_file}'에 저장되었습니다.")

### 3. 연령 레이블 인코딩 ###

# 파일 경로
input_file = "data/Life_Stage_인코딩.csv"
output_file = "data/연령_인코딩.csv"

# 데이터 읽기
df = pd.read_csv(input_file)

# 레이블 인코딩을 수행할 열 이름들
columns_to_encode = ['201807_연령', '201808_연령', '201809_연령', '201810_연령']

# 연령 변수 순서 정의
age_order = {
    "20대": 1,
    "30대": 2,
    "40대": 3,
    "50대": 4,
    "60대": 5,
    "70대이상": 6
}

# 레이블 인코딩 수행
for col in columns_to_encode:
    df[col] = df[col].map(age_order).astype(int)  # 정수형으로 변환

# 결과를 새로운 CSV 파일로 저장
df.to_csv(output_file, index=False)

print(f"레이블 인코딩 결과가 '{output_file}'에 저장되었습니다.")

### 4. 이용금액대 레이블 인코딩 ###

# 파일 경로
input_file = "data/연령_인코딩.csv"
output_file = "data/최종_인코딩.csv"

# 데이터 읽기
df = pd.read_csv(input_file)

# 레이블 인코딩을 수행할 열 이름들
columns_to_encode = ['201807_이용금액대', '201808_이용금액대', '201809_이용금액대', '201810_이용금액대']

# 연령 변수 순서 정의
amt_use_order = {
    "01.100만원+": 1,
    "02.50만원+": 2,
    "03.30만원+": 3,
    "04.10만원+": 4,
    "05.10만원-": 5,
    "09.미사용": 6
}

# 레이블 인코딩 수행
for col in columns_to_encode:
    df[col] = df[col].map(amt_use_order)

# 결과를 새로운 CSV 파일로 저장
df.to_csv(output_file, index=False)

print(f"레이블 인코딩 결과가 '{output_file}'에 저장되었습니다.")

### 5. 남녀구분코드 하나만 남기기 (불필요한 중복 열 제외) ###

# 파일 경로 설정
input_file = "data/최종_인코딩.csv"

# 데이터 불러오기
df = pd.read_csv(input_file)

# '201807_남녀구분코드' 열만 남기고 나머지 열 삭제
df = df.drop(columns=["201808_남녀구분코드", "201809_남녀구분코드", "201810_남녀구분코드"])

# '201807_남녀구분코드' 열 이름을 '남녀구분코드'로 수정
df = df.rename(columns={"201807_남녀구분코드": "남녀구분코드"})

# 결과를 기존 파일에 덮어쓰기
df.to_csv(input_file, index=False, encoding="utf-8-sig")
print(f"남녀구분코드 수정된 데이터를 '{input_file}'에 덮어썼습니다.")


### 6. 거주시도명, 직장시도명 하나만 남기기 ###

# 파일 경로 설정
input_file = "data/최종_인코딩.csv"

# 데이터 불러오기
df = pd.read_csv(input_file)

# '201807_직장시도명, 거주시도명' 열만 남기고 나머지 열 삭제
df = df.drop(columns=["201808_거주시도명", "201809_거주시도명", "201810_거주시도명", "201808_직장시도명", "201809_직장시도명", "201810_직장시도명"])

# '201807_직장, 거주' 열 이름을 '직장, 거주'로 수정
df = df.rename(columns={"201807_거주시도명": "거주시도명"})
df = df.rename(columns={"201807_직장시도명": "직장시도명"})

# 결과를 기존 파일에 덮어쓰기
df.to_csv(input_file, index=False, encoding="utf-8-sig")
print(f"거주, 직장시도명 수정된 데이터를 '{input_file}'에 덮어썼습니다.")

### 7. 거주시도명, 직장시도명 열 분리 ###
import ast  # 안전하게 문자열을 파싱하는 라이브러리

# "data/최종_인코딩.csv" 파일 로드
data = pd.read_csv("data/최종_인코딩.csv")

# 문자열 형태의 리스트를 실제 리스트로 변환하는 함수 정의
def parse_list(column):
    return column.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 거주시도명과 직장시도명 열의 문자열 리스트를 실제 리스트로 변환
data["거주시도명"] = parse_list(data["거주시도명"])
data["직장시도명"] = parse_list(data["직장시도명"])

# 거주시도명 리스트에서 각 범주를 개별 열로 분리
residence_categories = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
]
for idx, category in enumerate(residence_categories):
    column_name = f"거주시도명_{category}"
    data[column_name] = data["거주시도명"].apply(lambda x: x[idx] if isinstance(x, list) else 0)

# 직장시도명 리스트에서 각 범주를 개별 열로 분리
work_categories = residence_categories + ["없음"]
for idx, category in enumerate(work_categories):
    column_name = f"직장시도명_{category}"
    data[column_name] = data["직장시도명"].apply(lambda x: x[idx] if isinstance(x, list) else 0)

# 기존의 리스트 형태 열 제거
data = data.drop(columns=["거주시도명", "직장시도명"])

# 결과를 "data/열분리.csv"로 저장
data.to_csv("data/열분리.csv", index=False)

print("열 분리 후 저장 완료: data/열분리.csv")

########## 연령, VIP등급코드 하나만 남기기 (불필요한 중복 열 제외) ##########

### 8. 연령 중복 제거 ###
# 파일 경로 설정
input_file = "data/열분리.csv"

# 데이터 불러오기
df = pd.read_csv(input_file)

# '201807_남녀구분코드' 열만 남기고 나머지 열 삭제
df = df.drop(columns=["201808_연령", "201809_연령", "201810_연령"])

# '201807_남녀구분코드' 열 이름을 '남녀구분코드'로 수정
df = df.rename(columns={"201807_연령": "연령"})

# 결과를 기존 파일에 덮어쓰기
df.to_csv(input_file, index=False, encoding="utf-8-sig")
print(f"연령 수정된 데이터를 '{input_file}'에 덮어썼습니다.")

### 9. VIP 등급 코드 중복 제거 ###
# 파일 경로 설정
input_file = "data/열분리.csv"

# 데이터 불러오기
df = pd.read_csv(input_file)

# '201807_남녀구분코드' 열만 남기고 나머지 열 삭제
df = df.drop(columns=["201808_VIP등급코드", "201809_VIP등급코드", "201810_VIP등급코드"])

# '201807_남녀구분코드' 열 이름을 '남녀구분코드'로 수정
df = df.rename(columns={"201807_VIP등급코드": "VIP등급코드"})

# 결과를 기존 파일에 덮어쓰기
df.to_csv(input_file, index=False, encoding="utf-8-sig")
print(f"VIP등급코드 수정된 데이터를 '{input_file}'에 덮어썼습니다.")


# ----------------------------------------------------------------------------------

######### 영문 변경 ############

# 파일 경로 설정
input_file = "data/열분리.csv"
output_file = "data/열분리_영어변환.csv"

# CSV 파일 로드
data = pd.read_csv(input_file)

# 변수명 변경 맵핑 (접두어는 유지)
column_mapping = {
    "기준년월": "YearMonth",
    "발급회원번호": "MemberID",
    "VIP등급코드": "VIPGradeCode",
    "남녀구분코드": "GenderCode",
    "회원여부_이용가능": "MembershipAvailable",
    "거주시도명": "ResidenceCity",
    "직장시도명": "WorkCity",
    "유효카드수_신용": "ValidCardCount",
    "이용카드수_신용": "UsedCardCount",
    "연령": "Age",
    "Life_Stage": "LifeStage",
    "잔액_B0M": "Balance_B0M",
    "연체잔액_B0M": "OverdueBalance_B0M",
    "카드이용한도금액": "CardLimit",
    "청구금액_B0": "BillingAmount_B0",
    "청구금액_R3M": "BillingAmount_R3M",
    "이용금액_신용_B0M": "UsageAmount_Credit_B0M",
    "이용금액_온라인_R3M": "UsageAmount_Online_R3M",
    "이용금액_오프라인_R3M": "UsageAmount_Offline_R3M",
    "이용금액_온라인_B0M": "UsageAmount_Online_B0M",
    "이용금액_오프라인_B0M": "UsageAmount_Offline_B0M",
    "이용금액_페이_온라인_B0M": "UsageAmount_PayOnline_B0M",
    "이용금액_페이_오프라인_B0M": "UsageAmount_PayOffline_B0M",
    "이용금액_페이_온라인_R3M": "UsageAmount_PayOnline_R3M",
    "이용금액_페이_오프라인_R3M": "UsageAmount_PayOffline_R3M",
    "이용금액대": "UsageAmountCategory",
    "서울": "Seoul",
    "부산": "Busan",
    "대구": "Daegu",
    "인천": "Incheon",
    "광주": "Gwangju",
    "대전": "Daejeon",
    "울산": "Ulsan",
    "세종": "Sejong",
    "경기": "Gyeonggi",
    "강원": "Gangwon",
    "충북": "Chungbuk",
    "충남": "Chungnam",
    "전북": "Jeonbuk",
    "전남": "Jeonnam",
    "경북": "Gyeongbuk",
    "경남": "Gyeongnam",
    "제주": "Jeju",
    "없음": "None"
}

# 접두어 변환 맵핑
def translate_prefix(prefix):
    prefix_mapping = {
        "거주시도명": "ResidenceCity",
        "직장시도명": "WorkCity"
    }
    return prefix_mapping.get(prefix, prefix)

# 변수명 변환 (접두어 영어 번역 포함)
def rename_columns(column_name):
    parts = column_name.split('_', 1)  # 접두어와 나머지 분리
    if len(parts) == 2:
        prefix, base_name = parts
        translated_prefix = translate_prefix(prefix)  # 접두어 변환
        if base_name in column_mapping:
            return f"{translated_prefix}_{column_mapping[base_name]}"
        else:
            return f"{translated_prefix}_{base_name}"
    elif column_name in column_mapping:
        return column_mapping[column_name]
    return column_name

# 모든 열 이름 변환
data.columns = [rename_columns(col) for col in data.columns]

# 결과 저장
data.to_csv(output_file, index=False)
print(f"변수명을 영어로 변환한 파일이 저장되었습니다: {output_file}")