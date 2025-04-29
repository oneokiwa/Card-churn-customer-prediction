# ===================================================================================
# 📌 데이터 전처리 파이프라인 요약
# 1️⃣ 월별 병합(Merge)
# 2️⃣ VIP 필터링(Filter)
# 3️⃣ EXIT 변수 추가(Labeling)
# 4️⃣ 언더샘플링(UnderSampling) → 이탈 분석 데이터셋 생성
# 최종 결과 파일: "data/종합정보_회원번호기준_10월포함_이탈고객_균형.csv"
# ===================================================================================
###### 1️⃣. 월별 (7-9월) 데이터를 병합하여 하나의 종합 정보 파일을 생성 #####
import pandas as pd

def merge_monthly_data(month):
    # 파일명 패턴 설정
    member_file = f"data/{month}_회원정보.csv"
    balance_file = f"data/{month}_잔액정보.csv"
    credit_file = f"data/{month}_신용정보.csv"
    bill_file = f"data/{month}_청구정보.csv"
    sales_file = f"data/{month}_승인매출정보.csv"
    output_file = f"data/{month}_종합정보.csv"

    # 사용할 열 정의
    columns_member = [
        "기준년월", "발급회원번호", "VIP등급코드", "남녀구분코드",
        "회원여부_이용가능", "거주시도명", "직장시도명", "유효카드수_신용",
        "이용카드수_신용", "연령", "Life_Stage"
    ]

    columns_balance = [
        "기준년월", "발급회원번호", "연체잔액_B0M", "잔액_B0M"
    ]

    columns_credit = [
        "기준년월", "발급회원번호", "카드이용한도금액"
    ]

    columns_bill = [
        "기준년월", "발급회원번호", "청구금액_B0", "청구금액_R3M"
    ]

    columns_sales = [
        "기준년월", "발급회원번호",
        "이용금액_신용_B0M", "이용금액_온라인_R3M", "이용금액_온라인_B0M", "이용금액_오프라인_B0M",
        "이용금액_오프라인_R3M", "이용금액_페이_온라인_B0M","이용금액_페이_오프라인_B0M",
        "이용금액_페이_온라인_R3M", "이용금액_페이_오프라인_R3M", "이용금액대"
    ]

    try:
        # 파일 불러오기
        df_member = pd.read_csv(member_file, usecols=columns_member, nrows=3000000, encoding="utf-8")
        df_balance = pd.read_csv(balance_file, usecols=columns_balance, nrows=3000000, encoding="utf-8")
        df_credit = pd.read_csv(credit_file, usecols=columns_credit, nrows=3000000, encoding="utf-8")
        df_bill = pd.read_csv(bill_file, usecols=columns_bill, nrows=3000000, encoding="utf-8")
        df_sales = pd.read_csv(sales_file, usecols=columns_sales, nrows=3000000, encoding="utf-8")

        # 데이터 병합
        df_merged = pd.merge(df_member, df_balance, on=["기준년월", "발급회원번호"], how="inner")
        df_merged = pd.merge(df_merged, df_credit, on=["기준년월", "발급회원번호"], how="inner")
        df_merged = pd.merge(df_merged, df_bill, on=["기준년월", "발급회원번호"], how="inner")
        df_merged = pd.merge(df_merged, df_sales, on=["기준년월", "발급회원번호"], how="inner")

        # 결과 저장
        df_merged.to_csv(output_file, index=False, encoding="utf-8")

        print(f"{month} 병합 완료. 저장 파일: {output_file}")
        print(f"병합된 데이터의 크기: {df_merged.shape}\n")

    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}\n")
    except pd.errors.EmptyDataError:
        print(f"데이터가 비어 있습니다: {month}\n")
    except Exception as e:
        print(f"오류 발생: {e}\n")

merge_monthly_data("201807")
merge_monthly_data("201808")
merge_monthly_data("201809")


######## 2️⃣. 회원번호 기준 데이터 묶기 작업 #########
# 201807, 201808, 201809 파일 읽기
df_201807 = pd.read_csv("data/201807_종합정보.csv", encoding="utf-8")
df_201808 = pd.read_csv("data/201808_종합정보.csv", encoding="utf-8")
df_201809 = pd.read_csv("data/201809_종합정보.csv", encoding="utf-8")
df_201810 = pd.read_csv("data/201810_종합정보.csv", encoding="utf-8")

# '발급회원번호'를 제외한 나머지 칼럼에 기준연월 접두어 추가
df_201807 = df_201807.rename(columns=lambda x: f"201807_{x}" if x != "발급회원번호" else x)
df_201808 = df_201808.rename(columns=lambda x: f"201808_{x}" if x != "발급회원번호" else x)
df_201809 = df_201809.rename(columns=lambda x: f"201809_{x}" if x != "발급회원번호" else x)
df_201810 = df_201810.rename(columns=lambda x: f"201810_{x}" if x != "발급회원번호" else x)

# 발급회원번호를 기준으로 데이터 병합
df_merged = pd.merge(df_201807, df_201808, on="발급회원번호", how="outer")
df_merged = pd.merge(df_merged, df_201809, on="발급회원번호", how="outer")
df_merged = pd.merge(df_merged, df_201810, on="발급회원번호", how="outer")

# 결과 확인 (선택사항)
print(f"병합된 데이터 크기: {df_merged.shape}")

# 결과를 새로운 파일로 저장
df_merged.to_csv("data/종합정보_회원번호기준_10월포함_병합.csv", index=False, encoding="utf-8") #병합된 데이터 크기: (3000000, 101)


######### 3️⃣. vip 고객 필터링 ###########

# 종합정보_회원번호기준_10월포함_병합.csv 파일 읽기
df = pd.read_csv("data/종합정보_회원번호기준_10월포함_병합.csv", encoding="utf-8")

# vip등급코드가 '_'인 고객 제외
df_filtered = df[(df['201807_VIP등급코드'] != '_') &
                 (df['201808_VIP등급코드'] != '_') &
                 (df['201809_VIP등급코드'] != '_') &
                 (df['201810_VIP등급코드'] != '_')]

# 결과를 새로운 CSV 파일로 저장
df_filtered.to_csv("data/종합정보_회원번호기준_10월포함_VIP고객.csv", index=False, encoding="utf-8")

# 결과 확인
print(f"필터링된 데이터의 크기: {df_filtered.shape}") #(597592, 76)

######### 4️⃣. 이탈 변수 추가 ###########

# 파일 경로 설정
input_file = "data/종합정보_회원번호기준_10월포함_VIP고객.csv"
output_file = "data/종합정보_회원번호기준_10월포함_VIP고객_이탈추가.csv"

# 데이터 불러오기
df = pd.read_csv(input_file)

# '201810_청구금액_B0' 값이 0원인 경우 EXIT: 1, 아니면 EXIT: 0 설정
df['EXIT'] = df['201810_청구금액_B0'].apply(lambda x: 1 if x == 0 else 0)

# 결과 저장
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"EXIT 변수를 추가한 데이터를 '{output_file}'로 저장했습니다.")
print(f"필터링된 데이터의 크기: {df.shape}") #(597592, 102)

########## 5️⃣. 언더샘플링 작업 #########

# 데이터 읽기
df = pd.read_csv("data/종합정보_회원번호기준_10월포함_VIP고객_이탈추가.csv", encoding="utf-8")

# EXIT 값이 1인 고객의 수 출력
exit_count = df[df['EXIT'] == 1].shape[0]

print(f"이탈고객 분류 후 데이터 크기: {df.shape}") # 597592명
print(f"이탈 고객의 수: {exit_count}명") #  24587명

# 이탈 고객만 필터링
exit_customers = df[df['EXIT'] == 1]

# 유지 고객만 필터링
non_exit_customers = df[df['EXIT'] == 0]

# 유지 고객에서 무작위로 이탈 고객 수만큼 샘플링
non_exit_sampled = non_exit_customers.sample(n=exit_customers.shape[0], random_state=42)

# 이탈 고객과 샘플링된 유지 고객을 합침
balanced_df = pd.concat([exit_customers, non_exit_sampled])

# 새로운 데이터 저장 : 종합정보_회원번호기준_10월포함_이탈고객_균형
balanced_df.to_csv("data/종합정보_회원번호기준_10월포함_이탈고객_균형.csv", index=False, encoding="utf-8")

# 결과 확인
print(f"균형잡힌 데이터 크기: {balanced_df.shape}") # (49174, 102)