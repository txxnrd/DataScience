import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

# csv 파일명 저장
filenames = [
    "2017_6.csv", "2017_9.csv", "2017_11.csv",
    "2018_6.csv", "2018_9.csv", "2018_11.csv",
    "2019_6.csv", "2019_9.csv", "2019_11.csv",
    "2020_6.csv", "2020_9.csv", "2020_11.csv",
    "2021_6.csv", "2021_9.csv", "2021_11.csv",
    "2022_6.csv", "2022_9.csv", "2022_11.csv",
    "2023_6.csv", "2023_9.csv", "2023_11.csv",
    "2024_6.csv", "2024_9.csv", "2024_11.csv",
]

# Create a dictionary to hold the dataframes
dataframes = {} #key 값 2017_6의 형태로 데이터 저장
standard_max = {} # 각 시험마다 최댓값 저장(정규화에 사용)
standard_min = {} # 각 시험마다 최솟값 저장(정규화에 사용)

# key 값마다 dataframe 저장
for filename in filenames:
    max_and_min = []
    var_name = filename.split('.')[0]  # 2017_6 형태로 추출
    df = pd.read_csv(filename, encoding='utf-8')
    standard_max[var_name] = df['standard_score'].max()
    standard_min[var_name] = df['standard_score'].min()
    dataframes[var_name] = df
# train data, test data 추출
train_data = []
train_target = []
test_data = []
test_target = []

# 정규화를 위해 전체 표준점수 최댓값, 최솟갑 추출
global_max_score = min(standard_max.values())
global_min_score = max(standard_min.values())


# 표준점수 전체 표준점수 최댓값 최솟값 사이 값 같는 normalized_score로 변경하는 함수
def normalize_score(df, global_min, global_max, decimals=0):
    local_min = df['standard_score'].min()
    local_max = df['standard_score'].max()
    df['normalized_score'] = (df['standard_score'] - local_min) / (local_max - local_min) * (
                global_max - global_min) + global_min
    df['normalized_score'] = df['normalized_score'].round(decimals)
    return df


# normalize_score 함수 적용한 새로운 함수 만듦
for key in dataframes.keys():
    df = dataframes[key]
    normalized_df = normalize_score(df, global_min_score, global_max_score)
    dataframes[key] = normalized_df

    # Convert male and female columns to numeric, removing commas
    normalized_df['male'] = pd.to_numeric(normalized_df['male'].astype(str).str.replace(',', ''), errors='coerce')
    normalized_df['female'] = pd.to_numeric(normalized_df['female'].astype(str).str.replace(',', ''), errors='coerce')

    # Group by normalized_score and sum male and female columns
    grouped_df = normalized_df.groupby('normalized_score', as_index=False).agg({
        'male': 'sum',
        'female': 'sum'
    })


    # 모든 normalized_score를 갖는 새로운 dataframe 생성
    all_scores = pd.DataFrame({'normalized_score': range(global_min_score, global_max_score + 1)})
    # 비어있는 row를 다 0으로 대체하는 작업
    complete_df = pd.merge(all_scores, grouped_df, on='normalized_score', how='left').fillna(0)
    # int로 변환
    complete_df['male'] = complete_df['male'].astype(str).str.replace(',', '').astype(float).round().astype(int)
    complete_df['female'] = complete_df['female'].astype(str).str.replace(',', '').astype(float).round().astype(int)

    # 기존 값을 대체
    dataframes[key] = complete_df
    # Save to CSV with the name normalized_{key}.csv
    filename = f'normalized_{key}.csv'
    complete_df.to_csv(filename, index=False)

# 실제 train을 위한 merge 과정
for year in range(2017, 2022):
    june_data = dataframes[f'{year}_6']
    september_data = dataframes[f'{year}_9']
    november_data = dataframes[f'{year}_11']


    merged_data = pd.merge(june_data, september_data, on='normalized_score', suffixes=('_june', '_sept'))

    #  training and testing sets로 나눠주기
    if year == 2021:
        test_data.append(merged_data[['male_june', 'female_june', 'male_sept', 'female_sept']])
        test_target.append(november_data[['male', 'female']])
    else:
        train_data.append(merged_data[['male_june', 'female_june', 'male_sept', 'female_sept']])
        train_target.append(november_data[['male', 'female']])


X_train = pd.concat(train_data)
y_train = pd.concat(train_target)
X_test = pd.concat(test_data)
y_test = pd.concat(test_target)


print("Training data:")
print(X_train)
print(y_train)
print("Testing data:")
print(X_test)
print(y_test)

# 모델 생성
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')