import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

# df 보관 위해 2017_6 형태로 갖는 dictionary 생성
dataframes = {} # key 값 2017_6의 형태로 데이터 저장
standard_max = {} # 각 시험마다 최댓값 저장(정규화에 사용)
standard_min = {} # 각 시험마다 최솟값 저장(정규화에 사용)

# key 값마다 dataframe 저장
for filename in filenames:
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

    # string 형태 자료가 있어서 정수형으로 변환
    normalized_df['male'] = pd.to_numeric(normalized_df['male'].astype(str).str.replace(',', ''), errors='coerce')
    normalized_df['female'] = pd.to_numeric(normalized_df['female'].astype(str).str.replace(',', ''), errors='coerce')

    # Group by normalized_score  sum male ,female
    grouped_df = normalized_df.groupby('normalized_score', as_index=False).agg({
        'male': 'sum',
        'female': 'sum'
    })

    # 모든 normalized_score를 갖는 새로운 dataframe 생성
    all_scores = pd.DataFrame({'normalized_score': range(global_min_score, global_max_score + 1)})
    # 비어있는 row를 다 0으로 대체하는 작업
    complete_df = pd.merge(all_scores, grouped_df, on='normalized_score', how='left').fillna(0)
    # int로 변환
    complete_df['male'] = complete_df['male'].astype(float).round().astype(int)
    complete_df['female'] = complete_df['female'].astype(float).round().astype(int)

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

    # training and testing sets로 나눠주기
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

# PyTorch tensors로 데이터 변환
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# PyTorch Dataset, DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# neural network model 정의
# 두개의 은닉층,relu 활성화 함수 사용
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(X_train.shape[1], 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, y_train.shape[1])

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

# 모델 생성, MSE loss function 생성
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# neural network model 훈련
def train_model(model, criterion, optimizer, train_loader, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

train_model(model, criterion, optimizer, train_loader, num_epochs=100)

# neural network model 평가
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()
    mse_nn = mean_squared_error(y_test, y_pred)
    r2_nn = r2_score(y_test, y_pred)

print(f"Neural Network - Mean Squared Error: {mse_nn}, R^2 Score: {r2_nn}")

# 모델 받아서 평가하는 함수 생성
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# 선형 회귀, KNN,랜덤포레스트 모델 생성
models = {
    "Linear Regression": LinearRegression(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Random Forest": RandomForestRegressor()
}

# models 속의 모델로 평가
for model_name, model in models.items():
    mse, r2 = train_and_evaluate(model, X_train, y_train, X_test, y_test)
    print(f"{model_name} - Mean Squared Error: {mse}, R^2 Score: {r2}")
