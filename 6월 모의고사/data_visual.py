import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo

# 데이터 불러오기 (인코딩 문제 해결을 위해 'euc-kr'로 시도)
file_path = '2022_6.csv'  # 여기에 실제 파일 경로를 넣어주세요.
data = pd.read_csv(file_path, encoding='euc-kr')

# 인터랙티브 그래프 생성
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data['standard_score'],
    y=data['male'],
    mode='lines+markers',
    name='Male'
))

fig.add_trace(go.Scatter(
    x=data['standard_score'],
    y=data['female'],
    mode='lines+markers',
    name='Female'
))

fig.update_layout(
    title='Count of Males and Females by Standard Score',
    xaxis_title='Standard Score',
    yaxis_title='Count',
    template='plotly'
)

# 그래프를 인터랙티브하게 표시
pyo.plot(fig)
