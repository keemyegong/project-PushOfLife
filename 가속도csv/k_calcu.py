import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# CSV 파일 읽기
file_path = 'monk_accelerometer_data.csv'
data = pd.read_csv(file_path)

# 타임스탬프와 가속도 데이터 추출
time_data = pd.to_datetime(data.iloc[:, 0])  # 첫 번째 열이 타임스탬프라고 가정
acc_data = data.iloc[:, 1].values            # 두 번째 열이 가속도 데이터라고 가정

# 타임스탬프 간 시간 차이를 계산하여 dt 계산
time_diffs = (time_data.diff().dt.total_seconds()).dropna().values
dt = np.mean(time_diffs)

# 상태 전이 행렬 F와 제어 입력 행렬 B 정의
F = np.array([[1, dt], [0, 1]])
B = np.array([[0.5 * dt**2], [dt]])
H = np.array([[1, 0], [0, 1]])

# 최적의 Q, R 값을 여기서 지정
Q = np.array([[0.1, 0], [0, 0.1]])  # 최적 Q (고정)
R = np.array([[0.001, 0], [0, 0.001]])  # 최적 R (고정)

# 초기 상태와 공분산 행렬
x = np.zeros((2, 1))  # 초기 상태
P = np.eye(2)         # 초기 공분산

# 실시간 시각화를 위한 빈 리스트
K_values = []  # 이득 값 기록

# 그래프 초기화
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', label='Kalman Gain K[0,0]')  # K[0,0] 이득 값
ax.set_xlim(0, 50)  # x축 범위를 0에서 50까지만 표시
ax.set_ylim(0, 1)  # y축 범위: 이득 값 K의 예상 범위
ax.set_title("Kalman Gain (K[0,0]) over Time")
ax.set_xlabel("Time Step")
ax.set_ylabel("Kalman Gain K[0,0]")
ax.legend()

# 애니메이션 업데이트 함수
def update(frame):
    global x, P

    # 현재 가속도 입력
    u = acc_data[frame]

    # 예측 단계
    x_pred = F @ x + B * u
    P_pred = F @ P @ F.T + Q

    # 칼만 이득 계산
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    K_values.append(K[0, 0])  # K[0,0] 값 저장

    # 측정 업데이트
    x = x_pred + K @ (u - H @ x_pred)
    P = (np.eye(2) - K @ H) @ P_pred

    # y축 범위 동적 설정
    ax.set_ylim(min(K_values) - 0.001, max(K_values) + 0.001)

    # x축 범위에 맞춰 최신 50개 데이터만 표시
    line.set_data(range(len(K_values[-50:])), K_values[-50:])

    return line,

# 애니메이션 설정
ani = FuncAnimation(fig, update, frames=len(acc_data), blit=True, interval=20)

plt.show()
