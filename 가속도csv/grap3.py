import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

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

# 탐색할 값의 개수를 설정
num_points = 50

# Q와 R의 값 범위를 세밀하게 설정
Q_values = [q * np.eye(2) for q in np.linspace(0.001, 0.1, num_points)]
R_values = [r * np.eye(2) for r in np.linspace(0.001, 0.1, num_points)]

# 에러 행렬 초기화
error_matrix = np.zeros((num_points, num_points))

# Grid Search
for qi, Q in enumerate(Q_values):
    for ri, R in enumerate(R_values):
        total_error = 0  # 현재 Q, R 조합에 대한 전체 오차
        x = np.zeros((2, 1))  # 초기 상태
        P = np.eye(2)         # 초기 공분산

        # 칼만 필터 계산 반복문
        for u in acc_data:
            # 예측 단계
            x_pred = F @ x + B * u
            P_pred = F @ P @ F.T + Q

            # 칼만 이득 계산
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)

            # 측정 오차 계산
            measurement_error = np.linalg.norm(u - x_pred[0, 0])
            total_error += measurement_error

            # 상태 업데이트
            x = x_pred + K @ (u - H @ x_pred)
            P = (np.eye(2) - K @ H) @ P_pred

        # 총 오차 저장
        error_matrix[qi, ri] = total_error

# 최적의 Q와 R 찾기
min_error_idx = np.unravel_index(np.argmin(error_matrix, axis=None), error_matrix.shape)
best_Q = Q_values[min_error_idx[0]]
best_R = R_values[min_error_idx[1]]

print("최적 Q:\n", best_Q)
print("최적 R:\n", best_R)

# Q와 R 범위에 따라 에러 값 시각화
Q_mesh, R_mesh = np.meshgrid(np.linspace(0.001, 0.1, num_points), np.linspace(0.001, 0.1, num_points))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Q_mesh, R_mesh, error_matrix.T, cmap='viridis')
ax.set_xlabel('Q Value')
ax.set_ylabel('R Value')
ax.set_zlabel('Total Error')
ax.set_title('Kalman Filter Total Error for Different Q and R Values')
fig.colorbar(surf)

# X, Y, Z 축 범위 설정
ax.set_xlim(0, 0.1)
ax.set_ylim(0, 0.1)
ax.set_zlim(np.min(error_matrix) - 0.1, np.max(error_matrix) + 0.1)

plt.show()
