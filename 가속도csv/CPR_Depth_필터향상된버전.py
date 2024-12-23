import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import cumulative_trapezoid
from collections import deque

class CPRDepthSimulator:
    def __init__(self, sample_rate=100, window_size=500, csv_file="song_accelerometer_data.csv"):
        self.tracker = VerticalDisplacementTracker(sample_rate)
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.window_size = window_size
        
        # 데이터 버퍼
        self.times = deque(maxlen=window_size)
        self.depths = deque(maxlen=window_size)
        
        # CSV 파일에서 데이터 읽기
        self.acc_data = pd.read_csv(csv_file)['Accelerometer Value'].values
        self.data_index = 0  # 현재 읽고 있는 데이터 인덱스

    def get_next_acceleration(self):
        """CSV 데이터에서 다음 가속도 값 가져오기"""
        if self.data_index < len(self.acc_data):
            acc = self.acc_data[self.data_index]
            self.data_index += 1
            return acc
        else:
            return 0  # 데이터가 끝났다면 0 반환
    
    def update_plot(self, frame):
        """실시간 그래프 업데이트"""
        # CSV 파일에서 가속도 데이터 가져오기
        acc_z = self.get_next_acceleration()
        
        # 모든 필터가 적용된 최종 깊이
        depth = self.tracker.calculate_displacement(acc_z)
        
        # 시간 및 깊이 데이터 저장
        current_time = self.data_index * self.dt
        self.times.append(current_time)
        self.depths.append(depth * 100)  # Convert to cm
        
        # 그래프 업데이트
        self.ax1.clear()
        
        # 필터 적용 후 최종 깊이 그래프
        self.ax1.plot(list(self.times), list(self.depths), 'b-', linewidth=2, label='Filtered Depth')
        self.ax1.set_ylim(-10, 10)
        self.ax1.axhline(y=-5, color='r', linestyle='--', linewidth=2, label='Target Depth (5cm)')
        self.ax1.set_ylabel('Depth (cm)')
        self.ax1.set_title('Filtered CPR Compression Depth Monitoring')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # 최근 5초의 데이터만 표시
        min_time = max(0, current_time - 5)
        self.ax1.set_xlim(min_time, current_time)

class VerticalDisplacementTracker:
    def __init__(self, sample_rate=100, moving_avg_window_size=5):
        """
        수직 변위 추적기 초기화
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        # 칼만 필터 변수 초기화
        self.Q = np.array([[0.1, 0], [0, 0.1]])
        self.R = np.array([[0.001, 0], [0, 0.001]])
        self.P = np.eye(2)
        self.state = np.zeros(2)  # [displacement, velocity]
        
        # 필터 파라미터
        self.alpha = 0.6
        self.filtered_acc = 0
        self.acc_buffer = deque(maxlen=moving_avg_window_size)  # 이동 평균 필터용 버퍼
        self.last_displacement = 0
        
        # CPR 특성 파라미터
        self.max_displacement = 0.06
        self.min_displacement = 0.0
        
        # 중력 보정용 변수
        self.gravity_calibration = 0
        self.calibration_samples = 0
        self.is_calibrated = False
        self.acceleration_scale = 1.2
        self.integration_scale = 1.0

    def calibrate_gravity(self, acc_z):
        """중력 보정값 계산"""
        if not self.is_calibrated:
            self.calibration_samples += 1
            self.gravity_calibration += acc_z
            if self.calibration_samples >= 5:
                self.gravity_calibration /= self.calibration_samples
                self.is_calibrated = True
    
    def weighted_low_pass_filter(self, raw_acc, weight=0.6):
        """가중치가 적용된 저주파 통과 필터"""
        self.filtered_acc = weight * raw_acc + (1 - weight) * self.filtered_acc
        return self.filtered_acc

    def calculate_displacement(self, acc_z):
        """모든 필터를 거친 최종 깊이 계산"""
        try:
            # 1. 중력 보정
            if not self.is_calibrated:
                self.calibrate_gravity(acc_z)
                return 0.0
            
            # 2. 중력 성분 제거 및 저주파 필터 적용
            linear_acc = (acc_z - self.gravity_calibration) * self.acceleration_scale
            filtered_acc = self.weighted_low_pass_filter(linear_acc, weight=self.alpha)
            self.acc_buffer.append(filtered_acc)
            
            if len(self.acc_buffer) < 2:
                return 0.0

            # 3. 적분 필터를 통해 속도 계산
            acc_array = np.array(list(self.acc_buffer))
            velocity = cumulative_trapezoid(acc_array, dx=self.dt, initial=0)[-1]
            velocity *= self.integration_scale
            
            # 4. 이동 평균 필터 적용
            moving_avg_velocity = np.mean(list(self.acc_buffer))
            
            # 5. 칼만 필터 적용
            measurement = np.array([moving_avg_velocity, filtered_acc])
            displacement = self.kalman_filter(measurement)
            
            # 6. 변위 범위 제한
            displacement = np.clip(displacement, -self.max_displacement, self.min_displacement)
            
            # 7. 순간적 변화 제한
            if abs(displacement - self.last_displacement) > 0.05:
                displacement = self.last_displacement + np.sign(displacement - self.last_displacement) * 0.05
            
            self.last_displacement = displacement
            return displacement
            
        except Exception as e:
            print(f"Error in displacement calculation: {e}")
            return self.last_displacement
    
    def kalman_filter(self, measurement):
        """칼만 필터 적용"""
        F = np.array([[1, self.dt], [0, 1]])
        
        # 예측 단계
        predicted_state = F @ self.state
        predicted_P = F @ self.P @ F.T + self.Q
        
        # 업데이트 단계
        H = np.eye(2)
        S = H @ predicted_P @ H.T + self.R
        K = predicted_P @ H.T @ np.linalg.inv(S)
        
        # 상태 업데이트
        y = measurement - H @ predicted_state
        self.state = predicted_state + K @ y
        self.P = (np.eye(2) - K @ H) @ predicted_P
        
        return self.state[0]

def main():
    # 시뮬레이터 생성
    simulator = CPRDepthSimulator()
    
    # 그래프 초기화
    plt.style.use('default')
    fig, simulator.ax1 = plt.subplots(1, 1, figsize=(12, 6))
    fig.tight_layout(pad=3.0)
    
    # 배경색 설정
    fig.patch.set_facecolor('white')
    simulator.ax1.set_facecolor('white')
    
    # 애니메이션 시작
    ani = FuncAnimation(fig, simulator.update_plot, interval=20, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()
