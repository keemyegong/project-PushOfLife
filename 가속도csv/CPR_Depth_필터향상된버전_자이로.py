import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import cumulative_trapezoid  # cumtrapz 대신 사용
import time
from collections import deque

class CPRDepthSimulator:
    def __init__(self, sample_rate=100, window_size=500):
        self.tracker = VerticalDisplacementTracker(sample_rate)
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        # 데이터 버퍼
        self.window_size = window_size
        self.times = deque(maxlen=window_size)
        self.depths = deque(maxlen=window_size)
        self.accelerations = deque(maxlen=window_size)
        
        # 시뮬레이션 파라미터
        self.compression_freq = 100  # 분당 압박 횟수
        self.target_depth = 0.05  # 목표 깊이 (5cm)
        
        # 초기 가속도 기록을 위한 버퍼
        self.acc_buffer = deque(maxlen=10)
            
    def generate_simulated_data(self):
        """CPR 동작을 시뮬레이션하여 가속도와 자이로 데이터 생성"""
        t = time.time()
        
        # 기본 CPR 주기 생성
        freq = self.compression_freq / 60
        
        # 가속도 데이터 생성
        acceleration = self.target_depth * (2 * np.pi * freq)**2 * np.sin(2 * np.pi * freq * t)
        acc_noise = np.random.normal(0, 0.5)
        acceleration += acc_noise
        
        # 자이로 데이터 생성 (미세한 회전 움직임 시뮬레이션)
        gyro_x = np.sin(2 * np.pi * freq * t) * 0.1  # 작은 X축 회전
        gyro_y = np.cos(2 * np.pi * freq * t) * 0.1  # 작은 Y축 회전
        gyro_z = np.random.normal(0, 0.05)           # 랜덤 Z축 회전
        
        return acceleration, np.array([gyro_x, gyro_y, gyro_z])

    def update_plot(self, frame):
        """실시간 그래프 업데이트"""
        # 시뮬레이션 데이터 생성
        acc_z, gyro_data = self.generate_simulated_data()
    
        # 깊이 계산
        depth = self.tracker.calculate_displacement(acc_z, gyro_data)
        
        # 데이터 저장 (깊이를 cm로 변환: m * 100)
        current_time = time.time()
        self.times.append(current_time)
        self.depths.append(depth * 100)  # Convert to cm
        self.accelerations.append(acc_z)
        
        # 그래프 업데이트
        self.ax1.clear()
        self.ax2.clear()
        
        # 깊이 그래프
        self.ax1.plot(list(self.times), list(self.depths), 'b-', linewidth=2, 
                    label='Compression Depth')
        self.ax1.set_ylim(-10, 10)  # Range: -10cm to 5cm
        self.ax1.axhline(y=-5, color='r', linestyle='--', linewidth=2, 
                        label='Target Depth (5cm)')
        self.ax1.set_ylabel('Depth (cm)')
        self.ax1.set_title('CPR Compression Depth Monitoring')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # 가속도 그래프 (y축 범위 조정)
        self.ax2.plot(list(self.times), list(self.accelerations), 'g-', 
                    linewidth=2, label='Acceleration')
        self.ax2.set_ylim(-15, 15)  # 가속도 범위 조정
        self.ax2.set_ylabel('Acceleration (m/s²)')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        
        # 최근 5초의 데이터만 표시
        if len(self.times) > 0:
            min_time = current_time - 5
            self.ax1.set_xlim(min_time, current_time)
            self.ax2.set_xlim(min_time, current_time)

class VerticalDisplacementTracker:
    def __init__(self, sample_rate=100):
        """
        수직 변위 추적기 초기화
        Args:
            sample_rate (int): 샘플링 레이트 (Hz), 기본값 100Hz
        """
        # 기본 파라미터 설정
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        # 칼만 필터 파라미터 완화
        self.Q = np.array([[0.15, 0],    # 프로세스 노이즈 증가
                          [0, 0.15]])     
        self.R = np.array([[0.05, 0],    # 측정 노이즈 감소
                          [0, 0.05]])     
        self.P = np.eye(2)              
        self.state = np.zeros(2)        
        
        # 필터링 파라미터 크게 완화
        self.alpha = 0.6                 # 필터 계수 크게 증가 (더 빠른 반응)
        self.filtered_acc = 0
        
        # 버퍼 설정
        self.acc_buffer = deque(maxlen=3)    # 버퍼 크기 감소
        self.last_displacement = 0
        
        # CPR 특성 파라미터 조정
        self.max_displacement = 0.06      # 최대 변위 6cm로 제한
        self.min_displacement = 0.0       # 최소 변위
        
        # 중력 보정
        self.gravity = 9.81
        self.gravity_calibration = 0
        self.calibration_samples = 0
        self.is_calibrated = False
        
        # 스케일링 팩터 조정
        self.acceleration_scale = 1.2     # 가속도 스케일 팩터 감소
        self.integration_scale = 1.0      # 적분 스케일 팩터 감소
        
    def calibrate_gravity(self, acc_z):
        """중력 보정값 계산"""
        if not self.is_calibrated:
            self.calibration_samples += 1
            self.gravity_calibration += acc_z
            if self.calibration_samples >= 5:  # 보정 샘플 수 감소
                self.gravity_calibration /= self.calibration_samples
                self.is_calibrated = True
    
    def low_pass_filter(self, raw_acc):
        """완화된 저주파 통과 필터"""
        self.filtered_acc = self.alpha * raw_acc + (1 - self.alpha) * self.filtered_acc
        return self.filtered_acc

    def calculate_displacement(self, acc_z, gyro_data):
        """수직 이동거리 계산"""
        try:
            # 1. 중력 보정
            if not self.is_calibrated:
                self.calibrate_gravity(acc_z)
                return 0.0
                
            # 2. 중력 성분 제거 및 스케일링
            linear_acc = (acc_z - self.gravity_calibration) * self.acceleration_scale
            
            # 3. 간단한 필터링
            filtered_acc = self.low_pass_filter(linear_acc)
            self.acc_buffer.append(filtered_acc)
            
            if len(self.acc_buffer) < 2:
                return 0.0

            # 4. 속도 계산
            acc_array = np.array(list(self.acc_buffer))
            velocity = cumulative_trapezoid(acc_array, dx=self.dt, initial=0)[-1]
            velocity *= self.integration_scale
            
            # 5. 칼만 필터
            measurement = np.array([velocity, filtered_acc])
            displacement = self.kalman_filter(measurement)
            
            # 6. 기본적인 범위 제한만 적용
            displacement = np.clip(displacement, -self.max_displacement, self.min_displacement)
            
            # 7. 매우 완화된 변화 제한
            if abs(displacement - self.last_displacement) > 0.05:  # 5cm까지 순간 변화 허용
                displacement = self.last_displacement + np.sign(displacement - self.last_displacement) * 0.05
            
            self.last_displacement = displacement
            return displacement
            
        except Exception as e:
            print(f"Error in displacement calculation: {e}")
            return self.last_displacement
    
    def kalman_filter(self, measurement):
        """단순화된 칼만 필터"""
        F = np.array([[1, self.dt],
                     [0, 1]])
        
        # 예측 단계
        predicted_state = F @ self.state
        predicted_P = F @ self.P @ F.T + self.Q
        
        # 업데이트 단계
        H = np.eye(2)
        S = H @ predicted_P @ H.T + self.R
        K = predicted_P @ H.T @ np.linalg.inv(S)
        
        # 상태 업데이트 (이상치 감지 제거)
        y = measurement - H @ predicted_state
        self.state = predicted_state + K @ y
        self.P = (np.eye(2) - K @ H) @ predicted_P
        
        return self.state[0]

def main():
    # 시뮬레이터 생성
    simulator = CPRDepthSimulator()
    
    # 그래프 초기화
    plt.style.use('default')
    fig, (simulator.ax1, simulator.ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.tight_layout(pad=3.0)
    
    # 배경색 설정
    fig.patch.set_facecolor('white')
    simulator.ax1.set_facecolor('white')
    simulator.ax2.set_facecolor('white')
    
    # 애니메이션 시작 (경고 메시지 제거를 위한 수정)
    ani = FuncAnimation(fig, simulator.update_plot, 
                       interval=20,
                       cache_frame_data=False)  # 캐시 비활성화
    plt.show()

if __name__ == "__main__":
    main()