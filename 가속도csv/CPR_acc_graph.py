import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

class CPRDepthSimulator:
    def __init__(self, sample_rate=100, window_size=500, csv_file="강사님_csv.csv"):
        self.tracker = VerticalDisplacementTracker(sample_rate)
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.window_size = window_size
        
        # 데이터 버퍼
        self.times = deque(maxlen=window_size)
        self.accelerations = deque(maxlen=window_size)
        self.filtered_accelerations = deque(maxlen=window_size)  # 필터링된 가속도
        
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
        """실시간 가속도 그래프 업데이트"""
        # CSV 파일에서 가속도 데이터 가져오기
        acc_z = self.get_next_acceleration()
        gyro_data = np.array([0, 0, 0])  # 자이로 데이터를 사용하지 않으므로 0으로 설정
        
        # 가중 이동 평균 필터 적용된 가속도 계산
        filtered_acc = self.tracker.calculate_weighted_filtered_acc(acc_z)
        
        # 데이터 저장
        current_time = self.data_index * self.dt
        self.times.append(current_time)
        self.accelerations.append(acc_z)
        self.filtered_accelerations.append(filtered_acc)
        
        # 가속도 그래프 업데이트 (원본 vs 필터링된 가속도)
        self.ax.clear()
        self.ax.plot(list(self.times), list(self.accelerations), 'g-', linewidth=1.5, label='Original Acceleration')
        # self.ax.plot(list(self.times), list(self.filtered_accelerations), 'b-', linewidth=1.5, label='Filtered Acceleration')
        self.ax.set_ylim(-15, 50)
        self.ax.set_ylabel('Acceleration (m/s²)')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_title('Acceleration Comparison (Original vs Weighted Filtered)')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # 최근 5초의 데이터만 표시
        min_time = max(0, current_time - 5)
        self.ax.set_xlim(min_time, current_time)

        # 3초에 도달했을 때 그래프 저장
        if current_time >= 3.0 and current_time < 3.0 + self.dt:
            plt.savefig('no_filter_graph_at_3_seconds.png')
            print("Graph saved at 3 seconds")


class VerticalDisplacementTracker:
    def __init__(self, sample_rate=100, moving_avg_window_size=5):
        """
        수직 변위 추적기 초기화
        Args:
            sample_rate (int): 샘플링 레이트 (Hz), 기본값 100Hz
            moving_avg_window_size (int): 이동 평균 필터 창 크기, 기본값 5
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.alpha = 0.6  # 저주파 필터용 계수
        self.filtered_acc = 0
        self.acc_buffer = deque(maxlen=4)  # 과거 4개의 가속도 값을 저장할 버퍼

    def calculate_weighted_filtered_acc(self, acc_z):
        """가속도 값에 가중 이동 평균 필터 적용"""
        # 버퍼에 현재 가속도 추가
        self.acc_buffer.append(acc_z)

        # 가중치를 줄 값이 적을 경우 버퍼 크기만큼 사용할 수 있도록 초기값 사용
        weights = np.array([0.4, 0.3, 0.2, 0.1])  # 가장 최근 값에 높은 가중치를 부여
        weights = weights[:len(self.acc_buffer)]  # 버퍼의 길이에 맞춰 가중치 크기 조정
        weights /= weights.sum()  # 가중치 합이 1이 되도록 정규화
        
        # 가속도 값과 가중치의 가중 평균 계산
        weighted_acc = np.dot(weights, list(self.acc_buffer))
        
        return weighted_acc

def main():
    simulator = CPRDepthSimulator()
    
    # 가속도 그래프 초기화
    plt.style.use('default')
    fig, simulator.ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.tight_layout(pad=3.0)
    
    # 배경색 설정
    fig.patch.set_facecolor('white')
    simulator.ax.set_facecolor('white')
    
    # 애니메이션 시작
    ani = FuncAnimation(fig, simulator.update_plot, interval=20, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()
