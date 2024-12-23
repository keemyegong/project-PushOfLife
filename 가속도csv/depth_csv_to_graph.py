import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from datetime import datetime
from collections import deque

class RealTimeDepthPlotter:
    def __init__(self, csv_file, window_size=500):
        self.csv_file = csv_file
        self.window_size = window_size
        
        # 데이터 버퍼
        self.times = deque(maxlen=window_size)
        self.depths = deque(maxlen=window_size)
        
        # 시작 시간 기준으로 타임스탬프를 상대 시간으로 변환하기 위해 시작 시간을 저장
        self.start_time = None
        self.last_read_line = 0  # 마지막으로 읽은 라인 번호

    def read_new_data(self):
        """CSV 파일에서 새로운 데이터를 읽어옴"""
        try:
            with open(self.csv_file, 'r') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                
                # 새로운 행만 처리
                new_rows = rows[self.last_read_line:]
                self.last_read_line = len(rows)  # 마지막 읽은 라인 업데이트
                
                for row in new_rows:
                    # 타임스탬프를 datetime 형식으로 변환
                    timestamp = datetime.strptime(row['Timestamp'], '%H:%M:%S.%f')
                    if self.start_time is None:
                        self.start_time = timestamp
                    # 상대 시간(초) 계산
                    elapsed_time = (timestamp - self.start_time).total_seconds()
                    
                    # 깊이 데이터를 가져옴
                    depth = float(row['Depth'])
                    
                    # 데이터 버퍼에 추가
                    self.times.append(elapsed_time)
                    self.depths.append(depth)
        except Exception as e:
            print(f"Error reading CSV file: {e}")

    def update_plot(self, frame):
        """실시간 그래프 업데이트"""
        # CSV에서 새로운 데이터 읽기
        self.read_new_data()
        
        # 그래프 업데이트
        self.ax.clear()
        
        # 깊이 그래프
        self.ax.plot(list(self.times), list(self.depths), 'b-', linewidth=2, 
                     label='Compression Depth (cm)')
        self.ax.set_ylim(-10, 10)  # 깊이 범위 -10cm ~ 10cm
        self.ax.axhline(y=-5, color='r', linestyle='--', linewidth=2, 
                        label='Target Depth (5cm)')
        self.ax.set_ylabel('Depth (cm)')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_title('Real-Time Compression Depth Monitoring')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # 최근 5초의 데이터만 표시
        if len(self.times) > 0:
            current_time = self.times[-1]
            min_time = current_time - 5
            self.ax.set_xlim(min_time, current_time)

def main():
    # 실시간 그래프 생성
    plotter = RealTimeDepthPlotter('depth_data.csv')
    
    # 그래프 초기화
    plt.style.use('default')
    fig, plotter.ax = plt.subplots(figsize=(12, 6))
    fig.tight_layout(pad=3.0)
    
    # 배경색 설정
    fig.patch.set_facecolor('white')
    plotter.ax.set_facecolor('white')
    
    # 애니메이션 시작
    ani = FuncAnimation(fig, plotter.update_plot, 
                        interval=1000, cache_frame_data=False)  # 1초마다 업데이트
    plt.show()

if __name__ == "__main__":
    main()
