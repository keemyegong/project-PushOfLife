import csv

# txt 파일 경로와 생성할 csv 파일 경로를 지정
txt_file_path = "강사님 데이터.txt"
csv_file_path = "강사님_csv.csv"

# txt 파일을 utf-8로 읽고 필요한 정보만 추출하여 csv로 변환
with open(txt_file_path, "r", encoding="utf-8") as txt_file, open(csv_file_path, "w", newline='', encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    
    # CSV 파일에 헤더 작성
    writer.writerow(["Timestamp", "Accelerometer Value"])
    
    for line in txt_file:
        # 'Accelerometer'가 포함된 줄에서 필요한 정보만 추출
        if "Accelerometer" in line:
            parts = line.split()  # 공백 기준으로 분할
            timestamp = parts[0] + " " + parts[1]  # 시간 정보
            accelerometer_value = parts[-1]  # 마지막 값이 가속도 값
            
            # 추출한 값을 CSV에 작성
            writer.writerow([timestamp, accelerometer_value])

print(f"CSV 파일이 생성되었습니다: {csv_file_path}")