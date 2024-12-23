import csv

def txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r', encoding='utf-8') as infile, open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Timestamp', 'Depth'])  # CSV 헤더 작성

        for line in infile:
            if "깊이" in line:
                parts = line.split()  # 공백으로 분할
                timestamp = parts[1]  # 타임스탬프 추출
                depth = parts[-2]  # 깊이 값 추출
                writer.writerow([timestamp, depth])

txt_to_csv('distance_text.txt', 'depth_data.csv')
