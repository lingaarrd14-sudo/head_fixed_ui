import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

def generate_vr_legibility_data(n_samples=500, folder=parent_dir, filename="vr_legibility_data.csv"):
    np.random.seed(42)
    
    # 1. Feature 생성: 좌표(-1.0 ~ 1.0 정규화), 크기(0.01 ~ 0.15 비율)
    x_coords = np.random.uniform(-960, 960, n_samples)
    y_coords = np.random.uniform(-540, 540, n_samples)
    widths = np.random.uniform(5, 200, n_samples)
    heights = widths
    
    # 2. 중심으로부터의 거리 계산 (유클리디안 거리)
    distances = np.sqrt(x_coords**2 + y_coords**2)
    
    labels = []
    for d, w, h in zip(distances, widths, heights):
        area = w * h
        # 거리가 멀어질수록 가독성에 필요한 최소 면적(threshold)이 증가
        threshold = 0.002 + (d * 0.005) 
        
        if area > threshold:
            labels.append(1)
        else:
            labels.append(0)
            
    # 3. 데이터 현실성을 위한 노이즈 추가 (10% 라벨 반전)
    noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    for idx in noise_indices:
        labels[idx] = 1 - labels[idx]
        
    # 4. DataFrame 생성 및 CSV 저장
    df = pd.DataFrame({
        'width': np.round(widths, 4),
        'height': np.round(heights, 4),
        'x_coord': np.round(x_coords, 4),
        'y_coord': np.round(y_coords, 4),
        'label': labels
    })
    
    df.to_csv(os.path.join(parent_dir, filename), index=False)
    print(f"{n_samples}개의 데이터가 {filename}로 저장되었습니다.")
    return df

# 스크립트 실행
df_sample = generate_vr_legibility_data()