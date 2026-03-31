import pandas as pd
import itertools
import os

# 범위 설정 (마지막 숫자를 포함하기 위해 +1)
x_range = range(-55, 56)
y_range = range(-53, 44)

# 모든 조합 생성
combinations = list(itertools.product(x_range, y_range))

# 데이터프레임 변환 및 저장
df = pd.DataFrame(combinations, columns=['x_theta', 'y_theta'])
df.to_csv(os.path.join('..', 'coordinates.csv'), index=False)

print(f"총 {len(df)}개의 행을 가진 CSV 파일이 생성되었습니다.")
