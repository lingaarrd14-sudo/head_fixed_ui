import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import os

os.makedirs(os.path.dirname(dir), exist_ok=True)

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리
# ---------------------------------------------------------
print("데이터를 로드하고 전처리를 시작합니다...")
df = pd.read_csv('vr_legibility_sample.csv')

# 다중공선성 방지: width와 height가 동일하므로 height 열 제거
if 'height' in df.columns:
    df = df.drop('height', axis=1)

# 특성(X)과 타겟(y) 분리
X = df.drop('label', axis=1)
y = df['label']

# 학습/테스트 데이터 분할 (클래스 불균형 유지를 위해 stratify=y 적용)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 2. 파이프라인 (스케일링 + 모델) 구축
# ---------------------------------------------------------
# SVM은 거리를 계산하므로 스케일링(StandardScaler)이 필수입니다.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(class_weight='balanced', random_state=42)) # 불균형 해소를 위해 balanced 적용
])

# ---------------------------------------------------------
# 3. 하이퍼파라미터 튜닝 (GridSearchCV) 설정
# ---------------------------------------------------------
# 탐색할 C, gamma, 커널 매개변수 지도를 만듭니다.
param_grid = {
    'svm__kernel': ['rbf'],                # 비선형 분류인 rbf 커널 고정
    'svm__C': [0.1, 1, 10, 100],           # 규제 강도 (마진의 엄격함)
    'svm__gamma': [0.01, 0.1, 1, 10, 'scale']  # 데이터 영향 범위
}

# 교차 검증 객체 생성 
# 주의: label 0과 1을 모두 잘 맞추도록 평균을 내는 'f1_macro'를 사용합니다. 
# (데이터의 90%가 1이므로, 일반 f1이나 accuracy를 쓰면 0을 아예 무시하는 모델이 될 수 있습니다)
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5,               # 5-Fold 교차 검증
    scoring='f1_macro', # 성능 평가 기준
    n_jobs=-1           # 가용한 모든 CPU 코어 사용 (속도 향상)
)

# ---------------------------------------------------------
# 4. 모델 훈련 및 최적 파라미터 탐색
# ---------------------------------------------------------
print("최적의 하이퍼파라미터를 찾는 중입니다. (시간이 조금 걸릴 수 있습니다)...")
grid_search.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. 최종 평가 및 출력
# ---------------------------------------------------------
# GridSearch가 찾은 가장 성능이 좋은 모델로 테스트 데이터를 예측합니다.
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n=========================================")
print("🎯 [GridSearchCV 탐색 결과]")
print("=========================================")
print(f"가장 좋은 파라미터 조합: {grid_search.best_params_}")
print(f"최고 교차 검증 점수 (F1 Macro): {grid_search.best_score_:.4f}\n")

print("=========================================")
print("📊 [테스트 데이터 최종 평가 결과]")
print("=========================================")
print("혼동 행렬 (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))
print("\n분류 리포트 (Classification Report):")
print(classification_report(y_test, y_pred))