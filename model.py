import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# 데이터셋 로드 (Kaggle에서 다운로드하여 로컬에 저장했다고 가정)
data = pd.read_csv("train.csv")

# 데이터 전처리
data = data.drop(["Cabin", "Ticket", "Name"], axis=1)  # 불필요한 열 제거
data = data.dropna()  # 결측값 제거
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})  # 성별 인코딩
data = pd.get_dummies(data, columns=["Embarked"], drop_first=True)  # One-hot 인코딩

# 특성과 레이블 분리
X = data.drop("Survived", axis=1)
y = data["Survived"]

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 초기 성능 평가
y_pred = model.predict(X_test_scaled)
initial_accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Accuracy: {initial_accuracy:.4f}")

# 교차 검증
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# 부트스트래핑
n_iterations = 1000
bootstrapped_scores = []

for _ in range(n_iterations):
    # 부트스트랩 샘플 생성
    X_resample, y_resample = resample(X_train_scaled, y_train)
    model.fit(X_resample, y_resample)
    y_pred = model.predict(X_test_scaled)
    score = accuracy_score(y_test, y_pred)
    bootstrapped_scores.append(score)

print(f"Bootstrap Accuracy: {pd.Series(bootstrapped_scores).mean():.4f}")
