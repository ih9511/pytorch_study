import torch
import torch.nn as nn


# 예측값과 실제값
y_pred = torch.tensor([2.5, 3.0, 4.5])
y_true = torch.tensor([3.0, 3.0, 5.0])

# MSE 손실 함수 정의
loss_fn = nn.MSELoss()

# 손실 값 계산
loss = loss_fn(y_pred, y_true)
print(f"MSE 손실 값: {loss.item()}")

# 3개 클래스에 대한 예측값 (Logits) -> 아직 Softmax 적용 전
y_pred = torch.tensor([[2.0, 1.0, 0.1]])

# 정답 라벨 (클래스 0, 1, 2 중 하나)
y_true = torch.tensor([0])

# CrossEntropy 손실 함수
loss_fn = nn.CrossEntropyLoss()

# 손실 값 계산
loss = loss_fn(y_pred, y_true)
print(f"CrossEntropy 손실 값: {loss.item()}")


import torch.optim as optim

# 간단한 선형 모델
model = nn.Linear(2, 1)

# 손실 함수
loss_fn = nn.MSELoss()

# 최적화 알고리즘
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 더미 데이터
x = torch.tensor([[1.0, 2.0]], requires_grad=True)
y_true = torch.tensor([[3.0]])

# 모델 순전파
y_pred = model(x)

# 손실 계산
loss = loss_fn(y_pred, y_true)
print(f"초기 손실 값: {loss.item()}")

# 역전파
optimizer.zero_grad() # optimizer 초기화
loss.backward() # 역전파
optimizer.step() # parameter 조정

# 업데이트 후 새로운 예측
y_pred_new = model(x)
loss_new = loss_fn(y_pred_new, y_true)
print(f"업데이트 후 손실 값: {loss_new.item()}")