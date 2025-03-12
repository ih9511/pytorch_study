# 손실 함수 & 최적화 알고리즘

## 1. 손실 함수(Loss Function)란?

손실 함수는 모델이 얼마나 잘못 예측했는지를 측정하는 함수입니다. 모델이 학습하는 목적은 손실(loss)을 최소화하는 방향으로 가중치를 업데이트하는 것입니다.

### 대표적인 손실 함수

| 손실 함수 | 사용 예시 | PyTorch 함수 |
| --- | --- | --- |
| MSE (Mean Squared Error) | 회귀 모델 (Regression) | `nn.MSELoss()` |
| MAE (Mean Absolute Error) | 회귀 모델 (노이즈 존재 시) | `nn.L1Loss()` |
| BCE (Binary Cross Entropy) | 이진 분류 (Binary Classification) | `nn.BCELoss()` |
| CrossEntropyLoss | 다중 클래스 분류 (Multi-Class) | `nn.CrossEntopyLoss()` |
- 회귀 모델 → MSE / MAE
- 이진 분류 → BCE
- 다중 클래스 분류 → CrossEntropyLoss

## 2. 손실 함수 예제

이제 PyTorch에서 손실 함수를 직접 사용해보겠습니다.

### MSE 손실 (회귀 모델)

```python
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
```

- 예측값과 실제값의 차이가 클수록 loss가 커짐

### CrossEntropyLoss (다중 클래스 분류)

CrossEntropyLoss는 Softmax + 로그 손실(MLLLoss)를 포함한 함수입니다. y_true는 one-hot encoding이 아니라 정수값으로 줘야 합니다.

```python
# 3개 클래스에 대한 예측값 (Logits) -> 아직 Softmax 적용 전
y_pred = torch.tensor([[2.0, 1.0, 0.1]])

# 정답 라벨 (클래스 0, 1, 2 중 하나)
y_true = torch.tensor([0])

# CrossEntropy 손실 함수
loss_fn = nn.CrossEntropyLoss()

# 손실 값 계산
loss = loss_fn(y_pred, y_true)
print(f"CrossEntropy 손실 값: {loss.item()}")
```

- `nn.CrossEntropyLoss()` 는 내부적으로 `Softmax`를 포함.
- 정답 라벨(`y_true`)은 정수형 인덱스 (0, 1, 2, …)로 제공.
- `y_pred` 는 소프트맥스 이전의 logits 값을 넣어야 함.

## 3. 최적화 알고리즘(Optimizer)이란?

최적화 알고리즘은 손실 함수를 최소화하기 위해 가중치를 업데이트하는 방법입니다. 손실 함수만 정의하면 모델이 자동으로 학습하는 게 아니라, 어떻게 가중치를 조정할지를 결정해야 합니다. 이 역할을 하는 게 바로 Optimizer 입니다.

### 대표적인 최적화 알고리즘

| 최적화 알고리즘 | 특징 | PyTorch 함수 |
| --- | --- | --- |
| SGD (확률적 경사 하강법) | 기본적인 경사 하강법 | `torch.optim.SGD()` |
| Adam | 가장 많이 사용됨, 빠르고 안정적 | `torch.optim.Adam()` |
| RMSprop | RNN에서 자주 사용됨 | `torch.optim.RMSprop()` |
| Adagrad | 희소 데이터(Sparse Data)에 적합 | `torch.optim.Adagrad()` |
- 일반적으로는 Adam이 가장 많이 사용됨.
- SGD는 가장 기본적인 방식이며, 간단한 경우에 사용됨.

## 4. 최적화 알고리즘 적용 예제

이제 실제로 최적화 알고리즘을 적용해서 가중치를 업데이트하는 과정을 보겠습니다.

```python
import torch.optim as optim

# 간단한 선형 모델
model = nn.Linear(2, 1) # 입력 2개, 출력 1개

# 손실 함수 (MSE)
loss_fn = nn.MSELoss()

# 최적화 알고리즘 (Optimizer)
optimizer = optim.Adam(mode.parameters(), lr=0.01)

# 더미 데이터
x = torch.tensor([[1.0, 2.0]], requires_grad=True) # 입력값
y_true = torch.tensor([[3.0]]) # 실제 정답

# 모델 순전파
y_pred = model(x)

# 손실 계산
loss = loss_fn(y_pred, y_true)
print(f"초기 손실 값: {loss.item()}")

# 역전파(Backpropagation)
optimezer.zero_grad() # 기존의 그래디언트 초기화
loss.backward() # 손실 함수의 그래디언트 계산
optimizer.step() # 가중치 업데이트

# 업데이트 후 새로운 예측
y_pred_new = model(x)
loss_new = loss_fn(y_pred_new, y_true)
print(f"업데이트 후 손실 값: {loss_new.item()}")
```

- `optimizer.zero_grad()` → 기존 그래디언트 초기화 (중첩 방지)
- `loss.backward()` → 손실 함수에 대한 가중치의 그래디언트 계산
- `optimizer.step()` → 가중치 업데이트