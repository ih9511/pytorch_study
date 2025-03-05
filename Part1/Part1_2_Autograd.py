import torch

# require_grad=True 설정된 텐서 (스칼라)
x = torch.tensor(2.0, requires_grad=True)
# x = torch.tensor(2.0)
y = x ** 2

print(f"y: {y}\n")

# y를 x에 대해 미분
y.backward()

# x에 대한 그래디언트 값 출력
print(f"x에 대한 그래디언트: {x.grad}\n")

# ===================================================================================

# 벡터 연산에서 backward() 호출
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

print(f"y: {y}\n")

# 벡터 연산이므로, backward() 실행 시 gradient 벡터를 제공해야 함
gradient = torch.tensor([1.0, 1.0, 1.0]) # dy/dx의 기준 벡터
y.backward(gradient)

print(f"x에 대한 그래디언트: {x.grad}\n")

# ===================================================================================

# torch.no_grad()로 미분 추적 방지
# 즉, 미분은 수행하고자 하지만 추적을 설정하지 않아 미분만 함.
x = torch.tensor(3.0, requires_grad=True)

# no_grad() 사용 -> 그래디언트 추적 x
with torch.no_grad():
    y = x ** 2 # 연산 수행 (하지만 requires_grad=False)
    
print(f"y: {y.requires_grad}\n")

# ===================================================================================

# detach()로 미분 추적 방지
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2

# detach() 사용 -> y가 더 이상 미분 추적하지 않음
y_detached = y.detach()

print(f"y: {y.requires_grad}\n")
print(f"y_detached: {y_detached.requires_grad}\n")

# ===================================================================================

# 1D Tensor에 대한 그래디언트 계산
# 데이터 샘플(x)과 실제 타겟(y)
x = torch.tensor(1.0, requires_grad=True)
y_true = torch.tensor(3.0)

# 가중치(학습해야 하는 값)
w = torch.tensor(2.0, requires_grad=True)

# 선형 모델: y_pred = w * x
y_pred = w * x

# 손실 함수: MSE Loss = (y_pred - y_true) ** 2
loss = (y_pred - y_true) ** 2

# 그래디언트 계산 d(loss)/d(w) = 2 * (w * x - y_true) * x
loss.backward()

# 가중치 w에 대한 그래디언트 출력
print(f"w에 대한 그래디언트: {w.grad}\n")

# ===================================================================================

# 2D Tensor에 대한 그래디언트 계산
# 2차원 입력 데이터(2개의 샘플, 3개의 특성)
X = torch.tensor(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]],
    requires_grad=True
)

# 실제 값(타겟 값) -> 2개 샘플에 대한 정답 레이블
y_true = torch.tensor([[10.0], [20.0]])

# 학습할 가중치 (3개의 특성을 가지므로 (3x1) 형태)
W = torch.tensor([[0.1], [0.2], [0.3]], requires_grad=True)

# 편향
b = torch.tensor(0.5, requires_grad=True)

# 선형 모델 계산 (y_pred = XW + b)
y_pred = X @ W + b
print(f"y_pred: \n{y_pred}\n")

# 손실 함수 계산(MSE)
loss = torch.mean((y_pred - y_true) ** 2)
print(f"loss: {loss.item()}\n")

# 역전파 수행
# 그래디언트 계산
loss.backward()

# 가중치 W의 그래디언트 출력
print(f"W에 대한 그래디언트: \n{W.grad}\n")

# 편향 b의 그래디언트 출력
print(f"b에 대한 그래디언트: {b.grad}\n")