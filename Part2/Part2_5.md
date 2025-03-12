# 모델 학습 및 평가

모델을 학습하려면 다음과 같은 단계를 거쳐야 합니다.

## 전체 학습 과정

1. 데이터 로드 (`Dataset` & `DataLoader` 사용)
2. 모델 정의(`nn.Module`)
3. 손실 함수 & 최적화 알고리즘 설정
4. 학습 루프 (Training Loop)
    - Forward (순전파)
    - 손실 (loss) 계산
    - Backward (역전파) & 가중치 업데이트
5. 모델 평가 (Evaluation)
    - 검증 데이터 (Validation Set) 사용
    - 정확도, 손실 값 비교

이제, 각 단계를 실제 코드로 구현해보겠습니다.

## 1. 간단한 데이터셋 생성

우선 더미 데이터를 사용해서 학습을 진행하겠습니다. 실제 데이터셋을 사용할 때는 `torchvision.datasets` 같이 데이터셋을 불러오는 기능들을 사용하면 됩니다.

```python
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

# 더미 데이터 생성(입력 2차원, 출력 1차원)
X = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]], dtype=torch.float32)
y = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]], dtype=torch.float32)

# Dataset & DataLoader 생성
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

## 2. 간단한 신경망 모델 정의

이제 PyTorch `nn.Module`을 상속받아 선형 회귀 모델을 만들어보겠습니다.

```python
class SimpleModel(nn.Module):
	def __init__(self, input_size, output_size):
		super(SimpleModel, self).__init__()
		self.fc = nn.Linear(input_size, output_size) # 선형 계층
		
	def forward(self, x):
		return self.fc(x) # 순전파
		
# 모델 생성
model = SimpleModel(input_size=2, output_size=1)
```

## 3. 손실 함수 & 최적화 알고리즘 설정

```python
# MSE 손실 함수 사용 (regression task)
loss_fn = nn.MSELoss()

# Adam 최적화 알고리즘 사용
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

## 4. 모델 학습 (Training Loop) & 학습 과정 시각화

이제 training loop를 만들어 모델을 학습해보겠습니다.

```python
import matplotlib.pyplot as plt

loss_history = []

# 학습 반복 횟수 (Epochs)
num_epochs = 100

# 학습 루프
for epoch in range(num_epochs):
	for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
		# 순전파
		# 이렇게 바로 x_batch를 넣을 수 있는 이유는 __call__ 특수 메서드 덕분에 가능함
		y_pred = model(x_batch)
		
		# 손실 계산
		loss = loss_fn(y_pred, y_batch)
		
		# 역전파
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
	loss_history.append(loss.item())
		
	if (epoch + 1) % 10 == 0:
		print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item}")
		
# 손실 값 그래프 출력
plt.plot(range(num_epochs), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.show()
```

## 5. 모델 평가 (Evaluation)

모델 학습이 끝났다면, 새로운 데이터에 대해 예측을 해보겠습니다.

```python
# 평가 모드 설정
model.eval()

# 테스트 데이터
test_X = torch.tensor([[6.0, 7.0], [7.0, 8.0]], dtype=torch.float32)

# 예측 수행
with torch.no_grad():
	test_pred = model(test_X)
	
print(f"테스트 데이터 예측값:\n{test_pred}")
```

- `model.eval()` → 모델을 평가 모드로 설정 (학습 X)
- `torch.no_grad()` → 그래디언트 계산을 막아 메모리 절약 & 속도 향상
- 새로운 입력 `test_X`에 대해 예측 수행