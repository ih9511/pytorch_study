# PyTorch로 신경망 만들기

## PyTorch 모델 정의 (`torch.nn.Module`)

PyTorch에서 신경망을 정의할 때는 torch.nn.Module을 상속받아 클래스를 만들면 됩니다. 이제 모델을 만들고, 순전파(forward) 과정을 직접 정의하는 방법을 알아보도록 하겠습니다.

## 1. 가장 간단한 신경망 모델 정의

우선 하나의 선형 계층(fully connected layer)을 포함하는 모델을 만들어보겠습니다.

```python
import torch
import torch.nn as nn

# 모델 정의(torch.nn.Module 상속)
class SimpleModel(nn.Module):
	def __init__(self, input_size, output_size):
		super(SimpleModel, self).__init__()
		self.fc = nn.Linear(input_size, output_size) # 선형 계층(Fully Connected Layer)
		
	def forward(self, x):
		return self.fc(x)
```

- `nn.Module`을 상속받아 신경망을 정의함.
- `__init__()`에서 사용할 layer를 정의 (`nn.Linear`는 선형 계층)
- `forward(x)`: 입력 데이터를 받아 순전파 연산을 수행하는 함수.

## 2. 모델 생성 및 실행

위에서 정의한 `SimpleModel`을 실제 데이터에 적용시켜 보겠습니다.

```python
# 모델 생성 (입력 크기:3, 출력 크기: 1)
model = SimpleModel(input_size = 3, output_size = 1)

# 더미 입력 데이터 (batch_size=2, feature_size=3)
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
									
# 모델 실행 (순전파)
output = model(x)

print(f"출력값: {output}")
```

- 입력 텐서 `x` 의 크기: (배치 크기: 2, 특성 크기: 3)
- 모델이 출력하는 결과: (배치 크기: 2, 출력 크기: 1)

## 3. 다층 신경망 (MLP) 모델 정의

다층 신경망을 정의해보겠습니다. 여러 개의 선형 계층과 활성화 함수(ReLU)를 추가해 더 깊은 모델을 만들어 보도록 하겠습니다.

```python
# 단일 은닉층
class MLPModel(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(MLPModel, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size) # 첫 번째 은닉층
		self.relu = nn.ReLU() # 활성화 함수
		self.fc2 = nn.Linear(hidden_size, output_size) # 출력층
		
	def forward(self, x):
		x = self.fc1(x) # 첫 번째 선형 계층
		x = self.relu(x) # 활성화 함수 적용
		x = self.fc2(x) # 출력층
		return x
		
# 다중 은닉층
class DeepMLP(nn.Module):
	def __init__(self, input_size, hidden_sizes, output_size):
		super(DeepMLP, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_sizes[0]) # 첫 번째 은닉층
		self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1]) # 두 번째 은닉층
		self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2]) # 세 번째 은닉층
		self.fc4 = nn.Linear(hidden_sizes[2], output_size) # 출력층
		self.relu = nn.ReLU() # 활성화 함수
		
	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		x = self.relu(x)
		x = self.fc4(x)
		return x
```

## 4. 모델 실행

```python
# MLP 모델 생성 (입력 3, 은닉층 4, 출력 1)
model = MLPModel(input_size=3, hidden_size=4, output_size=1)

# 더미 입력 데이터
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 모델 실행
output = model(x)
print(f"출력값:\n{output}")
```

## 5. `nn.Sequential`을 활용한 간결한 모델 정의

위처럼 클래스를 직접 만드는 대신, nn.Sequential()을 사용하면 더 간결한 방식으로 모델을 정의할 수 있습니다.

```python
model = nn.Sequential(
	nn.Linear(3, 4),
	nn.ReLU(),
	nn.Linear(4, 1),
)

# 더미 데이터 실행
output = model(x)
print(f"출력값:\n{output}")
```