# Autograd

PyTorch의 자동 미분(Autograd)가 중요한 이유는, 딥러닝 모델을 학습할 때 역전파(Backpropagation)를 자동으로 계산해주기 때문입니다.

## Autograd란?

PyTorch에서는 자동 미분(Autograd) 시스템을 제공합니다. 이 기능을 활용하면 텐서의 연산 기록을 저장하고, 자동으로 그래디언트를 계산할 수 있습니다. 즉, 딥러닝 모델을 학습할 때 손실 함수의 미분을 자동으로 구해서 최적화할 수 있습니다.

## 기본적인 Autograd 사용법

1. `requires_grad=True` 설정

	```python
	import torch

	# require_grad=True 설정된 텐서
	x = torch.tensor(2.0, requires_grad=True)
	y = x ** 2

	print("y:", y)

	# y를 x에 대해 미분 (dy/dx = 2x)
	y.backward()

	# x에 대한 그래디언트 값 출력
	print(f"x에 대한 그래디언트: {x.grad}")
	```

	- `requires_grad=True` → PyTorch가 이 텐서를 추적하고 미분 가능하도록 함.
	- `y.backward()` → y를 x에 대해 미분 (dy/dx)
	- `x.grad` → x에 대한 그래디언트 값 저장됨.
---
2. 벡터 연산에서 backward() 호출

	`backward()`는 스칼라 값(숫자 하나)에서만 직접 호출이 가능합니다. 벡터의 그래디언트를 구하려면 gradient 벡터를 명시해야 합니다.

	```python
	x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
	y = x ** 2 # y = [1, 4, 9]

	# 벡터 연산이므로, backward() 실행 시 gradient 벡터를 제공해야 함
	gradient = torch.tensor([1.0, 1.0, 1.0]) # dy/dx의 기준 벡터
	y.backward(gradient)

	print(f"x에 대한 그래디언트: {x.grad}")
	```

### `torch.no_grad()`로 미분 추적 방지

모델이 학습할 때는 `requires_grad=True`를 설정해야 하지만, 추론(inference) 단계에서는 불필요한 미분 계산을 막아 성능을 최적화해야 합니다.

```python
x = torch.tensor(3.0, requires_grad=True)

# no_grad() 사용 -> 그래디언트 추적 X
with torch.no_grad():
	y = x ** 2 # 연산 수행 (하지만 requires_grad=False)
	
print(y.requires_grad)
```

### `detach()`를 활용해 미분 추적 해제

`detach()`를 사용하면 특정 텐서를 미분 계산에서 분리할 수 있습니다.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2

# detach() 사용 -> y가 더 이상 미분 추적하지 않음
y_detached = y.detach()

print(y.requires_grad) # True
print(y_detached.requires_grad) # False
```