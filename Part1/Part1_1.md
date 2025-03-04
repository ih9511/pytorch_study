# Tensor 기본

## PyTorch 텐서란?

PyTorch에서 **텐서(Tensor)**는 **NumPy 배열과 비슷한 다차원 배열**이다.

GPU에서 계산할 수 있다는 점이 가장 큰 차이점이다.

## 텐서 생성

1. 기본적인 텐서 생성 방법

```python
import torch

# 1D 텐서
tensor_1d = torch.tensor([1, 2, 3, 4])
print(tensor_1d)

# 2D 텐서
tensor_2d = torch.tensor([[1, 2], [3, 4]])
print(tensor_2d)

# 3D 텐서
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(tensor_3d)
```

2. `torch.zeros()`, `torch.ones()`, `torch.rand()` 활용

```python
# 3x3 영행렬 (Zero Tensor)
zero_tensor = torch.zeros(3, 3)
print(zero_tensor)

# 3x3 모든 값이 1인 텐서
ones_tensor = torch.ones(3, 3)
print(ones_tensor)

# 3x3 랜덤 텐서
random_tensor = torch.rand(3, 3)
print(random_tensor)
```

3. NumPy 배열을 텐서로 변환

```python
import numpy as np

# NumPy 배열 생성
np_array = np.array([[1, 2, 3], [4, 5, 6]])

# NumPy -> PyTorch Tensor 변환
tensor_from_numpy = torch.from_numpy(np_array)
print(tensor_from_numpy)

# PyTorch Tensor -> NumPy 변환
numpy_from_tensor = tensor_from_numpy.numpy()
print(numpy_from_tensor)
```

## 텐서 연산

1. 기본 연산 (사칙연산)

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)
print(a - b)
print(a * b) # element-wise
print(a / b)
```

2. 행렬 곱 (Matrix Multiplication)

```python
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

# 행렬 곱
matmul_result = torch.matmul(A, B)
print(matmul_result)

# @ 연산자로도 가능
print(A @ B)
```

3. 텐서 차원 변경 (Reshape)

```python
x = torch.arange(6)
print(x.view(2, 3)) # (2x3) 형태로 변경
print(x.reshape(3, 2)) # (3x2) 형태로 변경
```