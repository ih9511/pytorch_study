import torch
import numpy as np


# 1D 텐서
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print(f"1D 텐서: \n{tensor_1d}\n")

# 2D 텐서
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"2D 텐서: \n{tensor_2d}\n")

# 3D 텐서
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D 텐서: \n{tensor_3d}\n")


# 3x3 영행렬
zero_tensor = torch.zeros(3, 3)
print(f"3x3 영행렬: \n{zero_tensor}\n")

# 3x3 모든 값이 1인 텐서
ones_tensor = torch.ones(3, 3)
print(f"3x3 모든 값이 1인 텐서: \n{ones_tensor}\n")

# 3x3 랜덤 텐서
rand_tensor = torch.rand(3, 3)
print(f"3x3 랜덤 텐서: \n{rand_tensor}\n")


# NumPy 배열을 텐서로 변환
# NumPy 배열 생성
np_array = np.array([[1, 2, 3], [4, 5, 6]])

# NumPy -> PyTorch 텐서로 변환
tensor_from_np = torch.from_numpy(np_array)
print(f"PyTorch 텐서: {type(tensor_from_np)}\n{tensor_from_np}\n")

# 텐서 -> NumPy 배열로 변환
np_from_tensor = tensor_from_np.numpy()
print(f"NumPy 배열: {type(np_from_tensor)}\n{np_from_tensor}\n")


# 텐서 연산
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(f"a + b = {a + b}\n")
print(f"a - b = {a - b}\n")
print(f"a * b = {a * b}\n")
print(f"a / b = {a / b}\n")

# 행렬 곱
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

matmul_result = torch.matmul(A, B)
print(f"행렬 곱: \n{matmul_result}\n")

# @ 연산자로도 행렬 곱 가능
matmul_result = A @ B
print(f"행렬 곱(@): \n{matmul_result}\n")


# 텐서 차원 변경(Reshape)
x = torch.arange(6)
print(f"{x.view(2, 3)}\n")
print(f"{x.reshape(3, 2)}\n")