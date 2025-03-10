# PyTorch 데이터 로딩

PyTorch에서 데이터를 로드할 때는 두 가지 클래스를 활용합니다.

1. `torch.utils.data.Dataset` → 데이터셋을 정의하는 역할
2. `torch.utils.data.DataLoader` → 배치 단위로 데이터를 불러오는 역할

## 1. `Dataset`을 활용한 커스텀 데이터셋 만들기

PyTorch는 기본적으로 `torchvision.datasets` 같은 유명한 데이터셋을 제공하지만, 우리는 커스텀 데이터셋을 만들어 직접 불러와보겠습니다.

```python
import torch
from torch.utils.data import Dataset

# 커스텀 데이터셋 정의
class CustomDataset(Dataset):
	def __init__(self):
		# 더미 데이터 (X: 입력, y: 정답)
		self.x_data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
		self.y_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
				
	def __len__(self):
		return len(self.x_data) # 데이터 샘플 개수 반환
				
	def __getitem__(self, idx):
		return self.x_data[idx], self.y_data[idx] # idx 번째 샘플 반환
```

- `__init__()`  → 데이터셋을 초기화하는 메서드
- `__len__()` → 데이터셋의 샘플 개수를 반환
- `__getitem__()` → 인덱스를 받아 해당 샘플을 반환

## 2. `DataLoader`로 미니배치 단위로 데이터 불러오기

이제 `Dataset`을 `DataLoader`와 함께 사용하여 미니배치 단위로 데이터를 불러올 수 있습니다.

```python
from torch.utils.data import DataLoader

# 데이터셋 생성
dataset = CustomDataset()

# DataLoader 설정 (배치 크기: 2, 데이터 섞기: True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# DataLoader에서 배치 단위로 데이터 가져오기
for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
	print(f"배치 {batch_idx + 1}:")
	print(f"입력값 x:\n{x_batch}")
	print(f"정답값 y:\n{y_batch}")		
```

- `batch_size=2` → 한 번에 2개의 샘플을 가져옴.
- `shuffle=True` → 데이터를 랜덤하게 섞음.
- `enumerate(dataloader)` → 한 번에 한 배치씩 데이터를 가져옴.

## 3. torchvision.datasets을 활용한 이미지 데이터 불러오기

PyTorch는 유명한 데이터셋(ex. MNIST, CIFAR-10)을 쉽게 불러올 수 있는 `torchvision.datasets`을 제공합니다.

```python
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# 데이터 변환 정의 (이미지를 텐서로 변환)
transform = transforms.ToTensor()

# MNIST 데이터셋 불러오기 (자동 다운로드)
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)

# DataLoader 설정 (배치 크기: 64)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 데이터 확인
images, labels = next(iter(train_loader))
print(f"첫 번째 배치 이미지 크기: {images.shape}")
print(f"첫 번째 배치 라벨: {labels}")
```

- `torchvision.transforms.ToTensor()` → 이미지를 PyTorch 텐서로 변환
- `download=True` → 데이터가 없으면 자동 다운로드