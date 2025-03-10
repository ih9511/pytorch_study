import torch

from torch.utils.data import DataLoader, TensorDataset


# 더미 데이터(10개의 샘플, 각 샘플의 특성 3개)
X = torch.arange(30).reshape(10, 3).float()
y = torch.arange(10).reshape(10, 1).float()

# 데이터셋 & DataLoader 생성 (배치 크기 4)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# 배치 단위로 데이터 출력
for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
    print(f"배치 {batch_idx + 1}:")
    print(f"입력값 X:\n{x_batch}\n")
    print(f"정답값 y:\n{y_batch}\n")
    print('-' * 30)