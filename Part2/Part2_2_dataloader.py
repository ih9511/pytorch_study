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
        return self.x_data[idx], self.y_data[idx]
    
from torch.utils.data import DataLoader

# 데이터셋 생성
dataset = CustomDataset()

# DataLoader 설정 (배치 크기: 2, 데이터 섞기: True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# DataLoader에서 배치 단위로 데이터 가져오기
for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
    print(f"배치 {batch_idx + 1}")
    print(f"입력값 x:\n{x_batch}\n")
    print(f"정답값 y:\n{y_batch}")