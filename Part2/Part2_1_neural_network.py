import torch
import torch.nn as nn


# 모델 정의(torch.nn.Module 상속)
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.fc(x)
    
# 모델 생성 및 실행    
model = SimpleModel(input_size=3, output_size=1)

# 더미 입력 데이터(batch_size=2, feature_size=3)
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 모델 실행(forward)
output = model(x)

print(f"Output: {output}")