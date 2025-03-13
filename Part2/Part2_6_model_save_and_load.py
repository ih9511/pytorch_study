import torch
import torch.nn as nn

# 간단한 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 1)
        
    def forward(self, x):
        return self.fc(x)
    
# 모델 생성
model = SimpleModel()

# state_dict()
# 모델 가중치 저장
torch.save(model.state_dict(), "../models/simple_model_weights.pth")
print("모델 가중치 저장 완료!")

# 새로운 모델 생성
new_model = SimpleModel()

# 저장된 가중치 로드
new_model.load_state_dict(torch.load("../models/simple_model_weights.pth"))

# 평가 모드로 설정
new_model.eval()
print("모델 가중치 불러오기 완료!")

# ===========================================================================

# torch.save(model)
torch.save(model, "../models/full_simple_model.pth")
print("모델 전체 저장 완료!")

# 저장된 전체 모델 불러오기
loaded_model = torch.load("../models/full_simple_model.pth", weights_only=False)

# 평가 모드 설정
loaded_model.eval()
print("전체 모델 불러오기 완료!")