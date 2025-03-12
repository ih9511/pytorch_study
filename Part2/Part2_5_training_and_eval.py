import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

# 더미 데이터 생성
X = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]], dtype=torch.float32)
y = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]], dtype=torch.float32)

# Dataset & DataLoader 생성
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.fc(x)
    
model = SimpleModel(input_size=2, output_size=1)

# MSE 손실 함수 사용
loss_fn = nn.MSELoss()

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

loss_history = []
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
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
        
# Loss 값 그래프 출력
plt.plot(range(num_epochs), loss_history)
plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.title("Training Loss Over Epochs")
plt.show()

# 평가 모드 설정
model.eval()

# 테스트 데이터
test_X = torch.tensor([[5.0, 6.0], [6.0, 7.0], [7.0, 8.0]], dtype=torch.float32)

# 예측 수행
with torch.no_grad():
    test_pred = model(test_X)
    
print(f"테스트 데이터 예측값:\n{test_pred}")