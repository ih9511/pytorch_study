# 모델 저장 & 불러오기

모델을 저장하는 방법은 크게 두 가지가 있습니다.

- PyTorch 모델 저장 방식
    1. 모델 가중치(Weights)만 저장 → `state_dict()` 사용
    2. 모델 전체(구조 + 가중치) 저장 → `torch.save(model)` 

각 방법을 자세히 알아보겠습니다.

## 1. `state_dict()`를 이용한 모델 가중치 저장

PyTorch에서 모델 가중치만 저장하려면 `state_dict()`를 사용하면 됩니다.

```python
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

# 모델 가중치 저장
torch.save(model.state_dict(), "model_weights.pth")
print("모델 가중치 저장 완료!")
```

- `model.state_dict()` → 모델의 모든 가중치를 딕셔너리 형태로 변환
- `“model_weights.pth”` → 파일로 저장됨

## 2. 저장된 가중치를 불러오기

이제 저장된 가중치를 새로운 모델에 로드(load)해보겠습니다.

```python
# 새로운 모델 생성
new_model = SimpleModel()

# 저장된 가중치 로드
new_model.load_state_dict(torch.load("model_weights.pth"))

# 평가 모드로 설정
new_model.eval()

print("모델 가중치 불러오기 완료!")
```

## 3. `torch.save(model)`을 이용한 전체 모델 저장

모델의 구조 + 가중치를 함께 저장하려면 `torch.save(model, “파일 이름”)`을 사용하면 됩니다.

```python
# 모델 전체 저장 (구조 + 가중치)
torch.save(model, "full_model.pth")
print("모델 전체 저장 완료!")
```

- 이 방식은 모델 구조까지 포함하여 저장하므로, 나중에 같은 모델 클래스를 정의하지 않아도 불러올 수 있음.

## 4. 저장된 전체 모델 불러오기

```python
# 저장된 전체 모델 불러오기
loaded_model = torch.load("full_model.pth", weights_only=False)

# 평가 모드 설정
loaded_model.eval()

print("전체 모델 불러오기 완료!")
```

## 5. 모델 저장 & 불러오기 비교 정리

| 방법 | 저장 내용 | 파일 크기 | 장점 | 단점 |
| --- | --- | --- | --- | --- |
| `state_dict()` | 모델 가중치만 저장 | 작음 | 다른 모델에서도 적용 가능 | 모델 클래스를 다시 정의해야 함 |
| `torch.save(model)` | 모델 구조 + 가중치 저장 | 큼 | 한 줄로 불러오기 가능 | PyTorch 버전이 다르면 오류 발생 가능 |
- 실무에서는 `state_dict()`를 사용하여 가중치만 저장하는 경우가 많음.
- 모델 구조까지 함께 저장해야 하면 `torch.save(model)`을 사용.