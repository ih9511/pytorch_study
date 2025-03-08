# Batch가 무엇인가요?

Batch는 신경망 학습에서 데이터를 한 번에 몇 개씩 처리할지를 결정하는 단위입니다. 딥러닝에서 batch라는 개념은 매우 중요하고, 학습 속도와 성능에 영향을 미치게 됩니다.

## 배치(Batch) 개념 정리

### 📌 전체 데이터셋을 한 번에 학습할 수 없을까?

일반적으로 딥러닝 모델은 수천 ~ 수백만 개의 데이터를 학습해야 합니다. 하지만 모든 데이터를 한 번에 모델에 넣어서 학습하면, 메모리 부족으로 학습이 불가능합니다. 따라서 데이터를 작은 단위(batch)로 나눠서 모델에 입력하는 방식을 사용합니다.

- 배치의 크기(Batch Size)
    - Mini-batch: 소규모 배치 학습, 일반적으로 32 ~ 512개
    - Full-batch: 한 번에 전체 데이터 학습, 거의 사용하지 않음.
    - Online Learning: 배치 크기=1, SGD에서 자주 사용됨.

### 📌 배치 학습이란?

예제를 통해 설명해보겠습니다.

*IF:*

- 1000개의 데이터 샘플을 가지고 있음.
- `batch_size=100`으로 설정.
- 즉, 한 번에 100개의 데이터를 처리하고, 가중치를 업데이트하게 됨.

_THEN:_

1. Batch 1 (1 ~ 100번 샘플) → 학습 & 가중치 업데이트
2. Batch 2 (101 ~ 200번 샘플) → 학습 & 가중치 업데이트
3. Batch 3 (201 ~ 300번 샘플) → 학습 & 가중치 업데이트
4. … (계속 반복) …
5. Batch 10 (901 ~ 1000번 샘플) → 학습 & 가중치 업데이트
    
    → 1000개의 샘플을 모두 학습하면 1 epoch 완료
    

이처럼 **모든 데이터를 한 번 학습하기 위해 여러 개의 배치를 사용하는 방식**이 미니배치 학습입니다.

### 📌 PyTorch에서 배치 개념 확인하기

이제 PyTorch `DataLoader`를 활용해서 배치가 실제로 어떻게 나눠지는지 확인해보겠습니다.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# 더미 데이터 (10개의 샘플, 각 샘플의 특성 3개)
X = torch.arange(30).view(10, 3).float()
y = torch.arange(10).view(10, 1).float()

# 데이터셋 & DataLoader 생성 (배치 크기 4)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# 배치 단위로 데이터 출력
for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
    print(f"배치 {batch_idx + 1}:")
    print(f"입력값 X:\n{x_batch}\n")
    print(f"정답값 y:\n{y_batch}\n")
    print("-" * 30)
```

- 입력 데이터 `X` → (10, 3)
- 출력 데이터 `y` → (10, 1)
- `batch_size=4` 이므로, 총 10개의 데이터를 4개씩 나누어 3번 반복
- 마지막 배치는 2개만 남아서 `(2, 3)` 크기로 출력됨

### 📌 배치 크기가 모델 학습에 미치는 영향

배치 크기를 어떻게 설정하느냐에 따라 모델 학습 성능이 달라질 수 있습니다.

- 배치 크기가 크면:
    - 학습이 안정적 (Gradient 평균값이 부드러움)
    - 하지만 메모리 사용량이 많음
- 배치 크기가 작으면:
    - 학습이 빠르게 진행되지만, 불안정할 수 있음 (SGD의 경우)
    - 모델이 더 일반화될 가능성이 있음 (데이터 다양성 증가)