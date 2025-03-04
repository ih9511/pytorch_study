# **🔥 PyTorch**

### **📌 1) PyTorch 기본 개념 익히기**

1. **PyTorch 소개 및 설치**
    - PyTorch란 무엇인가? (TensorFlow와 비교)
    - PyTorch 최신 버전 설치 (CPU/GPU 지원 확인)
    - `torch`, `torchvision`, `torchaudio` 개요
2. **PyTorch 텐서(Tensor) 기본**
    
    - `torch.Tensor`의 개념 및 속성
    - 텐서 생성 및 조작 (`view()`, `reshape()`, `squeeze()`, `unsqueeze()`)
    - 텐서 연산 (기본 연산, `torch.matmul()`, `torch.mm()`, `torch.einsum()`)
3. **Autograd & 자동 미분**
    
    - `torch.autograd`의 원리
    - 그래디언트 계산 (`requires_grad=True`)
    - 역전파(`backward()`)와 `.grad` 활용
4. **PyTorch의 Device 설정 (CPU vs GPU)**
    - `torch.device('cuda')` 및 `torch.cuda.is_available()`
    - CPU ↔ GPU 간 텐서 이동

---

### **📌 2) PyTorch로 신경망 만들기**

1. **PyTorch 모델 정의 (`torch.nn.Module`)**
    
    - 신경망의 개념 및 레이어 구조
    - `torch.nn.Linear`, `torch.nn.ReLU`, `torch.nn.Sequential` 활용
    - `forward()` 메서드 이해
2. **PyTorch의 데이터 로딩 (`torch.utils.data.Dataset` & `DataLoader`)**
    - `Dataset`과 `DataLoader`를 이용한 데이터셋 처리
    - `transform`을 활용한 전처리 (`torchvision.transforms`)
    - 배치 학습을 위한 미니배치 구성
3. **손실 함수(`torch.nn`)와 최적화(`torch.optim`)**
    - 주요 손실 함수 (`MSELoss`, `CrossEntropyLoss` 등)
    - SGD, Adam 등 최적화 기법 이해
4. **모델 학습 및 평가**
    - `zero_grad()`, `backward()`, `step()` 학습 과정
    - 모델 평가 및 정확도 계산 (`torch.no_grad()`)
    - 모델 저장 및 불러오기 (`torch.save()` / `torch.load()`)

---

### **📌 3) PyTorch 고급 학습 (실전 활용)**

1. **CNN(합성곱 신경망) 실습**
    - `torchvision.models` 활용
    - `torch.nn.Conv2d`, `torch.nn.MaxPool2d` 이해
    - ResNet, EfficientNet 등 최신 모델 적용
2. **RNN/LSTM/GRU (자연어 처리 - NLP)**
- `torch.nn.RNN`, `torch.nn.LSTM`, `torch.nn.GRU` 이해
- 텍스트 데이터 전처리 및 임베딩 (`torch.nn.Embedding`)
- Transformer 개념 및 PyTorch 구현
1. **PyTorch에서 Transformer 활용 (BERT, GPT)**
- `torch.nn.Transformer` 기반 모델 구축
- Hugging Face `transformers` 라이브러리 활용
- 미세 조정(Fine-tuning) 및 사전 학습 모델 사용

---

### **📌 4) PyTorch 최신 기능 및 최적화**

1. **PyTorch 2.x의 `torch.compile()` 활용**
- PyTorch 2.0의 동적 그래프 컴파일 기능
- `torch.compile()`을 이용한 모델 최적화
1. **PyTorch Lightning으로 간편하게 모델 관리**
- `pytorch_lightning`을 활용한 코드 간소화
- 자동 로그 저장 및 체크포인트 관리
1. **ONNX & TorchScript (PyTorch 모델 배포)**
- 모델을 ONNX로 변환 (`torch.onnx.export()`)
- TorchScript (`torch.jit.trace()`)를 활용한 최적화
1. **PyTorch를 활용한 분산 학습 & 대형 모델 훈련**
- `torch.distributed`를 활용한 멀티 GPU 학습
- DeepSpeed와 Fully Sharded Data Parallel (FSDP) 기법 적용

---

### **📌 5) 프로젝트 실습**

1. **프로젝트 1 - 이미지 분류 (ResNet)**
- CIFAR-10 데이터셋을 이용한 ResNet 학습
- 데이터 증강 및 하이퍼파라미터 튜닝
1. **프로젝트 2 - 자연어 처리 (Transformer 기반 번역 모델)**
- 번역 데이터셋을 활용한 Transformer 모델 구현
- Hugging Face 모델을 활용한 파인튜닝
1. **프로젝트 3 - GAN (생성 모델)**
- DCGAN을 활용한 이미지 생성
- Stable Diffusion 기반 텍스트-이미지 변환 실습

---

## 🎯 **학습 목표**

✅ PyTorch의 기초 개념부터 실무 적용까지 학습

✅ 최신 PyTorch 2.x 버전 기능(`torch.compile()`, `Lightning`, `ONNX`) 습득

✅ 실전 프로젝트를 통해 딥러닝 모델을 직접 구축 및 최적화

---