# 3D Point Cloud Denoising for Autonomous Driving ❄️🚗

딥러닝 기반의 3D Point Cloud 전처리 네트워크입니다. 
자율주행 라이다(LiDAR) 센서 데이터에 낀 악천후 노이즈(눈, 비)를 픽셀(포인트) 단위로 찾아내어 제거합니다.

## 🚀 Project Overview
* **Task:** 3D Point Cloud Binary Semantic Segmentation (Noise vs. Clean)
* **Architecture:** Sparse 3D U-Net (MinkUNet inspired)
* **Framework:** PyTorch, `spconv` (Optimized for modern GPUs like RTX 30/40/50 series)
* **Dataset:** nuScenes-mini (with synthetic snow corruption)

## 📊 Performance Evaluation

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Overall Accuracy** | **97.75%** | 전체 포인트 클라우드에 대한 분류 정확도 |
| **Precision** | **84.65%** | **원본 보존율**: 정상 포인트를 노이즈로 오검출하지 않은 비율 |
| **Recall** | **82.39%** | **눈 제거율**: 실제 눈 노이즈를 정확하게 찾아내 제거한 비율 |
| **F1 Score** | **0.8351** | Precision과 Recall의 조화 평균 |
| **Snow Class IoU** | **71.68%** | 눈 노이즈 클래스에 대한 정밀 평가지표 (Intersection over Union) |

## 🛠️ Environment Setup
\`spconv\`를 활용하여 복잡한 C++ 빌드 없이 구축 가능합니다. (CUDA 12.0 기준)

```bash
pip install torch torchvision torchaudio
pip install spconv-cu120
pip install open3d nuscenes-devkit scipy tqdm
```

## 🚩 Folder Structure
```
PointCloud-Denoising/
├── data/                   # (Git 제외) nuScenes 데이터셋
├── checkpoints/            # (Git 제외) 학습된 가중치
├── results/                # (Git 제외) 시각화 결과물
├── dataset.py              # 데이터 로더 (cKDTree 라벨링 포함)
├── model.py                # spconv 기반 MinkUNet 구조
├── train.py                # 학습 루프 (Focal Loss)
├── eval.py                 # 지표 평가 (IoU, Recall, Precision)
├── inference.py            # 단일 파일 추론 및 정제 모듈
├── visualize.py            # Open3D 시각화 스크립트
├── requirements.txt        # 필요 패키지 목록
└── README.md               # 프로젝트 설명서
```
