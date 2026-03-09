# pointcloud-denoising
---
# folder structure
```
3D-PointCloud-Denoising/
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
