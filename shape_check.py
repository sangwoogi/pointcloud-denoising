# day 4
# 이전 스텝(Day 2-3)에서 완성한 Dataset 클래스 임포트
from dataset import NuScenesSnowDataset
from torch.utils.data import DataLoader

# 데이터셋 객체 생성
train_dataset = NuScenesSnowDataset(dataroot='data/nuScenes_snow_sev5', version='v1.0-mini')

# voxel_size는 통상적으로 자율주행 라이다의 경우 0.05 (5cm) 또는 0.1 (10cm)을 많이 사용합니다.
collate_fn = NuScenesSnowDataset.get_sparse_collate_fn(voxel_size=0.1)

# PyTorch DataLoader 생성
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,        # GPU 메모리에 맞게 조절 (보통 4 ~ 8)
    shuffle=True,        # 학습 시에는 데이터를 섞어줍니다.
    collate_fn=collate_fn,
    num_workers=4        # 데이터 로딩에 사용할 CPU 코어 수
)

# 데이터로더가 잘 작동하는지 1개 배치만 뽑아서 확인해보기 (Day 6를 위한 테스트)
if __name__ == '__main__':
    for step, (coords, features, labels) in enumerate(train_dataloader):
        print(f"Batch {step}:")
        print(f"  - Coords shape: {coords.shape}  # (N, 4) -> [batch_idx, x, y, z]")
        print(f"  - Features shape: {features.shape} # (N, 1) -> [intensity]")
        print(f"  - Labels shape: {labels.shape}   # (N,) -> 0 or 1")
        break # 1개 배치만 확인하고 종료