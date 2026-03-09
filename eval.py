import torch
import spconv.pytorch as spconv
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import NuScenesSnowDataset
from model import MinkUNet

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}...")

    # 1. 평가용 데이터 로더 준비
    # 실제로는 Train/Val 폴더를 나누어야 하지만, 현재는 파이프라인 검증을 위해 동일 데이터 사용
    dataset = NuScenesSnowDataset(dataroot='data/nuScenes_snow_sev5', version='v1.0-mini')
    collate_fn = NuScenesSnowDataset.get_sparse_collate_fn(voxel_size=0.1)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # 2. 모델 로드 및 학습된 가중치 덮어씌우기
    model = MinkUNet(in_channels=1, out_channels=2).to(device)
    
    # 학습 단계에서 저장한 20 에포크 가중치 경로
    checkpoint_path = 'checkpoints/minkunet_epoch_20.pth' 
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # 평가 모드 전환 (Dropout, BatchNorm 등의 동작 고정)
    model.eval() 

    # 3. 혼동 행렬(Confusion Matrix) 계산용 변수 초기화
    TP = 0  # True Positive: 눈을 눈이라고 정확히 예측
    FP = 0  # False Positive: 정상을 눈이라고 잘못 예측 
    FN = 0  # False Negative: 눈을 정상이라고 잘못 예측
    TN = 0  # True Negative: 정상을 정상이라고 정확히 예측

    print("평가를 시작합니다...")
    
    with torch.no_grad(): # 평가 시에는 기울기 계산을 하지 않음 (메모리 절약, 속도 향상)
        for coords, features, labels in tqdm(dataloader):
            coords = coords.to(device)
            features = features.to(device)
            labels = labels.to(device)

            spatial_shape = (coords[:, 1:].max(dim=0)[0] + 1).tolist()
            b_size = coords[:, 0].max().item() + 1
            
            sparse_input = spconv.SparseConvTensor(
                features=features, indices=coords.int(), 
                spatial_shape=spatial_shape, batch_size=b_size
            )

            # 예측 수행
            output = model(sparse_input)
            preds = torch.argmax(output.features, dim=1) # 클래스 0(정상) 또는 1(눈)

            # 배치별 TP, FP, FN, TN 누적 (클래스 1: 눈 기준)
            TP += ((preds == 1) & (labels == 1)).sum().item()
            FP += ((preds == 1) & (labels == 0)).sum().item()
            FN += ((preds == 0) & (labels == 1)).sum().item()
            TN += ((preds == 0) & (labels == 0)).sum().item()

    # 4. 최종 지표 계산
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*40)
    print("         [모델 평가 결과]")
    print("="*40)
    print(f"Overall Accuracy : {(TP + TN) / (TP + TN + FP + FN) * 100:.2f}%")
    print(f"Precision (원본 보존) : {precision * 100:.2f}%")
    print(f"Recall    (눈 제거율) : {recall * 100:.2f}%")
    print(f"F1 Score             : {f1_score:.4f}")
    print(f"Snow Class IoU       : {iou * 100:.2f}%")
    print("="*40)

if __name__ == '__main__':
    evaluate_model()