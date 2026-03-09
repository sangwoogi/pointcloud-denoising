import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import spconv.pytorch as spconv

from dataset import NuScenesSnowDataset
# 기존 model.py 파일에서 MinkUNet 클래스를 불러옵니다.
from model import MinkUNet 

def overfit_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 데이터 로더 준비
    dataset = NuScenesSnowDataset(dataroot='data/nuScenes_snow_sev5', version='v1.0-mini')
    collate_fn = NuScenesSnowDataset.get_sparse_collate_fn(voxel_size=0.1)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # 2. 모델, 손실 함수, 옵티마이저 초기화
    model = MinkUNet(in_channels=1, out_channels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    # Adam 옵티마이저 사용 (학습률 0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    # 3. [핵심] 단 1개의 배치만 추출
    coords, features, labels = next(iter(dataloader))
    coords = coords.to(device)
    features = features.to(device)
    labels = labels.to(device)
    
    # spconv를 위한 Spatial Shape 및 Batch Size 계산
    spatial_shape = (coords[:, 1:].max(dim=0)[0] + 1).tolist()
    batch_size = coords[:, 0].max().item() + 1
    
    sparse_input = spconv.SparseConvTensor(
        features=features, 
        indices=coords.int(), 
        spatial_shape=spatial_shape, 
        batch_size=batch_size
    )
    
    print("=== 단일 배치 오버피팅 테스트 시작 ===")
    epochs = 100
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        # 기울기 초기화
        optimizer.zero_grad()
        
        # 순전파
        output = model(sparse_input)
        
        # 손실 계산
        loss = criterion(output.features, labels)
        
        # 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()
        
        # 10 에폭마다 결과 출력
        if epoch % 10 == 0 or epoch == 1:
            # 예측 클래스 추출 (Logit 값이 가장 큰 인덱스)
            preds = torch.argmax(output.features, dim=1)
            # 정확도 계산
            correct = (preds == labels).sum().item()
            accuracy = correct / labels.size(0) * 100
            
            print(f"Epoch [{epoch:3d}/{epochs}] Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    overfit_test()