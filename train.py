import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import spconv.pytorch as spconv
from tqdm import tqdm

from dataset import NuScenesSnowDataset
from model import MinkUNet

# --- Focal Loss 정의 ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C), targets: (N,)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss) # 예측 확률
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training using device: {device}")

    # 1. 하이퍼파라미터 및 저장 경로 설정
    num_epochs = 20
    batch_size = 4
    learning_rate = 0.001
    voxel_size = 0.1
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # 2. 데이터 로더 준비
    train_dataset = NuScenesSnowDataset(dataroot='data/nuScenes_snow_sev5', version='v1.0-mini')
    collate_fn = NuScenesSnowDataset.get_sparse_collate_fn(voxel_size=voxel_size)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )

    # 3. 모델, 손실 함수, 옵티마이저 초기화
    model = MinkUNet(in_channels=1, out_channels=2).to(device)
    criterion = FocalLoss(alpha=0.5, gamma=2.0) # 클래스 불균형 해소를 위한 Focal Loss 적용
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4. 학습 루프
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total_points = 0
        
        # tqdm을 사용한 진행률 표시
        pbar = tqdm(train_dataloader, desc=f"Epoch [{epoch}/{num_epochs}]")
        
        for coords, features, labels in pbar:
            coords = coords.to(device)
            features = features.to(device)
            labels = labels.to(device)
            
            # spconv 텐서 생성
            spatial_shape = (coords[:, 1:].max(dim=0)[0] + 1).tolist()
            b_size = coords[:, 0].max().item() + 1
            
            sparse_input = spconv.SparseConvTensor(
                features=features, 
                indices=coords.int(), 
                spatial_shape=spatial_shape, 
                batch_size=b_size
            )
            
            # 순전파 및 역전파
            optimizer.zero_grad()
            output = model(sparse_input)
            loss = criterion(output.features, labels)
            loss.backward()
            optimizer.step()
            
            # 메트릭 계산
            total_loss += loss.item()
            preds = torch.argmax(output.features, dim=1)
            correct += (preds == labels).sum().item()
            total_points += labels.size(0)
            
            # 진행률 바에 현재 Loss 업데이트
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # 에포크 단위 결과 출력
        avg_loss = total_loss / len(train_dataloader)
        epoch_acc = (correct / total_points) * 100
        print(f"-> Epoch [{epoch}/{num_epochs}] Summary | Avg Loss: {avg_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
        # 5. 모델 체크포인트 저장 (5 에포크마다 저장)
        if epoch % 5 == 0 or epoch == num_epochs:
            save_path = os.path.join(save_dir, f"minkunet_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train()