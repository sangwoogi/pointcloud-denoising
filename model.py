import torch
import torch.nn as nn
import spconv.pytorch as spconv
from torch.utils.data import DataLoader

from dataset import NuScenesSnowDataset 

class BasicBlock(nn.Module):
    """spconv 기반의 Submanifold Convolution Block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class MinkUNet(nn.Module):
    """spconv 기반의 3D Sparse U-Net"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # [Encoder]
        self.enc1 = BasicBlock(in_channels, 32)
        # 다운샘플링 시 indice_key를 부여하여 추후 복원 시 위치를 기억
        self.pool1 = spconv.SparseConv3d(32, 32, kernel_size=2, stride=2, indice_key="pool1", bias=False)
        
        self.enc2 = BasicBlock(32, 64)
        self.pool2 = spconv.SparseConv3d(64, 64, kernel_size=2, stride=2, indice_key="pool2", bias=False)
        
        self.enc3 = BasicBlock(64, 128)
        
        # [Decoder]
        self.dec3 = BasicBlock(128, 64)
        # SparseInverseConv3d는 인코더의 indice_key를 참조하여 정확히 같은 좌표로 복원
        self.up2 = spconv.SparseInverseConv3d(64, 64, kernel_size=2, indice_key="pool2", bias=False)
        
        self.dec2 = BasicBlock(128, 64)  # Skip Connection (64 + 64 = 128)
        self.up1 = spconv.SparseInverseConv3d(64, 32, kernel_size=2, indice_key="pool1", bias=False)
        
        self.dec1 = BasicBlock(64, 32)   # Skip Connection (32 + 32 = 64)
        
        # [Final Layer] (정상 0, 눈 1)
        self.final = spconv.SubMConv3d(32, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        e1 = self.enc1(x)
        x = self.pool1(e1)
        
        e2 = self.enc2(x)
        x = self.pool2(e2)
        
        x = self.enc3(x)
        
        x = self.dec3(x)
        x = self.up2(x)
        
        # Skip Connection (e2 결합)
        x = x.replace_feature(torch.cat([x.features, e2.features], dim=1))
        
        x = self.dec2(x)
        x = self.up1(x)
        
        # Skip Connection (e1 결합)
        x = x.replace_feature(torch.cat([x.features, e1.features], dim=1))
        
        x = self.dec1(x)
        out = self.final(x)
        return out

if __name__ == '__main__':
    train_dataset = NuScenesSnowDataset(dataroot='data/nuScenes_snow_sev5', version='v1.0-mini')
    collate_fn = NuScenesSnowDataset.get_sparse_collate_fn(voxel_size=0.1)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MinkUNet(in_channels=1, out_channels=2).to(device)
    
    print(f"Model successfully initialized on {device}!")

    for step, (coords, features, labels) in enumerate(train_dataloader):
        coords = coords.to(device)
        features = features.to(device)
        labels = labels.to(device)
        
        # spconv를 위한 Spatial Shape 계산 (배치 내 최대 좌표값 기준)
        spatial_shape = (coords[:, 1:].max(dim=0)[0] + 1).tolist()
        batch_size = coords[:, 0].max().item() + 1
        
        # spconv.SparseConvTensor 생성
        sparse_input = spconv.SparseConvTensor(
            features=features, 
            indices=coords, 
            spatial_shape=spatial_shape, 
            batch_size=batch_size
        )
        
        output = model(sparse_input)
        
        print(f"\n--- Batch {step} Forward Pass Test ---")
        print(f"Input SparseTensor features shape : {sparse_input.features.shape}")
        print(f"Output SparseTensor features shape: {output.features.shape}") 
        print(f"Labels shape                      : {labels.shape}")
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output.features, labels) 
        print(f"Initial Loss (Untrained)          : {loss.item():.4f}")
        break