import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes

class NuScenesSnowDataset(Dataset):
    def __init__(self, dataroot, version='v1.0-mini'):
        self.dataroot = dataroot
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        self.lidar_tokens = []
        for sample in self.nusc.sample:
            self.lidar_tokens.append(sample['data']['LIDAR_TOP'])

    def __len__(self):
        return len(self.lidar_tokens)

    def __getitem__(self, idx):
        lidar_token = self.lidar_tokens[idx]
        lidar_data = self.nusc.get('sample_data', lidar_token)
        
        pcl_path = os.path.join(self.dataroot, lidar_data['filename'])
        label_path = pcl_path.replace('samples', 'labels').replace('.pcd.bin', '.npy')

        scan = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)
        coords = scan[:, :3] 
        features = scan[:, 3:4] 
        labels = np.load(label_path)

        return torch.tensor(coords, dtype=torch.float32), \
               torch.tensor(features, dtype=torch.float32), \
               torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def get_sparse_collate_fn(voxel_size=0.1):
        def sparse_collate_fn(batch):
            coords_list, features_list, labels_list = [], [], []

            for batch_idx, (coords, features, labels) in enumerate(batch):
                # 1. Voxelization
                discrete_coords = torch.floor(coords / voxel_size).int()
                
                # 2. Sparse Quantize (중복 Voxel 제거)
                # numpy unique를 사용하여 첫 번째 고유 포인트의 인덱스 추출
                _, unique_indices = np.unique(discrete_coords.cpu().numpy(), axis=0, return_index=True)
                
                unique_coords = discrete_coords[unique_indices]
                unique_features = features[unique_indices]
                unique_labels = labels[unique_indices]
                
                # 3. Batch Index 추가 (N, 4) 형태로 생성: [batch_idx, x, y, z]
                batch_idx_tensor = torch.full((unique_coords.shape[0], 1), batch_idx, dtype=torch.int32)
                batched_coords = torch.cat([batch_idx_tensor, unique_coords], dim=1)
                
                coords_list.append(batched_coords)
                features_list.append(unique_features)
                labels_list.append(unique_labels)

            coords_batch = torch.cat(coords_list, dim=0)
            features_batch = torch.cat(features_list, dim=0)
            labels_batch = torch.cat(labels_list, dim=0)

            # spconv는 좌표가 무조건 0 이상의 정수여야 하므로 음수 좌표를 양수로 평행 이동
            coords_batch[:, 1:] -= coords_batch[:, 1:].min(dim=0)[0]

            return coords_batch, features_batch, labels_batch
        
        return sparse_collate_fn