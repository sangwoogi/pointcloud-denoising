import os
import argparse
import numpy as np
import torch
import open3d as o3d
import spconv.pytorch as spconv
from model import MinkUNet

def denoise_pointcloud(input_path, output_path, model_path, voxel_size=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Loading model from {model_path}...")
    
    # 1. 모델 초기화 및 가중치 로드
    model = MinkUNet(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"[*] Processing input file: {input_path}")
    
    # 2. Point Cloud 로드 (라벨 불필요)
    scan = np.fromfile(input_path, dtype=np.float32).reshape(-1, 5)
    coords = scan[:, :3]
    features = scan[:, 3:4]

    # 3. 전처리 (Voxelization)
    discrete_coords = np.floor(coords / voxel_size).astype(np.int32)
    unique_coords, unique_indices, inverse_indices = np.unique(
        discrete_coords, axis=0, return_index=True, return_inverse=True
    )
    unique_features = features[unique_indices]

    # spconv 입력 텐서 변환
    tensor_coords = torch.tensor(unique_coords, dtype=torch.int32).to(device)
    tensor_features = torch.tensor(unique_features, dtype=torch.float32).to(device)
    
    batch_idx = torch.zeros((tensor_coords.shape[0], 1), dtype=torch.int32).to(device)
    batched_coords = torch.cat([batch_idx, tensor_coords], dim=1)
    batched_coords[:, 1:] -= batched_coords[:, 1:].min(dim=0)[0] 

    spatial_shape = (batched_coords[:, 1:].max(dim=0)[0] + 1).tolist()
    sparse_input = spconv.SparseConvTensor(
        features=tensor_features, indices=batched_coords, 
        spatial_shape=spatial_shape, batch_size=1
    )

    # 4. 추론 (Inference)
    print("[*] Running inference...")
    with torch.no_grad():
        output = model(sparse_input)
        unique_preds = torch.argmax(output.features, dim=1).cpu().numpy()

    # 원본 해상도로 복원
    full_preds = unique_preds[inverse_indices]

    # 5. 눈 노이즈 제거 (Prediction == 0 인 정상 포인트만 추출)
    clean_coords = coords[full_preds == 0]
    
    # 6. 결과 저장 (.pcd 형식)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(clean_coords)
    o3d.io.write_point_cloud(output_path, pcd)
    
    print(f"[+] Successfully saved denoised point cloud to: {output_path}")
    print(f"    - Original points: {len(coords)}")
    print(f"    - Denoised points: {len(clean_coords)}")
    print(f"    - Removed noise  : {len(coords) - len(clean_coords)} points")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D Point Cloud Snow Denoising Module")
    parser.add_argument('--input', type=str, required=True, help="Path to noisy .pcd.bin file")
    parser.add_argument('--output', type=str, default='results/clean_output.pcd', help="Path to save clean .pcd file")
    parser.add_argument('--ckpt', type=str, default='checkpoints/minkunet_epoch_20.pth', help="Model checkpoint path")
    
    args = parser.parse_args()
    denoise_pointcloud(args.input, args.output, args.ckpt)