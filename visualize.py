import os
import numpy as np
import torch
import open3d as o3d
import spconv.pytorch as spconv
from model import MinkUNet

def visualize_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 모델 로드
    model = MinkUNet(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load('checkpoints/minkunet_epoch_20.pth', map_location=device))
    model.eval()

    # 2. 테스트할 샘플 데이터 1개 선택
    sample_bin = 'data/nuScenes_snow_sev5/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin'
    sample_label = 'data/nuScenes_snow_sev5/labels/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.npy'
    
    # 파일이 존재하는지 확인 (없으면 목록의 첫 번째 파일 자동 선택)
    if not os.path.exists(sample_bin):
        import glob
        sample_bin = glob.glob('data/nuScenes_snow_sev5/samples/LIDAR_TOP/*.pcd.bin')[0]
        sample_label = sample_bin.replace('samples', 'labels').replace('.pcd.bin', '.npy')

    print(f"Testing on: {os.path.basename(sample_bin)}")

    # 3. 데이터 로드
    scan = np.fromfile(sample_bin, dtype=np.float32).reshape(-1, 5)
    coords = scan[:, :3]
    features = scan[:, 3:4]
    labels = np.load(sample_label)

    # 4. 전처리 (Voxelization 및 Inverse Map 추출)
    voxel_size = 0.1
    discrete_coords = np.floor(coords / voxel_size).astype(np.int32)
    
    # return_inverse=True를 사용하여 모델의 예측 결과를 원래 포인트 수백만 개로 복원할 수 있도록 합니다.
    unique_coords, unique_indices, inverse_indices = np.unique(
        discrete_coords, axis=0, return_index=True, return_inverse=True
    )
    
    unique_features = features[unique_indices]

    # spconv 입력 텐서 생성
    tensor_coords = torch.tensor(unique_coords, dtype=torch.int32).to(device)
    tensor_features = torch.tensor(unique_features, dtype=torch.float32).to(device)
    
    batch_idx = torch.zeros((tensor_coords.shape[0], 1), dtype=torch.int32).to(device)
    batched_coords = torch.cat([batch_idx, tensor_coords], dim=1)
    batched_coords[:, 1:] -= batched_coords[:, 1:].min(dim=0)[0] # 양수 좌표계 변환

    spatial_shape = (batched_coords[:, 1:].max(dim=0)[0] + 1).tolist()
    
    sparse_input = spconv.SparseConvTensor(
        features=tensor_features, indices=batched_coords, 
        spatial_shape=spatial_shape, batch_size=1
    )

    # 5. 추론 (Inference)
    with torch.no_grad():
        output = model(sparse_input)
        unique_preds = torch.argmax(output.features, dim=1).cpu().numpy()

    # 6. 예측 결과를 원본 포인트 클라우드 크기로 복원
    full_preds = unique_preds[inverse_indices]

    # 7. Open3D 시각화 및 저장 준비
    os.makedirs('results', exist_ok=True)
    
    # --- (1) Ground Truth (실제 눈: 빨간색, 정상: 회색) ---
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(coords)
    gt_colors = np.zeros((len(coords), 3))
    gt_colors[labels == 1] = [1, 0, 0] # 눈(Red)
    gt_colors[labels == 0] = [0.5, 0.5, 0.5] # 정상(Gray)
    gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors)
    o3d.io.write_point_cloud("results/1_ground_truth.pcd", gt_pcd)

    # --- (2) Prediction (모델이 예측한 눈: 빨간색, 예측한 정상: 파란색) ---
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(coords)
    pred_colors = np.zeros((len(coords), 3))
    pred_colors[full_preds == 1] = [1, 0, 0] # 모델이 눈으로 예측(Red)
    pred_colors[full_preds == 0] = [0, 0, 1] # 모델이 정상으로 예측(Blue)
    pred_pcd.colors = o3d.utility.Vector3dVector(pred_colors)
    o3d.io.write_point_cloud("results/2_prediction.pcd", pred_pcd)

    # --- (3) Denoised Result (모델이 눈이라고 판단한 포인트를 제거한 최종 결과) ---
    denoised_coords = coords[full_preds == 0] # 예측이 0(정상)인 것만 필터링
    denoised_pcd = o3d.geometry.PointCloud()
    denoised_pcd.points = o3d.utility.Vector3dVector(denoised_coords)
    
    # 높이(Z값)에 따라 색상 입히기 (보기 좋게)
    z_vals = denoised_coords[:, 2]
    z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())
    denoised_colors = np.zeros((len(denoised_coords), 3))
    denoised_colors[:, 1] = 1 - z_norm # Green
    denoised_colors[:, 2] = z_norm     # Blue
    denoised_pcd.colors = o3d.utility.Vector3dVector(denoised_colors)
    
    o3d.io.write_point_cloud("results/3_denoised_final.pcd", denoised_pcd)

    print("시각화 결과 파일이 'results/' 폴더에 저장되었습니다.")
    print("- 1_ground_truth.pcd : 실제 라벨")
    print("- 2_prediction.pcd   : 모델 예측 결과")
    print("- 3_denoised_final.pcd : 눈이 제거된 최종 결과물")

if __name__ == '__main__':
    # pip install open3d
    visualize_inference()