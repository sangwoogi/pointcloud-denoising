import os
import glob
import numpy as np
from scipy.spatial import cKDTree

def generate_snow_labels():
    raw_dir = 'data/nuScenes/samples/LIDAR_TOP'
    snow_dir = 'data/nuScenes_snow_sev5/samples/LIDAR_TOP'
    
    label_out_dir = 'data/nuScenes_snow_sev5/labels/LIDAR_TOP'
    os.makedirs(label_out_dir, exist_ok=True)

    snow_files = glob.glob(os.path.join(snow_dir, '*.pcd.bin'))
    print(f"총 {len(snow_files)}개의 파일에 대한 라벨링을 시작합니다...")

    for snow_filepath in snow_files:
        filename = os.path.basename(snow_filepath)
        raw_filepath = os.path.join(raw_dir, filename)

        # 1. 파일 읽기
        snow_pc = np.fromfile(snow_filepath, dtype=np.float32).reshape(-1, 5)
        raw_pc = np.fromfile(raw_filepath, dtype=np.float32).reshape(-1, 5)

        # 2. KD-Tree를 사용하여 원본 포인트 클라우드(x,y,z)의 고속 검색 구조 생성
        tree = cKDTree(raw_pc[:, :3])
        
        # 3. snow_pc의 각 포인트가 raw_pc에 존재하는지 거리 계산으로 확인 (k=1: 가장 가까운 1개 이웃)
        distances, _ = tree.query(snow_pc[:, :3], k=1)
        
        # 4. 거리가 0에 가까우면(예: 1e-4 이하) 기존 포인트, 거리가 멀면 새로 추가된 눈(snow) 포인트
        labels = (distances > 1e-4).astype(np.int32)

        # 5. .npy 파일로 저장
        label_filename = filename.replace('.pcd.bin', '.npy')
        label_save_path = os.path.join(label_out_dir, label_filename)
        np.save(label_save_path, labels)

    print("라벨 생성 및 저장 완료!")

if __name__ == '__main__':
    generate_snow_labels()