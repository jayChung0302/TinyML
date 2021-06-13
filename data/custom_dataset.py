from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): csv 파일의 경로
            root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = Image.open(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class random_dataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=4):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.x = torch.randn(num_samples,3,32,32)
        self.y = torch.randint(num_classes, size=(num_samples,))
    
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.num_samples

if __name__ == '__main__':
    # 경고 메시지 무시하기
    import warnings
    warnings.filterwarnings("ignore")

    plt.ion()   # 반응형 모드

    landmarks_frame = pd.read_csv('/Users/chung/workspace/faces/face_landmarks.csv')

    n = 1
    img_name = landmarks_frame.iloc[n, 0]
    landmarks = landmarks_frame.iloc[n, 1:]
    landmarks = np.asarray(landmarks)
    landmarks = landmarks.astype('float').reshape(-1, 2)

    print('Image name: {}'.format(img_name))
    print('Landmarks shape: {}'.format(landmarks.shape))
    print('First 4 Landmarks: {}'.format(landmarks[:4]))

    def show_landmarks(image, landmarks):
        """Show image with landmarks"""
        """ 랜드마크(landmark)와 이미지를 보여줍니다. """
        plt.imshow(image)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
        plt.pause(0.001)  # 갱신이 되도록 잠시 멈춥니다.

    plt.figure()
    show_landmarks(Image.open(os.path.join('/Users/chung/workspace/faces/', img_name)),
                landmarks)
    plt.show()
    plt.pause(10) 
