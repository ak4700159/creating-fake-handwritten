from layer import ConvBlock
import torch.nn as nn
import torch

# 판변자의 역할은 
class Discriminator(nn.Module):
    """
        판별자의 목적은 생성자가 생성한 가짜 내 손글씨 데이터를 가짜로 판단하도록 학습하는 것이 목적.:W 
        Args:
            category_num : 폰트 개수
            img_dim      : source imgae(1채널) + target image(1채널)
            disc_dim     : 초기 추출할 채널개수
    """
    def __init__(self, category_num: int, img_dim: int = 2, disc_dim: int = 64):

        super().__init__()
        
        # 이미지의 특징을 추출하는 레이어 -> features
        self.conv_blocks = nn.Sequential(
            ConvBlock(img_dim, disc_dim, use_bn=False),
            ConvBlock(disc_dim, disc_dim*2),
            ConvBlock(disc_dim*2, disc_dim*4),
            ConvBlock(disc_dim*4, disc_dim*8)
        )
        
        # 추출된 이미지 특징을 1차원으로 줄인다 -> patch_score
        self.patch_out = nn.Conv2d(disc_dim*8, 1, kernel_size=1, stride=1, padding=1)
        
        # 추출된 이미지 특징을 Flatten -> Linear 과정을 거쳐 25개의 출력 노드로 구성 -> category_score
        self.category_out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),            # 특징맵을 1 x 1 사이즈로 압축
            nn.Flatten(),                       # 평탄화
            nn.Linear(disc_dim*8, category_num) # FC : 폰트 개수만큼의 출력 노드 구성
        )
        
        # 1. convolution 블록을 통해 특징을 추출한다.
        # 2. 추출된 특징을 입력값으로, 패치 스코어, 카테고리 스코어를 출력값으로 얻는다.
    def forward(self, x):
        features = self.conv_blocks(x)
        patch_score = self.patch_out(features)
        category_score = self.category_out(features)
        # patch_score를 0 ~ 1 사이 값으로 반환, 
        return torch.sigmoid(patch_score), patch_score, category_score
