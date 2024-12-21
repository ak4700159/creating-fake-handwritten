import torch.nn as nn
import torch

# 인코더, 디코더 관계를 리드미 파일에서 한 눈에 볼 수 있음

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.8))  # momentum 조정
            layers.append(nn.InstanceNorm2d(out_channels))  # Instance normalization 추가
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True, dropout=0.0):
        super().__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, img_dim=1, conv_dim=128):
        super().__init__()
        
        # Conv2d 함수 사용 : input channels, output channels, (kernel_size=4, stride=2, padding=1, use_bn=True) 기본값 별도 설정
        self.conv1 = ConvBlock(img_dim, conv_dim, use_bn=False)         # 128x128 -> 64x64
        self.conv2 = ConvBlock(conv_dim, conv_dim * 2)                  # 64x64 -> 32x32
        self.conv3 = ConvBlock(conv_dim * 2, conv_dim * 4)              # 32x32 -> 16x16
        self.conv4 = ConvBlock(conv_dim * 4, conv_dim * 8)              # 16x16 -> 8x8
        self.conv5 = ConvBlock(conv_dim * 8, conv_dim * 8)              # 8x8 -> 4x4

        self.conv6 = nn.Sequential(
            nn.Conv2d(conv_dim * 8, conv_dim * 8, kernel_size=1),
            nn.BatchNorm2d(conv_dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 입출력 채널 수는 유지
        self.conv7 = nn.Conv2d(
            conv_dim * 8, 
            conv_dim * 8, 
            kernel_size=2,  # 4x4를 3x3로 줄이기 위한 커널
            stride=1, 
            padding=0
        )

    def forward(self, x):
        # Skip connections 저장
        skip_connections = {}
        
        skip_connections['e1'] = x1 = self.conv1(x)        # 64x64
        skip_connections['e2'] = x2 = self.conv2(x1)       # 32x32
        skip_connections['e3'] = x3 = self.conv3(x2)       # 16x16
        skip_connections['e4'] = x4 = self.conv4(x3)       # 8x8
        skip_connections['e5'] = x5 = self.conv5(x4)       # 4x4
        skip_connections['e6'] = x6 = self.conv6(x5)       # 특징 정제
        skip_connections['e7'] = x7 = self.conv7(x6)       # 최종 특징
        
        return x7, skip_connections

class Decoder(nn.Module):
    def __init__(self, img_dim=1, embedded_dim=1152, conv_dim=128):
        super().__init__()
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                embedded_dim,
                conv_dim * 8,
                kernel_size=2,  # 3x3 -> 4x4
                stride=1,
                padding=0,
                output_padding=0
            ),
            nn.BatchNorm2d(conv_dim * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.deconv2 = DeconvBlock(
            conv_dim * 16, 
            conv_dim * 8, 
            kernel_size=3,  # 4x4 크기 유지
            stride=1,      # 크기 확장 없음
            padding=1,     # 크기 유지를 위한 패딩
            dropout=0.3
        )

        self.deconv3 = DeconvBlock(conv_dim * 16, conv_dim * 8, dropout=0.5)
        self.deconv4 = DeconvBlock(conv_dim * 16, conv_dim * 4)
        self.deconv5 = DeconvBlock(conv_dim * 8, conv_dim * 2)
        self.deconv6 = DeconvBlock(conv_dim * 4, conv_dim)
        
        self.deconv7 = nn.Sequential(
            nn.ConvTranspose2d(conv_dim * 2, img_dim, 4, 2, 1),
            nn.Tanh()
        )
        # 3x3 -> 4x4 (deconv1, deconv2) -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 (deconv3-deconv7)

    def forward(self, x, skip_connections):
        x = self.deconv1(x)
        x = torch.cat([x, skip_connections['e6']], dim=1)
        
        x = self.deconv2(x)
        x = torch.cat([x, skip_connections['e5']], dim=1)
        
        x = self.deconv3(x)
        x = torch.cat([x, skip_connections['e4']], dim=1)
        
        x = self.deconv4(x)
        x = torch.cat([x, skip_connections['e3']], dim=1)
        
        x = self.deconv5(x)
        x = torch.cat([x, skip_connections['e2']], dim=1)
        
        x = self.deconv6(x)
        x = torch.cat([x, skip_connections['e1']], dim=1)
        
        return self.deconv7(x)
