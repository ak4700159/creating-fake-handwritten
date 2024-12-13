import torch.nn as nn  # 이 임포트가 가장 먼저 있어야 합니다
import torch.nn.functional as F
import torch
from generator import FontGAN
from pathlib import Path  # 현대적인 파일 경로 처리를 위한 패키지
from gan_config import GANConfig
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms  # 이미지 변환
import numpy as np
from typing import List, Optional
from tqdm import tqdm
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt



def generate_new_characters(self, source_images: torch.Tensor, font_embeddings: torch.Tensor, save_dir: str):
    """학습에 사용되지 않은 새로운 한글 문자 생성
    
    이 함수는 학습 데이터에 없던 문자들을 생성합니다. 특히 다음 사항들을 개선했습니다:
    1. 노이즈 제거를 위한 향상된 필터링
    2. 특징 맵의 더 나은 정규화
    3. 스타일 일관성 강화
    """
    # 생성할 문자 선택 (학습 데이터에 없는 문자들)
    new_chars = [
        '가', '나', '다', '라', '마',  # 기본 받침 없는 문자
        '갈', '난', '달', '람', '맘'   # 받침 있는 문자
    ]
    
    self.eval()  # 평가 모드로 전환
    results = []
    
    try:
        with torch.no_grad():
            for char in new_chars:
                # 1. 특징 추출 및 노이즈 제거 강화
                encoded_source, skip_connections = self.encoder(source_images)
                
                # 향상된 노이즈 제거 프로세스
                skip_connections = self._enhanced_denoising(skip_connections)
                
                # 2. 임베딩 처리 및 스타일 강화
                embedding = self._get_style_enhanced_embedding(font_embeddings)
                embedded = self._combine_features_with_style(encoded_source, embedding)
                
                # 3. 개선된 디코더 처리
                fake_image = self.decoder(embedded, skip_connections)
                
                # 후처리 및 품질 향상
                enhanced_image = self._post_process_image(fake_image)
                
                results.append((char, enhanced_image))
                
        # 결과 저장
        self._save_generated_results(results, save_dir)
        
    finally:
        self.train()  # 학습 모드로 복원

def prepare_source_image(font_path: str, char: str, size: int = 128) -> torch.Tensor:
    """
    TTF 폰트 파일을 사용하여 고딕체 문자 이미지를 생성하고 전처리합니다.
    
    이 함수는 다음과 같은 단계로 작동합니다:
    1. 빈 이미지 생성
    2. 폰트 로드 및 크기 조정
    3. 문자 렌더링
    4. 이미지 중앙 정렬
    5. 텐서 변환 및 정규화
    
    Args:
        font_path: TTF 폰트 파일 경로
        char: 생성할 한글 문자
        size: 출력 이미지 크기 (기본값: 128)
        
    Returns:
        torch.Tensor: 전처리된 이미지 텐서 [1, size, size]
    """
    try:
        # 이미지 크기의 1.5배로 폰트 크기 설정 (여백 확보)
        font_size = int(size * 0.8)
        
        # 폰트 로드
        font = ImageFont.truetype(font_path, font_size)
        
        # 흰색 배경의 이미지 생성
        image = Image.new('L', (size, size), 255)
        draw = ImageDraw.Draw(image)
        
        # 문자의 크기 측정
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 중앙 정렬을 위한 위치 계산
        x = (size - text_width) // 2 - bbox[0]  # bbox[0]을 빼서 오프셋 보정
        y = (size - text_height) // 2 - bbox[1]  # bbox[1]을 빼서 오프셋 보정
        
        # 검은색으로 문자 그리기
        draw.text((x, y), char, font=font, fill=0)
        
        # PIL Image를 텐서로 변환
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # [-1, 1] 범위로 정규화
        ])
        
        return transform(image)
        
    except Exception as e:
        print(f"고딕체 이미지 생성 중 오류 발생: {str(e)}")
        raise

def prepare_source_images_batch(font_path: str, chars: List[str], size: int = 128) -> torch.Tensor:
    """
    여러 문자에 대한 고딕체 이미지를 배치로 생성합니다.
    
    Args:
        font_path: TTF 폰트 파일 경로
        chars: 생성할 한글 문자 리스트
        size: 출력 이미지 크기
        
    Returns:
        torch.Tensor: 배치 처리된 이미지 텐서 [batch_size, 1, size, size]
    """
    batch_images = []
    for char in chars:
        img_tensor = prepare_source_image(font_path, char, size)
        batch_images.append(img_tensor)
    
    return torch.stack(batch_images)


def generate_and_evaluate_characters(model, config, device, save_dir: str):
    """
    새로운 한글 문자를 생성하고 평가하는 통합 함수입니다.
    학습되지 않은 문자들에 대해 고딕체 이미지를 생성하고, 이를 기반으로 손글씨 스타일로 변환합니다.
    """
    # 학습에 사용되지 않은 새로운 문자 선택
    test_chars = ['한', '글', '만', '들', '기', '정', '말', '재', '미', '있']
    
    # 폰트 경로 설정
    font_path = "./get_data/fonts/source/source_font.ttf"
    
    # 임베딩 로드
    embedding_path = Path("./fixed_dir/EMBEDDINGS.pkl")
    font_embeddings = torch.load(embedding_path, map_location=device)
    
    # 저장 디렉토리 생성
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델을 평가 모드로 설정
    model.eval()
    
    try:
        with torch.no_grad():
            # 배치로 소스 이미지 생성
            source_images = prepare_source_images_batch(font_path, test_chars, config.img_size).to(device)
            
            # 각 문자에 대해 개별적으로 처리
            for idx, (char, source_image) in enumerate(zip(test_chars, source_images)):
                # 인코더로 특징 추출
                encoded_source, skip_connections = model.encoder(source_image.unsqueeze(0))
                
                # 노이즈 제거 적용
                skip_connections = model._enhanced_denoising(skip_connections)
                
                # 임베딩 처리
                font_id = torch.tensor([1], device=device)  # 타겟 폰트 ID (10번 폰트 스타일)
                embedding = model._get_embeddings(font_embeddings, font_id)
                embedded = torch.cat([encoded_source, embedding], dim=1)
                
                # 디코더로 이미지 생성
                fake_target = model.decoder(embedded, skip_connections)
                
                # 후처리 및 노이즈 제거
                enhanced_image = model._post_process_image(fake_target)
                
                # 결과 저장을 위한 이미지 준비
                source = (source_image.unsqueeze(0) + 1) / 2
                generated = (enhanced_image + 1) / 2
                
                # 비교 이미지 생성 (원본 고딕체 | 생성된 손글씨)
                comparison = torch.cat([source, generated], dim=3)
                
                # 결과 저장
                save_path = save_dir / f'generated_{char}_{idx+1}.png'
                save_image(comparison, save_path, normalize=False)
                
                print(f"생성 완료: {char} -> {save_path}")
                
            # 전체 결과 시각화
            plt.figure(figsize=(20, 4))
            for idx, char in enumerate(test_chars):
                plt.subplot(2, 5, idx+1)
                img_path = save_dir / f'generated_{char}_{idx+1}.png'
                img = Image.open(img_path)
                plt.imshow(np.array(img), cmap='gray')
                plt.title(f'Character: {char}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_dir / 'all_results.png')
            plt.close()
            
    except Exception as e:
        print(f"생성 과정에서 오류 발생: {str(e)}")
        raise
    finally:
        model.train()  # 모델을 다시 학습 모드로 전환

def main():
    """
    메인 실행 함수입니다. 모델을 로드하고 새로운 문자 생성을 실행합니다.
    """
    # 설정 초기화
    config = GANConfig(
        img_size=128,
        embedding_dim=128,
        conv_dim=128,
        batch_size=1,
        fonts_num=26
    )
    
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")
    
    try:
        # 체크포인트 로드
        checkpoint_path = "./final_data/checkpoints/best_model.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("체크포인트 파일을 찾을 수 없습니다.")
        
        # 모델 초기화 및 가중치 로드
        model = FontGAN(config, device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print("모델 로드 완료")
        
        # 결과 저장 디렉토리 설정
        output_dir = "./generated_results/evaluation"
        
        # 문자 생성 및 평가 실행
        print("새로운 문자 생성 시작...")
        generate_and_evaluate_characters(model, config, device, output_dir)
        
        print(f"생성 완료! 결과물 저장 위치: {output_dir}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()