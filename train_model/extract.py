import torch
from generator import FontGAN
from gan_config import GANConfig
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
from torchvision.utils import save_image
from pathlib import Path
import os

def prepare_source_image(char: str, font_path: str, size: int = 128) -> torch.Tensor:
    """한글 문자의 고딕체 이미지를 생성하고 텐서로 변환"""
    image = Image.new('L', (size, size), 255)
    font = ImageFont.truetype(font_path, int(size * 0.8))
    draw = ImageDraw.Draw(image)
    
    bbox = draw.textbbox((0, 0), char, font=font)
    x = (size - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (size - (bbox[3] - bbox[1])) // 2 - bbox[1]
    
    draw.text((x, y), char, font=font, fill=0)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return transform(image)

def generate_handwritten_samples(model, config, device, save_dir: str):
    """학습되지 않은 문자들에 대한 손글씨체 생성"""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 테스트할 문자들
    test_chars = ['한', '글', '만', '들', '기', '정', '말', '재', '미', '있', '다']
    
    # 폰트 및 임베딩 로드
    font_path = "../create_train_data/fonts/source/source_font.ttf"
    font_embeddings = torch.load("./fixed_dir/EMBEDDINGS.pkl", map_location=device)
    font_id = torch.tensor([10], device=device)  # 10번 폰트 스타일 사용
    
    try:
        with torch.no_grad():
            # 모든 문자에 대한 소스 이미지 생성
            sources = []
            for char in test_chars:
                source = prepare_source_image(char, font_path)
                sources.append(source)
            sources = torch.stack(sources).to(device)
            
            # 배치 처리
            encoded_sources = []
            generated_images = []
            
            for source in sources:
                # 인코더로 특징 추출
                encoded_source, skip_connections = model.encoder(source.unsqueeze(0))
                
                # 임베딩 처리
                embedding = model._get_embeddings(font_embeddings, font_id)
                embedded = torch.cat([encoded_source, embedding], dim=1)
                
                # 디코더로 이미지 생성
                fake_target = model.decoder(embedded, skip_connections)
                
                encoded_sources.append(encoded_source)
                generated_images.append(fake_target)
            
            # 결과 이미지 생성 및 저장
            for idx, (char, source, generated) in enumerate(zip(test_chars, sources, generated_images)):
                comparison = torch.cat([
                    source.unsqueeze(0).to(device),  # 원본 고딕체
                    generated                         # 생성된 손글씨
                ], dim=3)
                
                # [-1, 1] -> [0, 1] 범위로 변환
                comparison = (comparison + 1) / 2
                
                # 저장
                save_path = save_dir / f"{char}_{idx+1}.png"
                save_image(comparison, save_path, 
                          normalize=False,
                          nrow=1,
                          padding=10)
                print(f"생성 완료: {char} -> {save_path}")
                
    except Exception as e:
        print(f"생성 과정에서 오류 발생: {str(e)}")
        raise
    finally:
        model.train()

def main():
    config = GANConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    try:
        model = FontGAN(config, device)
        checkpoint = torch.load(
            "./2024-12-21(3) pre_trained_data/checkpoints/best_model.pth", 
            map_location=device
        )
        
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print("모델 로드 완료")
        
        print("손글씨 생성 시작...")
        generate_handwritten_samples(model, config, device, "./generated_results/evaluation")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()