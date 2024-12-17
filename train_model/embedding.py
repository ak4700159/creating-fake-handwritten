import torch
import torch.nn.functional as F  # F로 자주 임포트되는 PyTorch의 함수형 인터페이스
from pathlib import Path

# 임베딩 벡터는 GAN 학습에 있어 중요한 역할을 한다. 
# 카테고리 벡터, 폰스 스타일 벡터라고 정의하는데 그 이유는
# 인코더를 거쳐 나온 특징들은 글자의 형태에 대한 특징들이다.
# 형태에 대한 특징에 이 임베딩 벡터가 붙여져 디코더의 입력값으로 들어간다
# 디코더의 출력은 특정 폰트의 글자가 나오게 된다.


def generate_font_embeddings(
    fonts_num: int,
    embedding_dim: int = 128,
    save_dir: str = "./fixed_dir",
    stddev: float = 0.01
):
    """폰트 스타일 임베딩을 생성하고 저장하는 함수
    
    이 함수는 각 폰트에 대해 3x3 공간 구조를 가진 임베딩을 생성
    왜 3x3 공간 구조를 가지도록 했냐면 단순 한글의 구조가 초성, 중성, 종성으로
    이루어져 있는데 1x1 공간 구조는 그만큼의 특징을 모두 담을 수 없을 거라고 생각.
    
    Args:
        fonts_num (int): 생성할 폰트 임베딩의 수
        embedding_dim (int): 각 임베딩의 차원 수
        save_dir (str): 임베딩을 저장할 디렉토리 경로
        stddev (float): 초기 랜덤 값의 표준편차
    
    Returns:
        torch.Tensor: 생성된 폰트 임베딩 텐서
    """
    
    # 직접 3x3 구조로 임베딩 생성
    embeddings = torch.randn(fonts_num, embedding_dim, 3, 3) * stddev
    
    # 각 폰트별로 normalize
    embeddings = embeddings.view(fonts_num, -1)  # [fonts_num, embedding_dim * 9]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings.view(fonts_num, embedding_dim, 3, 3)
    
    save_path = Path(save_dir) / 'EMBEDDINGS.pkl'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.save(embeddings, save_path)
        print(f"Font embeddings generated and saved to {save_path}")
        print(f"Shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        
    return embeddings


def load_embeddings(embedding_path: str, device: torch.device):
    """안전하게 임베딩 로드"""
    try:
        # weights_only=True로 설정하여 안전하게 로드
        font_embeddings = torch.load(
            embedding_path,
            weights_only=True,
            map_location=device
        )
        
        # 텐서 타입 및 형태 확인
        if not isinstance(font_embeddings, torch.Tensor):
            raise TypeError("Loaded embeddings is not a torch.Tensor")
            
        print(f"Loaded font embeddings with shape: {font_embeddings.shape}")
        return font_embeddings
        
    except Exception as e:
        raise Exception(f"Error loading font embeddings: {e}")
