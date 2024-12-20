from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
import torch

# 해당 클래스는 .pkl 파일에 있는 데이터 로딩 역할
# 기본 한글 폰트를 생성하고나서 .pkl로 만들 때 사용되는 TrainDataProvider 클래스와 다른 역할
class FontDataset:
    """Enhanced dataset class for font images"""
    # 객체가 생성될 때 데이터를 불러오고 필요한 변수 선언
    def __init__(
        self,
        data_dir: str,
        file_name: str,
        img_size: int = 128,
        resize_fix: int = 90,
        augment: bool = True,
    ):
        self.file_name = file_name
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.resize_fix = resize_fix
        self.augment = augment
        
        # 데이터 로드
        self.data = self._load_data()
        
    # 데이터 로드
    def _load_data(self) -> List[Tuple[int, np.ndarray]]:
        """Load and preprocess image data"""
        # 최종 추출된 데이터가 담긴다.
        # 라벨(폰트 아이디), 고딕체글자, 라벨에 맞는 글자
        processed_data = []
        try:
            with open(self.data_dir / self.file_name, "rb") as f:
                while True:
                    try:
                        # 데이터를 하나씩 로드
                        example = pickle.load(f)
                        if not example:
                            continue
                            
                        # 데이터 구조 확인 및 처리
                        if len(example) >= 2:  # 최소 (font_id, image_data) 형식
                            # 정확한 데이터 구조는 
                            # (label=font_id, charid, img_bytes) 형식을 가진다.
                            font_id = example[0]
                            img_data = example[-1]  # 마지막 요소가 이미지 데이터
                            
                            # 이미지 처리
                            source_img, target_img = self._load_image_pair(img_data)
                            processed_data.append((font_id, source_img, target_img))
                            
                    except EOFError:
                        break
                    except Exception as e:
                        print(f"Error processing data: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            
        if not processed_data:
            raise ValueError("No data could be loaded from the pickle file")
            
        print(f"Loaded {len(processed_data)} samples")
        return processed_data

    def _load_image_pair(self, img_data: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """Process a pair of source and target images"""
        try:
            # Convert bytes to PIL Image
            img = Image.open(BytesIO(img_data))
            img_array = np.array(img)
            
            # Split into source and target
            w = img_array.shape[1]
            target_img = img_array[:, :w//2]  
            source_img = img_array[:, w//2:] 
            
            return source_img, target_img
            
        except Exception as e:
            print(f"Error processing image: {e}")
            # 오류 발생시 기본 이미지 반환
            blank = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            return blank, blank
      
    # 데이터셋의 샘플 개수를 반환
    def __len__(self) -> int:
        return len(self.data)
    
    # 데이터셋에 인덱스에 해당되는 샘플을 불러옴옴
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        font_id, source_img, target_img = self.data[idx]
        
        # 이미지 채널 표시
        # [1(gray scale), 128, 128]
        #  = [channels, height, width]
        source_tensor = torch.from_numpy(source_img).unsqueeze(0)
        target_tensor = torch.from_numpy(target_img).unsqueeze(0)
        
        return source_tensor, target_tensor, font_id