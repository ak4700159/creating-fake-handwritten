from dataclasses import dataclass
from torch.serialization import add_safe_globals


@dataclass
class GANConfig:
    """Configuration for GAN training"""
    # 기본 설정
    img_size: int = 128
    embedding_dim: int = 128
    conv_dim: int = 128
    batch_size: int = 32
    fonts_num: int = 26
    
    # 학습 관련 설정
    max_epoch: int = 2000
    schedule: int = 100
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    
    # 손실함수 가중치 조정정
    # Stage 1 (초기 학습)
    stage1_l1_lambda: float = 80.0   
    stage1_const_lambda: float = 60.0 
    
    # Stage 2 (후기 학습)
    stage2_l1_lambda: float = 100.0   
    stage2_const_lambda: float = 30.0   

    adv_lambda: float = 3.0           
    cat_lambda: float = 1.2           
    
    # Stage 전환 기준
    stage_transition_step: int = 30000  # Stage 1 -> 2 전환 시점
    
    # 학습 진행 관련 설정
    log_step: int = 20
    eval_step: int = 5
    model_save_step: int = 100
    d_update_freq: int = 1

    # 평가 관련 설정
    eval_fonts: int = 1
    eval_samples: int = 8

# GANConfig를 안전한 전역 클래스로 등록
add_safe_globals([GANConfig])