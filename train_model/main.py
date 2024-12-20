import os
import torch
from pathlib import Path
import pickle
from embedding import generate_font_embeddings, load_embeddings
from torch.utils.data import DataLoader
from dataset import FontDataset
import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from generator import FontGAN
from gan_config import GANConfig
import csv
from datetime import datetime  # datetime.datetime 대신 datetime만 import
from torchvision.utils import save_image
import numpy as np
import os
from PIL import Image

# datach() : 파이토치는 텐서에서 이루어진 모든 연산을 기록해 놓는데 이 연산 기록에서 역전파가 이루어짐
        # detach 함수는 연산 기록에서 역전파를 중단하고 분리한 텐서를 반환한다
# cpu() : GPU 메모리에 올라가 있는 텐서를 CPU 메모리로 복사하는 함수다 numpy로 변환하기 위해선 먼저 해주어야한다.
# uniform(A, B) : 균등 분포 A = min / B = max 
# normal(A, B) : 정규 분포  A = 평균 / B = 분산
# celi(number) : 무조건 소수점 올림

def resume_training(checkpoint_path: str, config: GANConfig, data_dir: str, save_dir: str, device: torch.device):
    """이전 체크포인트에서 학습 재개"""
    print(f"\n=== Resuming training from {checkpoint_path} ===")
    
    # GAN 모델 초기화
    gan = FontGAN(config, device)
    try:
        checkpoint = load_checkpoint(checkpoint_path, device)
        
        # 인코더만 로드하고 고정
        gan.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        for param in gan.encoder.parameters():
            param.requires_grad = False
        # 인코더의 가중치는 평가모드
        gan.encoder.eval()
        gan.fine_tune = True
        
        print("Encoder loaded and frozen for transfer learning")
        
        # 새로운 학습 시작, 전이학습 시작
        return train_font_gan(config, data_dir, save_dir, device, 
                            start_epoch=0,
                            initial_model=gan)
                            
    except Exception as e:
        print(f"Failed to resume training: {e}")
        return None
    
def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """체크포인트를 안전하게 로드"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 필요한 정보만 추출
        safe_checkpoint = {
            'encoder_state_dict': checkpoint['encoder_state_dict'],
            'decoder_state_dict': checkpoint['decoder_state_dict'],
            'discriminator_state_dict': checkpoint['discriminator_state_dict'],
            'g_optimizer_state_dict': checkpoint['g_optimizer_state_dict'],
            'd_optimizer_state_dict': checkpoint['d_optimizer_state_dict'],
            'epoch': checkpoint['epoch'],
            'losses': checkpoint.get('losses', {})
        }
        
        return safe_checkpoint
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

def save_checkpoint(model: FontGAN, epoch: int, losses: dict, save_path: Path):
    """체크포인트를 안전하게 저장"""
    # 모델 상태만 저장
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'g_optimizer_state_dict': model.g_optimizer.state_dict(),
        'd_optimizer_state_dict': model.d_optimizer.state_dict(),
        'losses': losses
    }
    
    try:
        # weights_only 매개변수 없이 저장
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")

# GAN 학습 시작 함수
# 파라미터 설멸
# config : 학습 과정
def train_font_gan(config: GANConfig, data_dir: str, save_dir: str, device: torch.device, 
                  start_epoch: int = 0, initial_model: Optional[FontGAN] = None):
    """
        GAN 학습 함수
        1) data_dir를 통해 학습용, 평가용 데이터 모두 로딩
        2) 학습 기록을 남겨두기 위한 csv파일 생성
        3) 효율적인 학습을 위한 스케줄러 등록
        4) 매배치마다 train_step 메소드 호출
        5) 주기적으로 로그 출력, 체크 포인트 저장, csv 파일 기록, 모델 평가 + best모델 저장

        args:
            config         : GAN 설정 객체
            data_dir       : 데이터셋이 있는 경로
            save_dir       : 학습 결과물이 저장될 경로
            device         : CPU OR GPU
            start_epoch    : 전이 학습하지 않고 이전에 학습하던 모델 이어서 학습할 경우
            initial_model  : 기존 전이학습 모델, FontGAN 객체(가중치 모두 로딩)
    """
    print("\n=== Starting Font GAN Training ===")
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Save directory: {save_dir}")
    print(f"Starting from epoch: {start_epoch}")

    
    
    # 저장 디렉토리 생성
    save_dir = Path(save_dir)
    checkpoint_dir = save_dir / 'checkpoints'
    sample_dir = save_dir / 'samples'
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 폰트 임베딩 로드
    embedding_path = Path("./fixed_dir/EMBEDDINGS.pkl")
    if not embedding_path.exists():
        raise FileNotFoundError(f"Font embeddings not found at {embedding_path}")
    
    font_embeddings = load_embeddings(embedding_path, device)
    
    # 학습용 데이터 로더 설정
    train_dataset = FontDataset(data_dir, "handwritten_train.pkl", img_size = config.img_size)
    # nun_workers = 데이터를 로드할 때 복수개의 프로세스로 멀티 프로세싱을 수행, CPU가 빠르게 데이터를 로딩해서 GPU의 연산 시간 비율을 높이기 위함.
    # pin_meomeory = 메모리의 데이터를 GPU로 옮길 때 시간을 단축시키기 위함.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True, # 각 에포크마다 셔플을 한다.
        num_workers=4,
        pin_memory=True
    )

    # 평가용 데이터로더 설정
    val_dataset = FontDataset(data_dir, "handwritten_val.pkl", img_size = config.img_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # GAN 모델 초기화 또는 기존 모델 사용
    gan = initial_model if initial_model is not None else FontGAN(config, device)

    # 최저 손실값은 기록.
    best_loss = float('inf')

    # CSV 파일 경로 설정
    timestamp = datetime.now().strftime("%m%d-%H%M")
    metrics_dir = Path(save_dir) / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    train_loss_file = metrics_dir / f'training_losses_{timestamp}.csv'
    eval_loss_file = metrics_dir / f'evaluation_metrics_{timestamp}.csv'
    
    # CSV 헤더 작성 -> 에포크, 배치당 손실값을 기록 이는 나중에 모델 돌려놓고 확인하기 위해서 기록을 남겨둠(변화량 확인)
    with open(train_loss_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'batch', 'd_loss', 'g_loss', 'l1_loss', 'const_loss', 'cat_loss'])
        
    with open(eval_loss_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'l1_loss', 'const_loss', 'discriminator_acc', 'font_classification_acc'])

    # 학습률 스케줄러 설정
    # StepLR : step_size를 지정하여 step마다 learngin rate를 gamma씩 감소하는 스케줄러  
    schedulers = {
        'generator': torch.optim.lr_scheduler.StepLR(
            gan.g_optimizer,
            step_size=config.schedule,
            gamma=0.5  # 기존 학습률에 x 0.5
        ),
        'discriminator': torch.optim.lr_scheduler.StepLR(
            gan.d_optimizer,
            step_size=config.schedule,
            gamma=0.5
        )
    }

    # 본격적인 학습 시작. 
    # cofig 설정에 따라 에포크가 결정됨 배치는 당연히 배치 사이지에 따라 바뀜
    for epoch in range(start_epoch, start_epoch + config.max_epoch):
        print(f"\n=== Epoch {epoch+1}/{config.max_epoch + start_epoch} ===")
        epoch_losses = {
            'g_loss': [], 
            'd_loss': [], 
            'l1_loss': [],
            'const_loss': [],
            'cat_loss' : []
        }
        
        # train_loader를 통해 인덱스별로 (원본이미지, 타겟이미지, 폰트 식별 번호)를 추출함
        for batch_idx, (source, target, font_ids) in enumerate(train_loader):
            # 생성자 학습 -> 배치단위로 학습을 진행한다.
            losses = gan.train_step(source, target, font_embeddings, font_ids)
                
            for k, v in losses.items():
                epoch_losses[k].append(v)
            
            if batch_idx % config.log_step == 0:
                log_str = f"Epoch [{epoch+1}/{config.max_epoch + start_epoch}], "
                log_str += f"Batch [{batch_idx+1}/{len(train_loader)}], "
                log_str += ", ".join(f"{k}: {v:.4f}" for k, v in losses.items())
                print(log_str)
                
                # CSV에 손실값 기록
                with open(train_loss_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1,
                        batch_idx + 1,
                        losses['d_loss'],
                        losses['g_loss'],
                        losses['l1_loss'],
                        losses['const_loss'],
                        losses['cat_loss']
                    ])
        
        # 에포크 평균 손실 계산
        avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}

        # 학습률 조정
        schedulers['generator'].step()
        schedulers['discriminator'].step()
        
        def create_checkpoint(gan, epoch, avg_losses, font_embeddings):
            return {
                'epoch': epoch,
                'encoder_state_dict': gan.encoder.state_dict(),
                'decoder_state_dict': gan.decoder.state_dict(),
                'discriminator_state_dict': gan.discriminator.state_dict(),
                'g_optimizer_state_dict': gan.g_optimizer.state_dict(),
                'd_optimizer_state_dict': gan.d_optimizer.state_dict(),
                'config': config,
                'losses': avg_losses,
                'font_embeddings': font_embeddings
            }        
        
        # 현재 체크포인트 생성
        checkpoint = create_checkpoint(gan, epoch, avg_losses, font_embeddings)

        # 주기적으로 평가 수행
        if (epoch + 1) % config.eval_step == 0:
            print(f"\nEvaluating model at epoch {epoch + 1}...")
            metrics = gan.evaluate_metrics(val_loader, font_embeddings)
            
            # 평가 샘플 생성
            eval_dir = Path(save_dir) / 'evaluation'
            gan.generate_evaluation_samples(val_loader, font_embeddings, eval_dir, epoch=epoch)

            # CSV에 평가 지표 기록   
            with open(eval_loss_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    metrics['l1_loss'],
                    metrics['const_loss'],
                    metrics['discriminator_acc'],
                    metrics['font_classification_acc']
                ])

        # 일정 주기로 체크포인트 저장
        if (epoch + 1) % config.model_save_step == 0: 
            timestamp = datetime.now().strftime("%m%d-%H%M")
            save_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}_{timestamp}.pth'
            save_checkpoint(gan, epoch, avg_losses, save_path)
        
        # 최고 성능 모델 저장
        current_loss = avg_losses['g_loss']
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            print(f"Best model saved with loss: {best_loss:.4f}")

    print("Training completed!")
    return gan

def main():
    # Paths
    data_dir = "./dataset"
    save_dir = "./pre_trained_data"
    checkpoint_path = "./final_data/checkpoints/best_model.pth"  # 이전 체크포인트 경로
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 임베딩 벡터가 없으면 새롭게 생성한다, 임베딩 벡터에 대해 자세한 내용은 함수의 주석으로 확인 가능
    if not (Path("./fixed_dir") / "EMBEDDINGS.pkl").exists():
        print("새로운 임베딩 생성")
        generate_font_embeddings(GANConfig.fonts_num, GANConfig.embedding_dim)

    # 체크포인트 경로가 있으면 학습 재개, 없으면 새로 시작
    if os.path.exists(checkpoint_path):
        gan = resume_training(checkpoint_path, GANConfig, data_dir, save_dir, device)
    else:
        print("\nNo checkpoint found. Starting new training...")
        gan = train_font_gan(GANConfig, data_dir, save_dir, device)
        
if __name__ == "__main__":
    main()
