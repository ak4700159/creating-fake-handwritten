import torch.nn as nn  # 이 임포트가 가장 먼저 있어야 합니다
from gan_config import GANConfig
from layer import *
from discriminator import Discriminator
from torchvision.utils import save_image
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime




class FontGAN(nn.Module):
    def __init__(self, config: GANConfig, device: torch.device):
        super().__init__()  # 중요: 부모 클래스 초기화
        self.config = config
        self.device = device
        self.train_step_count = 0
        
        # 모델 구조 초기화
        self.encoder = Encoder(conv_dim=128).to(device)  # 원본 모델과 동일한 구조 유지
        self.decoder = Decoder(
            embedded_dim=1152,  # 인코더 출력(128 * 8) + 임베딩(128)
            conv_dim=128
        ).to(device)
        self.discriminator = Discriminator(category_num=26).to(device)  # 26개 폰트 유지
        
        # 전이 학습을 하게될 경우 인코더의 파라미터는 두고 디코더의 파라미터를 들고 와야된다.
        self.g_optimizer = torch.optim.Adam(
            list(self.decoder.parameters()) + list(self.encoder.parameters()),   
            lr=config.lr * 0.5,  # 더 작은 학습률 사용
            betas=(0.5, 0.999) # 모멘텀
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr * 0.5,
            betas=(0.5, 0.999)
        )

        # Loss functions
        self.l1_loss = nn.L1Loss().to(device)
        self.bce_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

    def eval(self):
        """평가 모드로 전환"""
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()

    # 인코더 학습 여부 = mode
    def train(self, mode=False):
        if mode:
            # 학습 모드에서도 인코더는 eval 모드 유지
            self.encoder.eval()
            self.decoder.train()
            self.discriminator.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.discriminator.eval()
        return self

    def save_samples(self, save_path: str, source: torch.Tensor, 
                    target: torch.Tensor, font_ids: torch.Tensor,
                    font_embeddings: torch.Tensor, num_samples: int = 8):
        """학습 중인 모델의 샘플 이미지를 생성하고 저장합니다.
        
        이 함수는 세 가지 이미지를 나란히 저장합니다:
        1. 원본 고딕체 이미지 (source)
        2. 실제 타겟 손글씨 이미지 (target)
        3. 모델이 생성한 가짜 손글씨 이미지 (fake)
        
        Args:
            save_path (str): 생성된 이미지를 저장할 경로
            source (torch.Tensor): 원본 고딕체 이미지 배치
            target (torch.Tensor): 실제 손글씨 이미지 배치
            font_ids (torch.Tensor): 폰트 ID 배치
            font_embeddings (torch.Tensor): 전체 폰트 임베딩
            num_samples (int, optional): 생성할 샘플 수. 기본값 8
        """
        # 현재 학습/평가 모드 상태 저장
        was_training = self.training
        
        # 평가 모드로 전환 (배치 정규화, 드롭아웃 등 비활성화)
        self.eval()
        
        try:
            with torch.no_grad():  # 메모리 효율성을 위해 그래디언트 계산 비활성화
                # 배치 크기 제한
                source = source[:num_samples]
                target = target[:num_samples]
                font_ids = font_ids[:num_samples]
                
                # 가짜 이미지 생성 과정
                encoded_source, skip_connections = self.encoder(source)
    
                
                # 폰트 임베딩 처리
                embedding = self._get_embeddings(font_embeddings, font_ids)
                embedded = torch.cat([encoded_source, embedding], dim=1)
                
                # 디코더를 통한 가짜 이미지 생성
                fake_target = self.decoder(embedded, skip_connections)
                
                # 이미지 값 범위 변환: [-1, 1] -> [0, 1]
                source = (source + 1) / 2
                target = (target + 1) / 2
                fake_target = (fake_target + 1) / 2
                
                # 시각화를 위한 이미지 그리드 생성
                # 각 행: [원본 고딕체 | 실제 손글씨 | 생성된 손글씨]
                comparison = torch.cat([source, target, fake_target], dim=3)
                
                # 저장 디렉토리 생성
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # 이미지 저장 (normalize=False는 이미 [0,1] 범위로 정규화했기 때문)
                save_image(
                    comparison,
                    save_path,
                    nrow=1,  # 각 행에 하나의 샘플 세트
                    padding=10,  # 이미지 간 여백
                    normalize=False
                )
                
                print(f"샘플 이미지 저장 완료: {save_path}")
                    
        except Exception as e:
            print(f"샘플 저장 중 오류 발생: {str(e)}")
            raise
            
        finally:
            # 이전 학습/평가 모드로 복원
            if was_training:
                self.train()
        
    def train_step(self, real_source, real_target, font_embeddings, font_ids):
        # 학습 단계에 따른 가중치 설정
        if self.train_step_count < self.config.stage_transition_step:
            l1_lambda = self.config.stage1_l1_lambda
            const_lambda = self.config.stage1_const_lambda
        else:
            l1_lambda = self.config.stage2_l1_lambda
            const_lambda = self.config.stage2_const_lambda
    
        
        # Move data to device
        real_source = real_source.to(self.device)
        real_target = real_target.to(self.device)
        font_embeddings = font_embeddings.to(self.device)
        font_ids = font_ids.to(self.device)
        
        # Generate fake image
        encoded_source, skip_connections = self.encoder(real_source)
        embedding = self._get_embeddings(font_embeddings, font_ids)
        embedded = torch.cat([encoded_source, embedding], dim=1)
        fake_target = self.decoder(embedded, skip_connections)
        
        # Train Discriminator
        # 생성한 이미지와 원본 타겟 이미지를 넣어 판별자 판단
        d_real_score, d_real_patch, d_real_cat = self.discriminator(
            torch.cat([real_source, real_target], dim=1)
        )
        d_fake_score, d_fake_patch, d_fake_cat = self.discriminator(
            torch.cat([real_source, fake_target.detach()], dim=1)
        )
        
        # torch.ones_like : 인풋과 동일한 크기를 가지면서 각각의 원소가 1인 텐서 생성

        # Label smoothing
        # 각자 다르게 
        real_labels = torch.ones_like(d_real_patch).to(self.device) * 0.9
        fake_labels = torch.zeros_like(d_fake_patch).to(self.device) * 0.1
        
        # Discriminator losses
        d_loss_real = self.bce_loss(d_real_patch, real_labels)
        d_loss_fake = self.bce_loss(d_fake_patch, fake_labels)
        d_loss_adv = (d_loss_real + d_loss_fake) * 0.5
        d_loss_cat = self.bce_loss(d_real_cat, F.one_hot(font_ids, self.config.fonts_num).float())
        
        d_total_loss = d_loss_adv + d_loss_cat
        
        # Update Discriminator
        self.d_optimizer.zero_grad()
        d_total_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        g_fake_score, g_fake_patch, g_fake_cat = self.discriminator(
           torch.cat([real_source, fake_target], dim=1)
        )
        
        # Generator losses
        g_loss_adv = self.bce_loss(g_fake_patch, torch.ones_like(g_fake_patch)) * self.config.lambda_adv
        g_loss_l1 = self.l1_loss(fake_target, real_target) * l1_lambda 
        g_loss_const = self._consistency_loss(encoded_source, fake_target) * const_lambda
        g_loss_cat = self.bce_loss(g_fake_cat, F.one_hot(font_ids, self.config.fonts_num).float())  * self.config.lambda_cat
        
        g_total_loss = g_loss_adv + g_loss_l1 + g_loss_const + g_loss_cat
        
        # Update Generator
        self.g_optimizer.zero_grad()
        g_total_loss.backward()
        self.g_optimizer.step()
        
        self.train_step_count += 1
        
        return {
            'd_loss': d_total_loss.item(),
            'g_loss': g_total_loss.item(),
            'l1_loss': g_loss_l1.item(),
            'const_loss': g_loss_const.item(),
            'cat_loss': g_loss_cat.item()
        }
    
            
    def _get_embeddings(self, embeddings: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        adjusted_ids = ids - 1
        selected = embeddings[adjusted_ids]  # 이미 [batch_size, 128, 3, 3] 형태
        return selected  # 추가 변환 필요 없음

        
    def _category_loss(self, real_cat: torch.Tensor, fake_cat: torch.Tensor, font_ids: torch.Tensor):
        font_ids = font_ids.to(self.device)
        # 실제 font_ids에 따른 one-hot 인코딩 생성
        real_labels = F.one_hot(font_ids, self.config.fonts_num).float()
        
        real_loss = self.bce_loss(real_cat, real_labels)
        fake_loss = self.bce_loss(fake_cat, real_labels)
        
        return (real_loss + fake_loss) * 0.5
    
    def _consistency_loss(self, encoded_source: torch.Tensor, fake_target: torch.Tensor) -> torch.Tensor:
        """Calculate consistency loss between encoded source and encoded fake"""
        # Encode the generated image
        encoded_fake, _ = self.encoder(fake_target)
        return self.mse_loss(encoded_source, encoded_fake)

    def evaluate_metrics(self, dataer, font_embeddings):
        """모델 성능 평가"""
        self.eval()
        metrics = {
            'l1_loss': [],
            'const_loss': [],
            'discriminator_acc': [],
            'font_classification_acc': []
            # FID score는 복잡한 계산이 필요하므로 일단 제외
        }
        
        try:
            with torch.no_grad():
                # 데이터 로더가 비어있는지 확인
                if len(dataer) == 0:
                    raise ValueError("Dataer is empty")
                    
                for batch_idx, (source, target, font_ids) in enumerate(dataer):
                    if batch_idx >= 100:  # 평가할 배치 수 제한
                        break
                        
                    source = source.to(self.device)
                    target = target.to(self.device)
                    font_ids = font_ids.to(self.device)
                    
                    # 가짜 이미지 생성
                    encoded_source, skip_connections = self.encoder(source)
                    embedding = self._get_embeddings(font_embeddings, font_ids)
                    embedded = torch.cat([encoded_source, embedding], dim=1)
                    fake_target = self.decoder(embedded, skip_connections)
                    
                    # L1 Loss 계산
                    l1_loss = self.l1_loss(fake_target, target)
                    metrics['l1_loss'].append(l1_loss.item())
                    
                    # Consistency Loss 계산
                    const_loss = self._consistency_loss(encoded_source, fake_target)
                    metrics['const_loss'].append(const_loss.item())
                    
                    # Discriminator 정확도 계산
                    real_score, _, real_cat = self.discriminator(torch.cat([source, target], dim=1))
                    fake_score, _, fake_cat = self.discriminator(torch.cat([source, fake_target], dim=1))
                    
                    disc_acc = ((real_score > 0.5).float().mean() + 
                            (fake_score < 0.5).float().mean()) / 2
                    metrics['discriminator_acc'].append(disc_acc.item())
                    
                    # 폰트 분류 정확도 계산
                    font_labels = F.one_hot(font_ids, self.config.fonts_num).float()
                    font_acc = (torch.argmax(real_cat, dim=1) == font_ids).float().mean()
                    metrics['font_classification_acc'].append(font_acc.item())
                    
                # 각 메트릭이 비어있지 않은지 확인
                for k, v in metrics.items():
                    if not v:
                        print(f"Warning: No values collected for metric {k}")
                        metrics[k] = [0.0]  # 기본값 설정
                
                # 평균 계산
                avg_metrics = {k: sum(v)/len(v) for k, v in metrics.items()}
                
                print("\nEvaluation Metrics:")
                print(f"L1 Loss (픽셀 유사도): {avg_metrics['l1_loss']:.4f}")
                print(f"Consistency Loss (특징 보존): {avg_metrics['const_loss']:.4f}")
                print(f"Discriminator Accuracy: {avg_metrics['discriminator_acc']:.4f}")
                print(f"Font Classification Accuracy: {avg_metrics['font_classification_acc']:.4f}")
                
                return avg_metrics
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {k: 0.0 for k in metrics.keys()}  # 오류 시 기본값 반환
            
        finally:
            self.train()

    def generate_evaluation_samples(self, dataer, font_embeddings, save_dir: Path):
        self.eval()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 전체 데이터셋에서 서로 다른 폰트를 가진 샘플 수집
        unique_samples = {}
        
        with torch.no_grad():
            for source, target, font_ids in dataer:
                for i, font_id in enumerate(font_ids):
                    font_id = font_id.item()
                    if font_id not in unique_samples and len(unique_samples) < 10:
                        unique_samples[font_id] = (source[i:i+1], target[i:i+1])
                    if len(unique_samples) == 10:
                        break
                if len(unique_samples) == 10:
                    break
            
            # 수집된 샘플로 이미지 생성
            for idx, (font_id, (source, target)) in enumerate(unique_samples.items()):
                source = source.to(self.device)
                target = target.to(self.device)
                font_id = torch.tensor([font_id], device=self.device)
                
                # encoded_source, skip_connections = self.encoder(source)
                # embedding = self._get_embeddings(font_embeddings, font_id)
                # embedded = torch.cat([encoded_source, embedding], dim=1)
                # fake_target = self.decoder(embedded, skip_connections)
                
                self.save_samples(
                    save_dir / f'eval_sample_font_{font_id.item()}.png',
                    source,
                    target,
                    font_id,
                    font_embeddings
                )