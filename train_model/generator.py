import torch.nn as nn  # 이 임포트가 가장 먼저 있어야 합니다
from gan_config import GANConfig
from layer import *
from discriminator import Discriminator
from torchvision.utils import save_image
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

# Optimizer : Optimization(최적화)를 수행하는 알고리즘을 말함.
#       --> 모델의 가중치와 편향을 업데이트하여 손실함수를 최소화하는 과정을 최적화
# 수업 시간에서 배웠듯이 SGD, GD, Adam ... 등 여러 알고리즘이 존재
# 여기선 Adam을 사용, 경사 하강법 알고리즘을 기반으로 현재 그래디언트와 이전 그래디언트을 고려해 가중치 업데이트

class FontGAN(nn.Module):
    def __init__(self, config: GANConfig, device: torch.device, fine_tune:bool = False):
        super().__init__()  # 중요: 부모 클래스 초기화
        self.config = config
        self.device = device
        self.train_step_count = 0
        
        # 모델 구조 초기화, 생성자(인코더 + 디코더)와 디코더로 이루어져 있다.
        self.encoder = Encoder(conv_dim=128).to(device)  # 원본 모델과 동일한 구조 유지
        self.decoder = Decoder(
            embedded_dim=1152,  # 인코더 출력(128 * 8 * (3 * 3)) + 임베딩(128 * (3 * 3))
            conv_dim=128
        ).to(device)
        self.discriminator = Discriminator(category_num=26).to(device)  # 26개 폰트(category_num)
        
        # 전이 학습을 하게될 경우 인코더의 파라미터는 얼리고 디코더의 파라미터를 들고 와야된다
        if fine_tune:
            self.g_optimizer = torch.optim.Adam(
                list(self.decoder.parameters()),   
                lr=config.lr,
                # 이전 그래디언트의 이동 평균(첫 번째 매개변수)과 
                # 그래디언트 제곱의 이동 평균(두 번째 매개변수)에 대한 가중치를 제어
                betas=(0.5, 0.999)
            )
        else:
            self.g_optimizer = torch.optim.Adam(
                list(self.decoder.parameters()) + list(self.encoder.parameters()),   
                lr=config.lr,
                betas=(0.5, 0.999)
            )
        # 판별자의 최적화 함수도 생성자와 동일하다
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr,
            betas=(0.5, 0.999)
        )

        # 사용될 손실함수 정의, 이 손실함수들을 이용해 새로운 목적 함수를 만든다.
        # L1Loss            : 평균 절대 오차를 계산하는 손실함수
        # MSELoss           : 평균 제곱 오차(MSE)를 계산하는 손실함수 
        # BCEwithLogitsLoss : nn.BCELoss에 Sigmoid함수가 포함된 형태여서 따로 활성화함수 사용 X
        # BCELoss           : 이진 교차 엔트로피(Binary Cross Entropy) 손실 함수, 예측값과 타겟값 사이의 교차 엔트로피 계산
        self.l1_loss = nn.L1Loss().to(device)
        self.bce_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

    def _get_embeddings(self, embeddings: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # adjusted_ids.shape = [batch_size]  # 예: [16]
        # 모든 값이 4 -> 데이터가 전부 5번 폰트일 때 : tensor([4, 4, 4, ..., 4])
        adjusted_ids = ids - 1               # 폰트 식별 번호로 인덱싱하기 위해 0부터 시작
        selected = embeddings[adjusted_ids]  # [batch_size, 128, 3, 3] 형태
        return selected                   
    
    def _adversial_loss(self, d_real_patch: torch.Tensor, d_fake_patch: torch.Tensor) -> torch.Tensor:
        """
            adv loss = 판별자가 생성된 가짜 이미지와 원본 target이미지를 판단하는 손실값
            판별자는 가짜 이미지는 0에 가깝게 target 이미지는 1에 가깝게 학습

            Label smoothing을 통해 모델의 과잉 확신을 방지하고 라벨 간 클러스터링이 더욱 밀집된 결과를 얻을 수 있게됨
        """
        real_labels = torch.ones_like(d_real_patch).to(self.device) * 0.9
        fake_labels = torch.zeros_like(d_fake_patch).to(self.device) * 0.1
        d_loss_real = self.bce_loss(d_real_patch, real_labels)
        d_loss_fake = self.bce_loss(d_fake_patch, fake_labels)
        return (d_loss_real + d_loss_fake) * 0.5
    
    def _consistency_loss(self, encoded_source: torch.Tensor, fake_target: torch.Tensor) -> torch.Tensor:
        """
            const loss = 생성된 이미지(fake_target)를 인코더에 통과시킨 후 만들어진 z벡터와
            고딕체 이미지를 인코더에 통과시킨 후 만들어진 z벡터(encoded_source) 간의 손실값
            해당 손실값을  통해 원래 글자의 형태를 잃지 않고 유지하게 된다
        """
        # Encode the generated image
        encoded_fake, _ = self.encoder(fake_target)
        return self.mse_loss(encoded_source, encoded_fake)
    
    def eval(self):
        """평가 모드로 전환"""
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()

    # 인코더 학습 여부 = mode
    def train(self, fine_tune=False):
        """학습 모드로 전환"""
        if fine_tune:
            # 학습 모드에서도 인코더는 eval 모드 유지
            self.encoder.eval()
            self.decoder.train()
            self.discriminator.train()
        else:
            self.encoder.train()
            self.decoder.train()
            self.discriminator.train()
        return self

    def save_samples(self, save_path: str, source: torch.Tensor, 
                    target: torch.Tensor, font_ids: torch.Tensor,
                    font_embeddings: torch.Tensor, num_samples: int = 8):
        """학습 중인 모델의 샘플 이미지를 생성하고 저장하는 함수
        
        이 함수는 세 가지 이미지를 가로로 저장
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
        """배치별 학습을 진행하는 함수
        
        Args:
            real_source (torch.Tensor): 원본 고딕체 이미지 배치
            real_target (torch.Tensor): 실제 손글씨 이미지 배치
            font_ids (torch.Tensor): 폰트 ID 배치
            font_embeddings (torch.Tensor): 전체 폰트 임베딩
        """
        # 학습 단계에 따른 가중치 설정
        if self.train_step_count < self.config.stage_transition_step:
            l1_lambda = self.config.stage1_l1_lambda
            const_lambda = self.config.stage1_const_lambda
        else:
            l1_lambda = self.config.stage2_l1_lambda
            const_lambda = self.config.stage2_const_lambda
    
        # 설정된 디바이스로 텐서를 옮긴다
        real_source = real_source.to(self.device)
        real_target = real_target.to(self.device)
        font_embeddings = font_embeddings.to(self.device)
        font_ids = font_ids.to(self.device)
        
        # 인코더에 고딕체 이미지를 입력 -> 글자 특징 백터(Z벡터)와 손실된 차원에 대한 백터 출력
        encoded_source, skip_connections = self.encoder(real_source)
        # font_ids 인덱스에 맞게 전체 임베딩 벡터에서 출력(embedding)
        embedding = self._get_embeddings(font_embeddings, font_ids)
        # 출력된 임베딩 벡터는 인코더에서 출력된 Z벡터와 결합
        embedded = torch.cat([encoded_source, embedding], dim=1)
        # 디코더에 인코딩된 벡터와 손실된 차원에 대한 벡터를 입력값으로 넣어 128 * 128 사이즈의 가짜 데이터를 출력
        fake_target = self.decoder(embedded, skip_connections)

        # =========================================================================================== #
        # 판별자 학습
        # [고딕체 글자 + 폰트에 따른 원본 글자] 판별
        d_real_score, d_real_patch, d_real_cat = self.discriminator(
            torch.cat([real_source, real_target], dim=1)
        )
        # [고딕체 글자 + 생성한 가짜 글자] 판별
        d_fake_score, d_fake_patch, d_fake_cat = self.discriminator(
            torch.cat([real_source, fake_target.detach()], dim=1)
        )
        
        # torch.ones_like : input과 동일한 크기를 가지면서 각각의 원소가 1인 텐서 생성
        # torch.zeros_like : input과 동일한 크기를 가지면서 각각의 원소가 0인 텐서 생성

        # 판별자의 losses
        # 1) d_loss_adv : 판별자가 제대로 이미지를 분류할 수 있도록 하기위한 손실값, 실제 이미지는 1에 가깝게, 가짜 이미지는 0이 나오도록 학습
        d_loss_adv = _adversial_loss(d_real_patch, d_fake_patch)
        # 2) d_loss_cat : 원본 target 데이터의 카테고리에 대한 손실값, 원본 target 데이터터를 더 잘 분류하기 위함
        d_loss_cat = self.bce_loss(d_real_cat, F.one_hot(font_ids, self.config.fonts_num).float())
        d_total_loss = d_loss_adv + d_loss_cat
        
        # 판별자 가중치 조정
        # zero_grad : 파이토치에선 gradients값들을 추후에 backward를 해줄때 계속 더해주기 때문에 gradients값을 0으로 세팅
        # backward : 가중치 계산(역전파)
        # step : 가중치 업데이트
        self.d_optimizer.zero_grad()
        d_total_loss.backward()
        self.d_optimizer.step()
        
        # =========================================================================================== #
        # 생성자 학습
        g_fake_score, g_fake_patch, g_fake_cat = self.discriminator(
           torch.cat([real_source, fake_target], dim=1)
        )
        
        # 생성자의 losses, 생성자 손실값에 각 각의 가중치를 부여
        # 1) g_loss_adv : 생성자가 판별자를 속이기 위한 손실값, 가짜 이미지 텐소의 스코어가 1에 가깝도록 학습 
        g_loss_adv = self.bce_loss(g_fake_patch, torch.ones_like(g_fake_patch)) * self.config.lambda_adv
        # 2) l1_loss : 생성된 이미지와 기존 폰트 글자와의 차이(pixel by pixel)  
        g_loss_l1 = self.l1_loss(fake_target, real_target) * l1_lambda
        # 3) const_loss : 기존 글자 특징 유지 
        g_loss_const = self._consistency_loss(encoded_source, fake_target) * const_lambda
        # 4) category_loss : 생성한 가짜 데이터의 카테고리에 대한 손실값, 생성한 가짜 데이터를 더 잘 분류하기 위함
        g_loss_cat = self.bce_loss(g_fake_cat, F.one_hot(font_ids, self.config.fonts_num).float())  * self.config.lambda_cat
        g_total_loss = g_loss_adv + g_loss_l1 + g_loss_const + g_loss_cat
        
        # 생성자 가중치 조정
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
    

    def evaluate_metrics(self, dataer, font_embeddings):
        """모델 성능 평가 함수"""
        self.eval()
        metrics = {
            'l1_loss': [],
            'const_loss': [],
            'discriminator_acc': [],
            'font_classification_acc': []
        }
        
        try:
            with torch.no_grad():
                # 데이터 로더가 비어있는지 확인
                if len(dataer) == 0:
                    raise ValueError("ValDataer is empty")
                    
                for batch_idx, (source, target, font_ids) in enumerate(dataer):
                    source = source.to(self.device)
                    target = target.to(self.device)
                    font_ids = font_ids.to(self.device)
                    
                    # 가짜 이미지 생성
                    encoded_source, skip_connections = self.encoder(source)
                    embedding = self._get_embeddings(font_embeddings, font_ids)
                    embedded = torch.cat([encoded_source, embedding], dim=1)
                    fake_target = self.decoder(embedded, skip_connections)
                    
                    # 생성자 L1 Loss 계산
                    l1_loss = self.l1_loss(fake_target, target)
                    metrics['l1_loss'].append(l1_loss.item())
                    
                    # 생성자 Consistency Loss 계산
                    const_loss = self._consistency_loss(encoded_source, fake_target)
                    metrics['const_loss'].append(const_loss.item())
                    
                    # 판별자 정확도 계산
                    real_score, _, real_cat = self.discriminator(torch.cat([source, target], dim=1))
                    fake_score, _, fake_cat = self.discriminator(torch.cat([source, fake_target], dim=1))
                    
                    disc_acc = ((real_score > 0.5).float().mean() + 
                            (fake_score < 0.5).float().mean()) / 2
                    metrics['discriminator_acc'].append(disc_acc.item())
                    
                    # 판별자 폰트 분류 정확도 계산
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
            # save_samples 함수에서 모델을 통해 이미지 생성
            self.save_samples(
                save_dir / f'eval_sample_font_{font_id.item()}.png',
                source,
                target,
                font_id,
                font_embeddings
            )