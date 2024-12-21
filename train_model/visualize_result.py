import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize_training_losses(csv_path):
    # CSV 파일 읽기
    data = pd.read_csv(csv_path, header=0)
    
    # x축 값 생성: epoch + batch/100 형태로 만들기
    data['x_axis'] = (data['epoch'] - 1) * 48 + data['batch']
    
    # Seaborn 스타일 설정
    sns.set_style("darkgrid")
    plt.figure(figsize=(15, 10))
    
    # 서브플롯 생성
    plt.subplot(2, 1, 1)
    plt.plot(data['x_axis'], data['g_loss'], label='Generator Loss', color='blue', alpha=0.7)
    plt.plot(data['x_axis'], data['d_loss'], label='Discriminator Loss', color='red', alpha=0.7)
    plt.title('Generator and Discriminator Losses over batch')
    plt.xlabel('batch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.ylim(0, 100)
    
    plt.subplot(2, 1, 2)
    plt.plot(data['x_axis'], data['l1_loss'], label='L1 Loss', color='green', alpha=0.7)
    plt.plot(data['x_axis'], data['const_loss'], label='Const Loss', color='purple', alpha=0.7)
    plt.plot(data['x_axis'], data['cat_loss'], label='Category Loss', color='orange', alpha=0.7)
    plt.title('Component Losses over batch')
    plt.xlabel('batch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.ylim(0, 15)
    
    plt.tight_layout()
    plt.show()


csv_path = "./pre_trained_data/metrics/training_losses_1221-2326.csv"  # CSV 파일 경로
visualize_training_losses(csv_path)  # 전체 학습 과정 시각화
