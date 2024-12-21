from PIL import ImageFont
import os
import random
import glob
import pickle as pickle
import threading
from PIL import Image, ImageDraw
import numpy as np   
from utils import centering_image

# 데이터가 저장될 경로
FONT_DATASET_PATH = "./font_dataset"
# 몇 개의 폰트를 이용할 것인지
MAX_FONT_COUNT = 25
# 하나의 폰트 당 몇 개의 데이터를 생성할지
MAX_RAMDOM_SELECTED_WORD = 2000
# 타겟 폰트가 있는 경로(디렉토리)
trg_font_pth = './fonts/target'
# 소스 폰트(고딕체)가 있는 경로
src_font_pth = './fonts/source/source_font.ttf'

def pickle_examples(from_dir, train_path, val_path, train_val_split=0.2, with_charid=False):
    """
    .pkl 파일로 변환해주는 함수
    args:
        from_dir :          데이터가 들어있는 디렉토리
        train_path :        훈련용 데이터 저장 경로
        val_path :          평가용 데이터 저장 경로
        train_val_splite :  훈련용/평가용 데이터 비율
        with_charid :       파일명에 글자 식별 번호 사용 여부
    """

    # glob 패키지를 이용해 훈련용, 평가용 데이터가 들어가 있는 디렉토리의 모든 파일명을 로딩한다.
    paths = glob.glob(os.path.join(from_dir, "*.png"))
    # pickle 패키지를 이용해 훈련용, 평가용 파일에 데이터 쓰기
    with open(train_path, 'wb') as ft:
        with open(val_path, 'wb') as fv:
            print('전체 데이터 갯수:', len(paths))
            c = 1
            val_count = 0
            train_count = 0
            # 파일명 안에 글자 식별 번호가 포함된 경우
            if with_charid:
                # paths 경로 안에는 font_dataset 안의 파일 이름 모두 담기게 된다
                for p in paths:
                    c += 1
                    # [fontID]_[charID].png
                    label = int(os.path.basename(p).split("_")[0])
                    charid = int(os.path.basename(p).split("_")[1].split(".")[0])
                    # .png 파일을 열고 이미지(바이너리 데이터)를 읽고 랜덤함수를 통해 데이터 분류(훈련용, 평가용)
                    with open(p, 'rb') as f:
                        img_bytes = f.read()
                        example = (label, charid, img_bytes)
                        r = random.random()
                        # 설정한 비율대로 train val 파일에 저장된다.
                        if r < train_val_split:
                            pickle.dump(example, fv)
                            val_count += 1
                        else:
                            pickle.dump(example, ft)
                            train_count += 1
                print("%d 개의 이미지가 저장되었습니다. %s.pkl" % val_count, val_path)
                print("%d 개의 이미지가 저장되었습니다. %s.pkl" % train_count, train_path)
            # 파일명 안에 글자 식별 번호가 포함되지 않은 경우
            else:
                for p in paths:
                    c += 1
                    # [fontID]_.png
                    label = int(os.path.basename(p).split("_")[0])
                    with open(p, 'rb') as f:
                        img_bytes = f.read()
                        example = (label, img_bytes)
                        r = random.random()
                        if r < train_val_split:
                            pickle.dump(example, fv)
                            val_count += 1
                        else:
                            pickle.dump(example, ft)
                            train_count += 1
                print("%d 개의 이미지가 저장되었습니다. %s.pkl" % val_count, val_path)
                print("%d 개의 이미지가 저장되었습니다. %s.pkl" % train_count, train_path)
            return

def generate_random_hangul_and_ascii():
    """
        한글 유니코드 내에서 랜덤한 한글 문자 1개와 그 문자의 아스키 코드를 생성
    """
    start = 0xAC00  # 가
    end = 0xD7A3  # 힣
    
    # 랜덤한 한글 유니코드 선택
    char_code = random.randint(start, end)
    
    # 유니코드를 문자로 변환
    hangul_char = chr(char_code)

    return hangul_char

def generated_dataset(font_id):
    """
    MAX_RAMDOM_SELECTED_WORD 만큼 [source, target] 이미지를 생성

    source = 고딕체 글자 , target = 특정 폰트 적용 글자 

    args : 
        font_id : 생성할 폰트 번호
    """
    if not os.path.exists(FONT_DATASET_PATH):
        os.mkdir(FONT_DATASET_PATH)

    # 사용된 문자가 저장된다. 폰트가 바뀔때마다 리셋
    ch_list = set()
    count = 0
    while True:
        if(count >= MAX_RAMDOM_SELECTED_WORD) : break
        ch = generate_random_hangul_and_ascii()
        if ch in ch_list : continue

        if font_id < 10 :
            trg_font_pth = trg_font_pth + "0" + str(font_id+1) + ".ttf"
        else :
            trg_font_pth = f"{trg_font_pth}{font_id+1}.ttf"
        
        # 폰트 객체 생성, 두번재 파라미터는 폰트 크기를 의미
        trg_font = ImageFont.truetype(trg_font_pth, 90)
        src_font = ImageFont.truetype(src_font_pth, 90)

        example_img = draw_example(ch, src_font, trg_font, 128)
        if example_img == None: continue

        # 이미지가 저장될 때 사용된 폰트 번호 _ 식별할 수 있는 문자값
        example_img.save(f"{FONT_DATASET_PATH}/{font_id}_{ord(ch)}.png", 'png', optimize=True)
        ch_list.add(ch)
        count += 1
    print(f'폰트 {font_id} 생성')

def draw_single_char(ch, font, canvas_size):
    # L = 흑백이미지를 의미 
    # 단순 128 * 128 사이즈의 흑백 이미지 생성
    image = Image.new('L', (canvas_size, canvas_size), color=255)
    # 흑백 이미지 그리기
    drawing = ImageDraw.Draw(image)

    _, _, w, h = font.getbbox(ch)
    drawing.text(
        ((canvas_size-w)/2, (canvas_size-h)/2),
        ch,
        fill=(0),
        font=font
    )
    flag = np.sum(np.array(image))
    
    # 해당 font에 글자가 없으면 return None
    # 즉 이때 흰색을 의미함.
    if flag == 255 * 128 * 128:
        return None
    
    return image

def draw_example(ch, src_font, dst_font, canvas_size):
    # 특정 스타일로 만들어낸 단일 글자 생성(target image)
    dst_img = draw_single_char(ch, dst_font, canvas_size)
    # 해당 이미지에 글자가 없으면 return None
    if not dst_img:
        return None
    # 열과 행을 슬라이싱 후 crop -> resize -> padding 과정을 거치게 된다.
    dst_img = centering_image(np.array(dst_img), pad_value = 255)
    dst_img = Image.fromarray(dst_img.astype(np.uint8))
    
    # 고딕체 스타일로 만들어낸 단일 글자 생성(source image)
    src_img = draw_single_char(ch, src_font, canvas_size)
    # 해당 이미지에 글자가 없으면 return None
    if not src_img:
        return None
    src_img = centering_image(np.array(src_img))
    src_img = Image.fromarray(src_img.astype(np.uint8))

    # 이미지 합치기
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255)).convert('L')
    # 왼쪽엔 특정 폰트로 만들어낸 단일 글자, 오른쪽엔 고딕체로 만들어낸 단일 글자
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))   
    
    return example_img

def main():
    threads = [] 
    # 멀티 스레드를 이용해 프로그램이 뻗기 전 모든 글자를 뽑아낸다
    for font_idx in range(0, MAX_FONT_COUNT):
        t = threading.Thread(target=generated_dataset, args=(font_idx,))
        t.start()
        threads.append(t)

    for font_idx in range(MAX_FONT_COUNT):
        threads[font_idx].join()   
        
if __name__ == "__main__":
    main()
    # 생성한 학습용 데이터를 train / value 데이터로 나누어 저장.
    pickle_examples('./handwritten_result', '../dataset/handwritten_train.pkl', '../dataset/handwritten_val.pkl', with_charid=True)

