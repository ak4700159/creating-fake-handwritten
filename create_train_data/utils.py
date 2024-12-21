# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from PIL import Image


def normalize_image(img):
    """
        이미지의 형태는 [W,H,C]

        각 픽셀의 값은 0 ~ 255 까지의 값을 가진다 -> -1 ~ 1 사이의 값을 가지도록 변환

        이는 나중에 활성화 함수에서 최적으로 작동된다.
    """
    normalized = (img / 127.5) - 1.
    return normalized
  
    
def tight_crop_image(img, resize_fix=False):
    """
        해당 함수에선 이미지에 대해 crop -> resize 까지 진행
    """
    # 입력 이미지의 높이를 가져옴 (정사각형 이미지 가정)
    img_size = img.shape[0]

    # 전체 흰색 값을 img_size로 설정
    full_white = img_size

    # axis=0은 열 방향으로 합을 계산
    # full_white - np.sum(img, axis=0)는 각 열의 '비어있는 정도'를 계산
    # np.where는 조건이 참인 인덱스를 반환
    col_sum = np.where(full_white - np.sum(img, axis=0) > 1)

    # axis=1은 행 방향으로 합을 계산
    # 각 행의 '비어있는 정도'를 계산
    row_sum = np.where(full_white - np.sum(img, axis=1) > 1)

    # 행 방향의 첫 번째와 마지막 비어있지 않은 픽셀의 인덱스
    y1, y2 = row_sum[0][0], row_sum[0][-1]

    # 열 방향의 첫 번째와 마지막 비어있지 않은 픽셀의 인덱스
    x1, x2 = col_sum[0][0], col_sum[0][-1]

    # 실제 글자 부분만 잘라냄
    cropped_image = img[y1:y2, x1:x2]

    # 잘라낸 이미지의 크기 저장
    cropped_image_size = cropped_image.shape

    # resize_fix가 정수인 경우
    if type(resize_fix) == int:
        origin_h, origin_w = cropped_image.shape
        
        # 높이가 너비보다 큰 경우
        if origin_h > origin_w:
            # 비율을 유지하면서 높이를 resize_fix로 조정
            resize_w = int(origin_w * (resize_fix / origin_h))
            resize_h = resize_fix
        else:
            # 비율을 유지하면서 너비를 resize_fix로 조정
            resize_h = int(origin_h * (resize_fix / origin_w))
            resize_w = resize_fix

        # numpy 배열을 PIL Image로 변환
        if isinstance(cropped_image, np.ndarray):
            img_pil = Image.fromarray(cropped_image.astype(np.uint8))
        else:
            img_pil = cropped_image
            
        # 이미지 크기 조정(resize_w * resize_h) 후 다시 numpy 배열로 변환
        cropped_image = np.array(img_pil.resize((resize_w, resize_h), Image.Resampling.LANCZOS))
        # 정규화 (-1 ~ 1 범위로 변환)
        cropped_image = normalize_image(cropped_image)
        cropped_image_size = cropped_image.shape

    # resize_fix가 실수인 경우
    elif type(resize_fix) == float:
        origin_h, origin_w = cropped_image.shape
        # 원본 크기에 resize_fix를 곱해서 새 크기 계산
        resize_h, resize_w = int(origin_h * resize_fix), int(origin_w * resize_fix)
        
        # 최대 크기 제한 (120 픽셀)
        if resize_h > 120:
            resize_h = 120
            resize_w = int(resize_w * 120 / resize_h)
        if resize_w > 120:
            resize_w = 120
            resize_h = int(resize_h * 120 / resize_w)

        # numpy 배열을 PIL Image로 변환하고 크기 조정
        if isinstance(cropped_image, np.ndarray):
            img_pil = Image.fromarray(cropped_image.astype(np.uint8))
        else:
            img_pil = cropped_image
        cropped_image = np.array(img_pil.resize((resize_w, resize_h), Image.Resampling.LANCZOS))
        cropped_image = normalize_image(cropped_image)
    return cropped_image


def add_padding(img, image_size=128, pad_value=None):
    """"
        crop -> resize 된 이미지에 대해 128 * 128 이미지로 padding
    """
    height, width = img.shape
    # 별도의 패딩 색상을 지정해주지 않으면 좌측 상단의 첫번째 픽셀을 색상으로 지정
    if not pad_value: 
        pad_value = img[0][0]
    
    # 너비 패딩
    pad_x_width = (image_size - width) // 2
    pad_x = np.full((height, pad_x_width), pad_value, dtype=np.float32)
    img = np.concatenate((pad_x, img), axis=1)
    img = np.concatenate((img, pad_x), axis=1)
    
    width = img.shape[1]

    # 높이 패딩
    pad_y_height = (image_size - height) // 2
    pad_y = np.full((pad_y_height, width), pad_value, dtype=np.float32)
    img = np.concatenate((pad_y, img), axis=0)
    img = np.concatenate((img, pad_y), axis=0)
    
    # Match to original image size
    width = img.shape[1]
    if img.shape[0] % 2:
        pad = np.full((1, width), pad_value, dtype=np.float32)
        img = np.concatenate((pad, img), axis=0)
    height = img.shape[0]
    if img.shape[1] % 2:
        pad = np.full((height, 1), pad_value, dtype=np.float32)
        img = np.concatenate((pad, img), axis=1)
    
    return img

def centering_image(img, image_size=128, resize_fix=90, pad_value=None): 
    """
    이미지 전처리 : crop -> resize -> padding

    args 
        img        : numpy 형태의 원본 img 데이터
        image_size : 출력될 이미지 크기
        resize_fix : 이미지 crop 이후 비율을 유지한 상태에서 이미지 크기
        pad_value  : 글자와 모서리 간 여백 색상값 0 ~ 255 (지정하지 않으면 여백은 횐색)
    """
    if not pad_value:
        pad_value = img[0][0]
    cropped_image = tight_crop_image(img, resize_fix=resize_fix)
    centered_image = add_padding(cropped_image, image_size=image_size, pad_value=pad_value)
    
    return centered_image


def round_function(i):
    if i < -0.95:
        return -1
    elif i > 0.95:
        return 1
    else:
        return i