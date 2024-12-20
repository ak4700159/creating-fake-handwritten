# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import scipy.misc
import numpy as np
from io import BytesIO
from PIL import Image

import matplotlib.pyplot as plt

def pad_seq(seq, batch_size):
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq


def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)


def normalize_image(img):
    """
        이미지의 형태는 [W,H,C]

        각 픽셀의 값은 0 ~ 255 까지의 값을 가진다 -> -1 ~ 1 사이의 값을 가지도록 변환

        이는 나중에 활성화 함수에서 최적으로 작동된다.
    """
    normalized = (img / 127.5) - 1.
    return normalized


def read_split_image(img):
    """"
        타겟이미지와 소스이미지를 나눈다.

        [h=128 * w=256] 이미지 파일을 반으로 갈라서 반환

        왼쪽이 source(고딕체글자), 오른쪽이 target(폰트별 글자)
    """
    mat = np.array(Image.open(img)).astype(np.float32)
    # shape[0] = height , shape[1] = weight
    side = int(mat.shape[1] / 2)
    assert side * 2 == mat.shape[1]
    img_A = mat[:, :side]  # 0 ~ side / target
    img_B = mat[:, side:]  # side ~ 전체 / source

    return img_A, img_B


def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    w, h = img.shape
    if isinstance(img, np.ndarray):
        # numpy 배열을 PIL Image로 변환
        img_pil = Image.fromarray(img)
    else:
        img_pil = img
    # 이미지를 쉬프트하고 다시 원본 크기 만큼 잘라내어 반환.
    enlarged = np.array(img_pil.resize((nh, nw), Image.Resampling.LANCZOS))
    return enlarged[shift_x:shift_x + w, shift_y:shift_y + h]
  
    
def tight_crop_image(img, verbose=False, resize_fix=False):
    """
        해당 함수에선 이미지에 대해 shift resize 
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
            
        # 이미지 크기 조정 후 다시 numpy 배열로 변환
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
        cropped_image_size = cropped_image.shape

    return cropped_image


def add_padding(img, image_size=128, verbose=False, pad_value=None):
    height, width = img.shape
    if not pad_value:
        pad_value = img[0][0]
    if verbose:
        print('original cropped image size:', img.shape)
    
    # Adding padding of x axis - left, right
    pad_x_width = (image_size - width) // 2
    pad_x = np.full((height, pad_x_width), pad_value, dtype=np.float32)
    img = np.concatenate((pad_x, img), axis=1)
    img = np.concatenate((img, pad_x), axis=1)
    
    width = img.shape[1]

    # Adding padding of y axis - top, bottom
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

    if verbose:
        print('final image size:', img.shape)
    
    return img

def centering_image(img, image_size=128, verbose=False, resize_fix=False, pad_value=None):
    if not pad_value:
        pad_value = img[0][0]
    cropped_image = tight_crop_image(img, verbose=verbose, resize_fix=resize_fix)
    centered_image = add_padding(cropped_image, image_size=image_size, verbose=verbose, pad_value=pad_value)
    
    return centered_image


def round_function(i):
    if i < -0.95:
        return -1
    elif i > 0.95:
        return 1
    else:
        return i