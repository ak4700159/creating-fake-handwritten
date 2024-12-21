# 폰트 파일명 변경.

import os
import re

def decrease_file_numbers(directory='.'):
    # 디렉토리 내의 모든 파일 목록을 가져옴
    files = os.listdir(directory)
    
    # 번호를 포함한 파일만 필터링하고 정렬
    number_pattern = re.compile(r'(\d{2})')
    numbered_files = [f for f in files if number_pattern.search(f)]
    numbered_files.sort()
    
    print("변경 전 파일 목록:", numbered_files)
    
    # 각 파일의 번호를 1씩 감소
    for filename in numbered_files:
        # 파일명에서 2자리 숫자를 찾음
        match = number_pattern.search(filename)
        if match:
            old_num = match.group(1)
            new_num = f"{int(old_num) - 1:02d}"  # 1 감소시키고 2자리 숫자로 포맷팅
            
            # 새 파일명 생성
            new_filename = filename.replace(old_num, new_num)
            
            # 파일 이름 변경
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            try:
                os.rename(old_path, new_path)
                print(f"변경: {filename} -> {new_filename}")
            except OSError as e:
                print(f"에러 발생 ({filename}): {e}")

    print("\n변경이 완료되었습니다.")

# 함수 실행
if __name__ == "__main__":
    # 현재 디렉토리에서 실행
    decrease_file_numbers('./fonts/target')
