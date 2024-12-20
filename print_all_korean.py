def generate_and_print_korean_syllables():
    # 초성 리스트
    cho = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    
    # 중성 리스트
    jung = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    
    # 종성 리스트 (공백 포함)
    jong = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    
    # 유니코드 기준 값
    base = 0xAC00
    
    # 모든 가능한 조합 생성과 출력
    total = 0
    result = []
    
    for c in cho:
        for j in jung:
            row = []
            for g in jong:
                if g:  # 종성이 있는 경우
                    char = chr(base + cho.index(c) * 588 + jung.index(j) * 28 + jong.index(g))
                else:  # 종성이 없는 경우
                    char = chr(base + cho.index(c) * 588 + jung.index(j) * 28)
                row.append(char)
                total += 1
            result.append(' '.join(row))
    
    print(f"=== 총 {total}개의 한글 음절 ===\n")
    
    # 모든 음절 출력 (10글자씩 줄바꿈)
    syllables = [char for row in result for char in row.split()]
    for i in range(0, len(syllables), 30):
        print(' '.join(syllables[i:i+30]))

# 실행
generate_and_print_korean_syllables()