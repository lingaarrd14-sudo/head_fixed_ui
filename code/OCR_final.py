import cv2
import os
import easyocr
import pytesseract
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
dir=os.path.join(parent_dir, 'pic')

# 두 가지 모델 모두 불러오기
print("모델들을 준비 중입니다...")
reader = easyocr.Reader(['en'])
pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'

image_path = os.path.join(dir, '1.jpg')
img = cv2.imread(image_path)

if img is None:
    print("이미지를 찾을 수 없습니다.")
else:
    print("1단계: EasyOCR로 텍스트 구역을 탐지합니다...")
    results = reader.readtext(image_path)
    
    result_img = img.copy() # 최종 박스를 그릴 도화지

    print("2단계: 찾아낸 구역 전처리 후 Tesseract로 알파벳 쪼개기...")

    for bbox, text, prob in results:
        # 1. EasyOCR이 찾은 박스의 네 모서리 좌표 가져오기
        tl, tr, br, bl = bbox
        
        # 2. 이미지를 자르기 위해 정확한 [시작점(최소값) ~ 끝점(최대값)] 구하기
        x_min = int(min(tl[0], bl[0]))
        x_max = int(max(tr[0], br[0]))
        y_min = int(min(tl[1], tr[1]))
        y_max = int(max(bl[1], br[1]))
        
        # 이미지 범위를 벗어나지 않도록 방어 코드 추가
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img.shape[1], x_max), min(img.shape[0], y_max)
        
        # 3. 원본 이미지에서 글자 부분만 가위로 오려내기
        cropped_img = img[y_min:y_max, x_min:x_max]
        if cropped_img.size == 0:
            continue

        padded_h, padded_w = cropped_img.shape[:2]
        
        # 4. 전처리된 이미지를 Tesseract에 넣기
        custom_config = r'--psm 7'
        boxes = pytesseract.image_to_boxes(cropped_img, lang='eng', config=custom_config)
        
       # 5. 좌표 역산 및 그리기
        for b in boxes.splitlines():
            b = b.split(' ')
            char = b[0]
            
            # Tesseract 좌표 (좌측 하단 기준)
            left = int(b[1])
            bottom = int(b[2])
            right = int(b[3])
            top = int(b[4])
            
            # Y좌표를 화면 좌상단 기준으로 변경 (Tesseract는 아래쪽이 0이므로 뒤집어줌)
            crop_y_top = padded_h - top
            char_width = right - left
            char_height = top - bottom
            
            # 최종 좌표, 폭, 넓이 계산
            final_x = x_min + left
            final_y = y_min + crop_y_top
            final_w = char_width
            final_h = char_height
            
            print(f"📝 글자: {char} / 📍 원본 위치: (X:{final_x}, Y:{final_y}) / 📏 크기: (W:{final_w}, H:{final_h})")
            
            # 원본 도화지에 최종 초록색 박스 그리기
            cv2.rectangle(result_img, (final_x, final_y), (final_x + final_w, final_y + final_h), (0, 255, 0), 1)

    cv2.imshow('Hybrid OCR Result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()