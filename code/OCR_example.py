import cv2
import easyocr
import pytesseract
import numpy as np

# 1. 두 가지 모델 모두 불러오기
print("모델들을 준비 중입니다...")
reader = easyocr.Reader(['en'])
pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'

image_path = 'D:\\python\\gui_low_vision\\4.png'
img = cv2.imread(image_path)

if img is None:
    print("이미지를 찾을 수 없습니다.")
else:
    print("1단계: EasyOCR로 텍스트 구역을 탐지합니다...")
    results = reader.readtext(image_path)
    
    result_img = img.copy() # 최종 박스를 그릴 도화지

    print("2단계: 찾아낸 구역 전처리 후 Tesseract로 알파벳 쪼개기...")
    
    # --- 전처리 설정값 ---
    SCALE = 2    # 확대 배율 (작은 글씨 인식률 향상)
    PAD = 15     # 여백 크기 (Tesseract가 글자를 잘 찾게 숨통 틔워주기)
    # -------------------

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
            
        # ==========================================
        # ★ 새로 추가된 전처리 (Pre-processing) 구역 ★
        # ==========================================
        
        # [전처리 1] 이미지 확대 (Upscaling)
        scaled_img = cv2.resize(cropped_img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        
        # [전처리 2] 여백 추가 (Padding) - 흰색(255,255,255) 테두리 생성
        padded_img = cv2.copyMakeBorder(scaled_img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # [전처리 3] 흑백 변환 및 이진화 (Grayscale & Otsu Thresholding)
        gray = cv2.cvtColor(padded_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # ==========================================

        padded_h, padded_w = thresh.shape[:2]
        
        # 4. 전처리된 이미지를 Tesseract에 넣기
        custom_config = r'--psm 6'
        boxes = pytesseract.image_to_boxes(thresh, lang='eng', config=custom_config)
        
        # 5. 좌표 역산 및 그리기
        for b in boxes.splitlines():
            b = b.split(' ')
            char = b[0]
            
            # Tesseract 좌표 (좌측 하단 기준)
            left = int(b[1])
            bottom = int(b[2])
            right = int(b[3])
            top = int(b[4])
            
            # Y좌표를 좌상단 기준으로 변경
            crop_y_top = padded_h - top
            char_width = right - left
            char_height = top - bottom
            
            # ★ 핵심: 추가했던 여백(PAD)을 빼고, 확대했던 배율(SCALE)로 나눠서 좌표 복원
            unpadded_x = left - PAD
            unpadded_y = crop_y_top - PAD
            
            orig_crop_x = unpadded_x / SCALE
            orig_crop_y = unpadded_y / SCALE
            orig_crop_w = char_width / SCALE
            orig_crop_h = char_height / SCALE
            
            # 원본 이미지에서의 최종 진짜 좌표 계산
            final_x = int(x_min + orig_crop_x)
            final_y = int(y_min + orig_crop_y)
            final_w = int(orig_crop_w)
            final_h = int(orig_crop_h)
            
            print(f"📝 글자: {char} / 📍 원본 위치: (X:{final_x}, Y:{final_y}) / 📏 크기: (W:{final_w}, H:{final_h})")
            
            # 원본 도화지에 최종 초록색 박스 그리기
            cv2.rectangle(result_img, (final_x, final_y), (final_x + final_w, final_y + final_h), (0, 255, 0), 1)

    cv2.imshow('Hybrid OCR Result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()