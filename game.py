import cv2
import time
import numpy as np
import random
from ultralytics import YOLO
import pyautogui
import pygame

# YOLOv11 모델 로드
model = YOLO('yolov8n.pt')

# 감지하고자 하는 클래스 목록
target_classes = ['cup', 'zebra', 'bus', 'elephant', 'cat', 'dog', 'bird', 'chair', 'book', 'car', 'apple', 'banana', 'person']

# 클래스 인덱스 찾기
target_class_indices = [i for i, name in model.names.items() if name in target_classes]

print("감지 대상 클래스:")
for idx in target_class_indices:
    print(f"{idx}: {model.names[idx]}")

# 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 웹캠 해상도 가져오기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 전체 해상도 가져오기 
screen_width, screen_height = pyautogui.size()
width = int(screen_width)
height = int(screen_height)

# pygame 초기화 및 음악 로드
pygame.mixer.init()
game_sound = pygame.mixer.Sound('game.mp3')  # 여기에 음악 파일 경로를 입력하세요
score_sound = pygame.mixer.Sound('score_sound.mp3') 
bgm = pygame.mixer.Sound('bgm.mp3') 

# 창 생성 및 크기 설정
cv2.namedWindow('YOLOv11 Real-time Detection', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('YOLOv11 Real-time Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

bgm.play()  # 효과음 재생

# 검정 화면 생성
black_screen = np.zeros((height, width, 3), dtype=np.uint8)
cv2.putText(black_screen, "Press '1' to start", (width//2 - 100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 검정 화면 표시
while True:
    cv2.imshow('YOLOv11 Real-time Detection', black_screen)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

bgm.stop()


# 카운트다운
for i in range(3, 0, -1):
    countdown_screen = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 텍스트 크기 계산
    text = str(i)
    font_scale = 5
    thickness = 5
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # 텍스트 위치 계산 (화면 중앙)
    text_x = (width - text_width) // 2
    text_y = (height + text_height) // 2
    
    cv2.putText(countdown_screen, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (0, 0, 255), thickness)
    
    cv2.imshow('YOLOv11 Real-time Detection', countdown_screen)
    cv2.waitKey(1000)

# 음악 시작 및 시작 시간 기록
game_sound.play()

# 시작 시간 기록
start_time = time.time()
duration = 60  # 총 실행 시간 (초)

# 점수 초기화
score = 0

# 랜덤 클래스 선택
target_class = random.choice(target_classes)

while True:
    # 현재 시간 확인
    current_time = time.time()
    elapsed_time = current_time - start_time
    remaining_time = max(0, duration - elapsed_time)
    
    # 시간이 다 되었는지 확인
    if remaining_time <= 0:
        print("60초가 지났습니다. 프로그램을 종료합니다.")
        break

    ret, frame = cap.read()
    if not ret:
        print("영상을 불러올 수 없습니다.")
        break

    # YOLO 모델을 사용해 객체 감지
    results = model(frame)

    # 결과 프레임 생성
    annotated_frame = frame.copy()

    # 가장 큰 객체 찾기 및 모든 객체 그리기
    largest_box = None
    largest_area = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls in target_class_indices:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                
                # 가장 큰 객체 업데이트
                if area > largest_area:
                    largest_area = area
                    largest_box = box

                # 모든 객체 검정색으로 그리기
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                class_name = model.names[cls]
                conf = float(box.conf[0])
                label = f'{class_name} {conf:.2f}'
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # 가장 큰 객체 빨간색으로 다시 그리기
    if largest_box is not None:
        x1, y1, x2, y2 = map(int, largest_box.xyxy[0].tolist())
        cls = int(largest_box.cls[0])
        class_name = model.names[cls]
        conf = float(largest_box.conf[0])
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)  # 빨간색
        label = f'{class_name} {conf:.2f}'
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        print(f"가장 큰 감지된 객체: {class_name}, 신뢰도: {conf:.2f}")

        # 점수 증가
        if class_name == target_class:
            score += 1
            score_sound.play()  # 효과음 재생
            target_class = random.choice(target_classes)  # 새로운 목표 클래스 선택

    # 목표 클래스를 화면 상단 중앙에 표시
    text = f"Target: {target_class}"
    font_scale = 1.5  # 텍스트 크기 증가
    thickness = 3  # 텍스트 두께 증가
    padding = 20  # 텍스트 주변 여백

    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (frame.shape[1] - text_width) // 2
    text_y = text_height + padding  # 텍스트 y 위치 조정

    # 배경 사각형 그리기
    cv2.rectangle(annotated_frame, 
                (text_x - padding, padding), 
                (text_x + text_width + padding, text_y + padding), 
                (255, 255, 255), 
                -1)

    # 텍스트 그리기
    cv2.putText(annotated_frame, text, 
                (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0, 0, 0), 
                thickness)

    # 점수를 화면 상단 우측에 표시
    cv2.rectangle(annotated_frame, (frame.shape[1]-150, 10), (frame.shape[1]-10, 50), (255, 255, 255), -1)
    cv2.putText(annotated_frame, f"Score: {score}", (frame.shape[1]-140, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # 남은 시간을 화면에 표시
    cv2.putText(annotated_frame, f"Time left: {int(remaining_time)}s", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    # 화면에 출력
    cv2.imshow('YOLOv11 Real-time Detection', annotated_frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 및 창 닫기
cap.release()

game_sound.stop()


# 최종 점수 표시
final_screen = np.zeros((height, width, 3), dtype=np.uint8)

# 텍스트 설정
text = f"Final Score: {score}"
font_scale = 1.5
thickness = 2

# 텍스트 크기 계산
(text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

# 텍스트 위치 계산 (화면 중앙)
text_x = (width - text_width) // 2
text_y = (height + text_height) // 2

# 텍스트 그리기
cv2.putText(final_screen, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, (255, 255, 255), thickness)
cv2.imshow('YOLOv11 Real-time Detection', final_screen)

bgm.play()  # 효과음 재생

# 10초 동안 최종 점수 화면 표시
cv2.waitKey(10000)

bgm.stop()

# pygame 정리
pygame.mixer.quit()

cv2.destroyAllWindows()
