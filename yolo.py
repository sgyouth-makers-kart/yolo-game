import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO('yolov8n.pt')  # 또는 다른 모델 파일 사용

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # YOLO로 객체 감지
    results = model(frame)

    # 결과를 프레임에 그리기
    annotated_frame = results[0].plot()

    # 원본 프레임과 YOLO 적용 프레임을 나란히 표시
    #combined_frame = np.hstack((frame, annotated_frame))

    # 창에 표시
    cv2.imshow('Webcam and YOLO', annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
