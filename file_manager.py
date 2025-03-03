import cv2
import mediapipe as mp
import os
import time
import psutil  # For closing the file manager

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=1
)
Draw = mp.solutions.drawing_utils

# Start capturing video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

file_manager_opened_time = 0  # Track when the file manager was last opened
file_manager_opened = False

def close_file_manager():
    """Closes the file manager."""
    if os.name == "nt":  # Windows
        os.system("taskkill /IM explorer.exe /F")  # Force close file explorer
        os.system("start explorer.exe")  # Restart explorer to avoid system issues
    else:  # Linux
        os.system("pkill nautilus")  # Close the default file manager on Linux

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Process = hands.process(frameRGB)

    landmarkList = []

    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                height, width, _ = frame.shape
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, x, y])
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    if len(landmarkList) >= 21:
        thumb_tip = landmarkList[4][2]
        index_knuckle = landmarkList[5][2]

        is_thumb_up = thumb_tip < index_knuckle
        is_thumb_down = thumb_tip > index_knuckle
        is_other_fingers_down = all(landmarkList[i][2] > index_knuckle for i in [8, 12, 16, 20])

        # Open file manager when thumbs-up is detected
        if not file_manager_opened and is_thumb_up and is_other_fingers_down:
            current_time = time.time()
            if current_time - file_manager_opened_time > 3:  # Prevent multiple openings
                print("👍 Thumbs-up detected! Opening File Manager...")
                os.system("explorer" if os.name == "nt" else "xdg-open .")
                file_manager_opened = True
                file_manager_opened_time = current_time

        # Close file manager when thumbs-down is detected
        if file_manager_opened and is_thumb_down and is_other_fingers_down:
            print("👎 Thumbs-down detected! Closing File Manager...")
            close_file_manager()
            file_manager_opened = False  # Reset flag after closing

    cv2.imshow('Thumbs-Up & Thumbs-Down Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
