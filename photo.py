import cv2
import mediapipe as mp
import os
import time
import subprocess

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

# Create "photos" folder if it doesn't exist
photo_folder = "photos"
os.makedirs(photo_folder, exist_ok=True)

# Start capturing video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

photo_count = 1  # Counter for photo filenames
last_capture_time = 0  # Timestamp of last photo capture

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
        index_tip = landmarkList[8][2]   # y-coordinate of index finger tip
        middle_tip = landmarkList[12][2] # y-coordinate of middle finger tip
        ring_knuckle = landmarkList[13][2] # y-coordinate of ring finger knuckle
        other_fingers_down = all(landmarkList[i][2] > ring_knuckle for i in [16, 20])  # Ring & pinky fingers down
        
        is_index_raised = index_tip < ring_knuckle
        is_middle_raised = middle_tip < ring_knuckle
        current_time = time.time()

        if is_index_raised and is_middle_raised and other_fingers_down and (current_time - last_capture_time > 10):  
            # Capture photo if 10 seconds have passed since last capture
            photo_path = os.path.join(photo_folder, f"photo_{photo_count}.jpg")
            print(f"Index & Middle fingers raised! Capturing {photo_path}...")
            cv2.imwrite(photo_path, frame)
            print(f"Photo saved as {photo_path}")

            # Open the captured photo
            if os.name == "nt":  # Windows
                os.startfile(photo_path)
            elif os.name == "posix":  # Linux/macOS
                subprocess.run(["xdg-open", photo_path])

            photo_count += 1  # Increment photo count
            last_capture_time = current_time  # Update last capture timestamp
            time.sleep(1)  # Small delay to avoid immediate repeat

    cv2.imshow('Gesture Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
