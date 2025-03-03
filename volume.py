import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=1)

Draw = mp.solutions.drawing_utils

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Start capturing video with higher quality
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

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
        # Get coordinates of thumb tip and little finger tip
        x1, y1 = landmarkList[4][1], landmarkList[4][2]   # Thumb tip
        x2, y2 = landmarkList[20][1], landmarkList[20][2] # Little finger tip
        
        # Calculate distance between thumb tip and little finger tip
        distance = math.hypot(x2 - x1, y2 - y1)
        
        # Map the distance to volume range
        vol_level = np.interp(distance, [15, 220], [-65.25, 0])
        volume.SetMasterVolumeLevel(vol_level, None)
    
    cv2.imshow('Volume Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()