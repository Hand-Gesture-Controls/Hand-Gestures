# Importing Libraries 
import cv2 
import mediapipe as mp 
from math import hypot 
import screen_brightness_control as sbc 
import numpy as np 

# Initializing the Model 
mpHands = mp.solutions.hands 
hands = mpHands.Hands( 
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.75, 
    max_num_hands=2) 

Draw = mp.solutions.drawing_utils 

# Start capturing video with higher quality 
cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

while True: 
    # Read video frame by frame 
    success, frame = cap.read() 
    if not success:
        continue

    # Flip image 
    frame = cv2.flip(frame, 1) 

    # Convert BGR image to RGB image 
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

    # Process the RGB image 
    Process = hands.process(frameRGB) 

    landmarkList = [] 
    # if hands are present in image(frame) 
    if Process.multi_hand_landmarks: 
        # detect handmarks 
        for handlm in Process.multi_hand_landmarks: 
            for _id, landmarks in enumerate(handlm.landmark): 
                # store height and width of image 
                height, width, _ = frame.shape 

                # calculate and append x, y coordinates 
                x, y = int(landmarks.x * width), int(landmarks.y * height) 
                landmarkList.append([_id, x, y]) 

            # draw Landmarks 
            Draw.draw_landmarks(frame, handlm, 
                                mpHands.HAND_CONNECTIONS) 

    # If landmarks list is not empty 
    if landmarkList: 
        # store x,y coordinates of (tip of) thumb 
        x_1, y_1 = landmarkList[4][1], landmarkList[4][2] 

        # store x,y coordinates of (tip of) index finger 
        x_2, y_2 = landmarkList[8][1], landmarkList[8][2] 

        # draw circle on thumb and index finger tip 
        cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED) 
        cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED) 

        # draw line from tip of thumb to tip of index finger 
        cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3) 

        # calculate distance between thumb and index finger 
        L = hypot(x_2 - x_1, y_2 - y_1) 

        # Map distance to brightness range 
        b_level = np.interp(L, [15, 220], [0, 100]) 

        # set brightness 
        sbc.set_brightness(int(b_level)) 

    # Add a colorful overlay to improve visibility
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 5)
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Display Video and when 'q' is entered, destroy window 
    cv2.imshow('Brightness Control', frame) 
    if cv2.waitKey(1) & 0xff == ord('q'): 
        break 

cap.release()
cv2.destroyAllWindows()
