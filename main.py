'''import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time
import subprocess
import psutil
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

class GestureControlSystem:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            max_num_hands=2)
        
        self.Draw = mp.solutions.drawing_utils
        
        # Initialize volume control
        self.setup_volume_control()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Photo capture variables
        self.photo_folder = "photos"
        os.makedirs(self.photo_folder, exist_ok=True)
        self.photo_count = 1
        self.last_capture_time = 0
        
        # File manager variables
        self.file_manager_opened_time = 0
        self.file_manager_opened = False
        
        # Gesture recognition thresholds
        self.gesture_cooldown = 1.0  # seconds between gesture activations
        
    def setup_volume_control(self):
        """Initialize system volume control using Pycaw"""
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
    
    def close_file_manager(self):
        """Closes the file manager."""
        if os.name == "nt":  # Windows
            os.system("taskkill /IM explorer.exe /F")
            os.system("start explorer.exe")
        else:  # Linux
            os.system("pkill nautilus")
    
    def process_gestures(self, landmarkList, frame):
        """Process all gestures based on hand landmarks"""
        if len(landmarkList) < 21:
            return
        
        # Get key points coordinates
        thumb_tip = landmarkList[4][2]
        index_tip = landmarkList[8][2]
        middle_tip = landmarkList[12][2]
        ring_knuckle = landmarkList[13][2]
        little_tip = landmarkList[20][2]
        
        # Reference points
        index_knuckle = landmarkList[5][2]
        middle_knuckle = landmarkList[9][2]
        
        # Gesture recognition
        current_time = time.time()
        
        # 1. Brightness Control (Thumb and Index finger)
        x1, y1 = landmarkList[4][1], landmarkList[4][2]  # Thumb tip
        x2, y2 = landmarkList[8][1], landmarkList[8][2]  # Index tip
        distance = math.hypot(x2 - x1, y2 - y1)
        b_level = np.interp(distance, [15, 220], [0, 100])
        sbc.set_brightness(int(b_level))
        
        # 2. Volume Control (Thumb and Little finger)
        x1, y1 = landmarkList[4][1], landmarkList[4][2]  # Thumb tip
        x2, y2 = landmarkList[20][1], landmarkList[20][2]  # Little tip
        distance = math.hypot(x2 - x1, y2 - y1)
        vol_level = np.interp(distance, [15, 220], [-65.25, 0])
        self.volume.SetMasterVolumeLevel(vol_level, None)
        
        # 3. Photo Capture (Index and Middle fingers up)
        is_index_raised = index_tip < ring_knuckle
        is_middle_raised = middle_tip < ring_knuckle
        other_fingers_down = all(landmarkList[i][2] > ring_knuckle for i in [16, 20])
        
        if (is_index_raised and is_middle_raised and other_fingers_down and 
            (current_time - self.last_capture_time > self.gesture_cooldown)):
            self.capture_photo(frame)
            self.last_capture_time = current_time
        
        # 4. File Manager Control (Thumbs up/down)
        is_thumb_up = thumb_tip < index_knuckle
        is_thumb_down = thumb_tip > index_knuckle
        is_other_fingers_down = all(landmarkList[i][2] > middle_knuckle for i in [8, 12, 16, 20])
        
        # Open file manager
        if (not self.file_manager_opened and is_thumb_up and is_other_fingers_down and 
            current_time - self.file_manager_opened_time > self.gesture_cooldown):
            os.system("explorer" if os.name == "nt" else "xdg-open .")
            self.file_manager_opened = True
            self.file_manager_opened_time = current_time
        
        # Close file manager
        if (self.file_manager_opened and is_thumb_down and is_other_fingers_down and 
            current_time - self.file_manager_opened_time > self.gesture_cooldown):
            self.close_file_manager()
            self.file_manager_opened = False
            self.file_manager_opened_time = current_time
    
    def capture_photo(self, frame):
        """Capture and save photo from camera frame"""
        photo_path = os.path.join(self.photo_folder, f"photo_{self.photo_count}.jpg")
        cv2.imwrite(photo_path, frame)
        
        # Open the captured photo
        if os.name == "nt":  # Windows
            os.startfile(photo_path)
        elif os.name == "posix":  # Linux/macOS
            subprocess.run(["xdg-open", photo_path])
        
        self.photo_count += 1
    
    def run(self):
        """Main loop for gesture control system"""
        while True:
            success, frame = self.cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Process = self.hands.process(frameRGB)
            
            landmarkList = []
            
            if Process.multi_hand_landmarks:
                for handlm in Process.multi_hand_landmarks:
                    for _id, landmarks in enumerate(handlm.landmark):
                        height, width, _ = frame.shape
                        x, y = int(landmarks.x * width), int(landmarks.y * height)
                        landmarkList.append([_id, x, y])
                    self.Draw.draw_landmarks(frame, handlm, self.mpHands.HAND_CONNECTIONS)
            
            if landmarkList:
                self.process_gestures(landmarkList, frame)
            
            # Display instructions
            self.display_instructions(frame)
            
            cv2.imshow('Gesture Control System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def display_instructions(self, frame):
        """Display gesture instructions on the frame"""
        instructions = [
            "Gesture Controls:",
            "1. Thumb+Index: Brightness",
            "2. Thumb+Little: Volume",
            "3. Index+Middle: Take Photo",
            "4. Thumb Up: Open File Manager",
            "5. Thumb Down: Close File Manager",
            "Press 'q' to quit"
        ]
        
        y_offset = 30
        for line in instructions:
            cv2.putText(frame, line, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

if __name__ == "__main__":
    system = GestureControlSystem()
    system.run()'''






import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time
import subprocess
import psutil
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

class GestureControlSystem:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            max_num_hands=2)
        
        self.Draw = mp.solutions.drawing_utils
        
        # Initialize volume control
        self.setup_volume_control()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Photo capture variables
        self.photo_folder = "photos"
        os.makedirs(self.photo_folder, exist_ok=True)
        self.photo_count = 1
        self.last_capture_time = 0
        self.photo_countdown = 0
        self.countdown_start_time = 0
        
        # File manager variables
        self.file_manager_opened_time = 0
        self.file_manager_opened = False
        
        # Gesture recognition thresholds
        self.gesture_cooldown = 1.0  # seconds between gesture activations
    
    def setup_volume_control(self):
        """Initialize system volume control using Pycaw"""
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
    
    def close_file_manager(self):
        """Closes the file manager."""
        if os.name == "nt":  # Windows
            os.system("taskkill /IM explorer.exe /F")
            os.system("start explorer.exe")
        else:  # Linux
            os.system("pkill nautilus")
    
    def capture_photo(self, frame):
        """Show countdown and then capture photo"""
        current_time = time.time()
        elapsed = current_time - self.countdown_start_time
        
        if elapsed < 3:  # Show countdown
            countdown_num = 3 - int(elapsed)
            cv2.putText(frame, f"Taking photo in: {countdown_num}", 
                       (frame.shape[1]//2 - 100, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            return False
        else:  # Take photo
            photo_path = os.path.join(self.photo_folder, f"photo_{self.photo_count}.jpg")
            cv2.imwrite(photo_path, frame)
            
            # Open the captured photo
            if os.name == "nt":  # Windows
                os.startfile(photo_path)
            elif os.name == "posix":  # Linux/macOS
                subprocess.run(["xdg-open", photo_path])
            
            self.photo_count += 1
            self.last_capture_time = current_time
            return True
    
    def process_gestures(self, landmarkList, frame):
        """Process all gestures based on hand landmarks"""
        if len(landmarkList) < 21:
            self.photo_countdown = 0  # Reset countdown if hand not detected
            return
        
        # Get key points coordinates
        thumb_tip = landmarkList[4][2]
        index_tip = landmarkList[8][2]
        middle_tip = landmarkList[12][2]
        ring_knuckle = landmarkList[13][2]
        little_tip = landmarkList[20][2]
        
        # Reference points
        index_knuckle = landmarkList[5][2]
        middle_knuckle = landmarkList[9][2]
        
        # Gesture recognition
        current_time = time.time()
        
        # 1. Brightness Control (Thumb and Index finger)
        x1, y1 = landmarkList[4][1], landmarkList[4][2]  # Thumb tip
        x2, y2 = landmarkList[8][1], landmarkList[8][2]  # Index tip
        distance = math.hypot(x2 - x1, y2 - y1)
        b_level = np.interp(distance, [15, 220], [0, 100])
        sbc.set_brightness(int(b_level))
        
        # 2. Volume Control (Thumb and Little finger)
        x1, y1 = landmarkList[4][1], landmarkList[4][2]  # Thumb tip
        x2, y2 = landmarkList[20][1], landmarkList[20][2]  # Little tip
        distance = math.hypot(x2 - x1, y2 - y1)
        vol_level = np.interp(distance, [15, 220], [-65.25, 0])
        self.volume.SetMasterVolumeLevel(vol_level, None)
        
        # 3. Photo Capture (Index and Middle fingers up)
        is_index_raised = index_tip < ring_knuckle
        is_middle_raised = middle_tip < ring_knuckle
        other_fingers_down = all(landmarkList[i][2] > ring_knuckle for i in [16, 20])
        
        if (is_index_raised and is_middle_raised and other_fingers_down and 
            (current_time - self.last_capture_time > self.gesture_cooldown)):
            if self.photo_countdown == 0:  # Start countdown
                self.photo_countdown = 1
                self.countdown_start_time = current_time
            else:  # Continue countdown
                if not self.capture_photo(frame):
                    # Countdown in progress (handled in capture_photo)
                    pass
        else:
            self.photo_countdown = 0  # Reset countdown
        
        # 4. File Manager Control (Thumbs up/down)
        is_thumb_up = thumb_tip < index_knuckle
        is_thumb_down = thumb_tip > index_knuckle
        is_other_fingers_down = all(landmarkList[i][2] > middle_knuckle for i in [8, 12, 16, 20])
        
        # Open file manager
        if (not self.file_manager_opened and is_thumb_up and is_other_fingers_down and 
            current_time - self.file_manager_opened_time > self.gesture_cooldown):
            os.system("explorer" if os.name == "nt" else "xdg-open .")
            self.file_manager_opened = True
            self.file_manager_opened_time = current_time
        
        # Close file manager
        if (self.file_manager_opened and is_thumb_down and is_other_fingers_down and 
            current_time - self.file_manager_opened_time > self.gesture_cooldown):
            self.close_file_manager()
            self.file_manager_opened = False
            self.file_manager_opened_time = current_time
    
    def display_instructions(self, frame):
        """Display gesture instructions on the frame"""
        instructions = [
            "Gesture Controls:",
            "1. Thumb+Index: Brightness",
            "2. Thumb+Little: Volume",
            "3. Index+Middle: Take Photo (3s countdown)",
            "4. Thumb Up: Open File Manager",
            "5. Thumb Down: Close File Manager",
            "Press 'q' to quit"
        ]
        
        y_offset = 30
        for line in instructions:
            cv2.putText(frame, line, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
    
    def run(self):
        """Main loop for gesture control system"""
        while True:
            success, frame = self.cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Process = self.hands.process(frameRGB)
            
            landmarkList = []
            
            if Process.multi_hand_landmarks:
                for handlm in Process.multi_hand_landmarks:
                    for _id, landmarks in enumerate(handlm.landmark):
                        height, width, _ = frame.shape
                        x, y = int(landmarks.x * width), int(landmarks.y * height)
                        landmarkList.append([_id, x, y])
                    self.Draw.draw_landmarks(frame, handlm, self.mpHands.HAND_CONNECTIONS)
            
            if landmarkList:
                self.process_gestures(landmarkList, frame)
            
            # Display instructions
            self.display_instructions(frame)
            
            cv2.imshow('Gesture Control System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = GestureControlSystem()
    system.run()