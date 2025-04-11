import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time
import subprocess
import threading
import speech_recognition as sr
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc


class GestureControlSystem:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            max_num_hands=2)
        self.Draw = mp.solutions.drawing_utils

        self.setup_volume_control()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.photo_folder = "photos"
        os.makedirs(self.photo_folder, exist_ok=True)
        self.photo_count = 1
        self.last_capture_time = 0
        self.photo_countdown = 0
        self.countdown_start_time = 0
        self.file_manager_opened_time = 0
        self.file_manager_opened = False
        self.gesture_cooldown = 1.0
        self.running = True

        threading.Thread(target=self.listen_for_voice_commands, daemon=True).start()

    def setup_volume_control(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))

    def close_file_manager(self):
        if os.name == "nt":
            os.system("taskkill /IM explorer.exe /F")
            os.system("start explorer.exe")
        else:
            os.system("pkill nautilus")

    def capture_photo(self, frame):
        current_time = time.time()
        elapsed = current_time - self.countdown_start_time
        if elapsed < 3:
            countdown_num = 3 - int(elapsed)
            cv2.putText(frame, f"Taking photo in: {countdown_num}",
                        (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            return False
        else:
            photo_path = os.path.join(self.photo_folder, f"photo_{self.photo_count}.jpg")
            cv2.imwrite(photo_path, frame)
            if os.name == "nt":
                os.startfile(photo_path)
            else:
                subprocess.run(["xdg-open", photo_path])
            self.photo_count += 1
            self.last_capture_time = current_time
            return True

    def process_gestures(self, landmarkList, frame):
        if len(landmarkList) < 21:
            self.photo_countdown = 0
            return

        thumb_tip = landmarkList[4][2]
        index_tip = landmarkList[8][2]
        middle_tip = landmarkList[12][2]
        ring_knuckle = landmarkList[13][2]
        little_tip = landmarkList[20][2]
        index_knuckle = landmarkList[5][2]
        middle_knuckle = landmarkList[9][2]
        current_time = time.time()

        # Brightness control
        x1, y1 = landmarkList[4][1], landmarkList[4][2]
        x2, y2 = landmarkList[8][1], landmarkList[8][2]
        distance = math.hypot(x2 - x1, y2 - y1)
        b_level = np.interp(distance, [15, 220], [0, 100])
        sbc.set_brightness(int(b_level))

        # Volume control
        x1, y1 = landmarkList[4][1], landmarkList[4][2]
        x2, y2 = landmarkList[20][1], landmarkList[20][2]
        distance = math.hypot(x2 - x1, y2 - y1)
        vol_level = np.interp(distance, [15, 220], [-65.25, 0])
        self.volume.SetMasterVolumeLevel(vol_level, None)

        # Photo gesture
        is_index_raised = index_tip < ring_knuckle
        is_middle_raised = middle_tip < ring_knuckle
        other_fingers_down = all(landmarkList[i][2] > ring_knuckle for i in [16, 20])
        if is_index_raised and is_middle_raised and other_fingers_down and \
                (current_time - self.last_capture_time > self.gesture_cooldown):
            if self.photo_countdown == 0:
                self.photo_countdown = 1
                self.countdown_start_time = current_time
            else:
                if not self.capture_photo(frame):
                    pass
        else:
            self.photo_countdown = 0

        # File manager gestures
        is_thumb_up = thumb_tip < index_knuckle
        is_thumb_down = thumb_tip > index_knuckle
        is_other_fingers_down = all(landmarkList[i][2] > middle_knuckle for i in [8, 12, 16, 20])

        if not self.file_manager_opened and is_thumb_up and is_other_fingers_down and \
                current_time - self.file_manager_opened_time > self.gesture_cooldown:
            os.system("explorer" if os.name == "nt" else "xdg-open .")
            self.file_manager_opened = True
            self.file_manager_opened_time = current_time

        if self.file_manager_opened and is_thumb_down and is_other_fingers_down and \
                current_time - self.file_manager_opened_time > self.gesture_cooldown:
            self.close_file_manager()
            self.file_manager_opened = False
            self.file_manager_opened_time = current_time

    def display_instructions(self, frame):
        instructions = [
            "Gesture & Voice Controls:",
            "1. Thumb+Index: Brightness",
            "2. Thumb+Little: Volume",
            "3. Index+Middle: Take Photo",
            "4. Thumb Up/Down: Open/Close File Manager",
            "Voice: 'increase brightness', 'set volume to 30', etc.",
            "Apps: 'open calculator', 'open chrome', 'open vs code'",
            "Press 'q' to quit"
        ]
        y_offset = 30
        for line in instructions:
            cv2.putText(frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

    def listen_for_voice_commands(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
        while self.running:
            try:
                with mic as source:
                    print("Listening for voice commands...")
                    audio = recognizer.listen(source, timeout=5)
                    command = recognizer.recognize_google(audio).lower()
                    print("Heard:", command)

                    if "increase brightness" in command:
                        sbc.set_brightness(min(sbc.get_brightness()[0] + 20, 100))
                    elif "decrease brightness" in command:
                        sbc.set_brightness(max(sbc.get_brightness()[0] - 20, 0))
                    elif "increase volume" in command:
                        self.volume.SetMasterVolumeLevel(
                            min(self.volume.GetMasterVolumeLevel() + 5.0, 0.0), None)
                    elif "decrease volume" in command:
                        self.volume.SetMasterVolumeLevel(
                            max(self.volume.GetMasterVolumeLevel() - 5.0, -65.25), None)
                    elif "click photo" in command:
                        self.countdown_start_time = time.time()
                        self.photo_countdown = 1
                    elif "open file manager" in command:
                        os.system("explorer" if os.name == "nt" else "xdg-open .")
                        self.file_manager_opened = True
                    elif "close file manager" in command:
                        self.close_file_manager()
                        self.file_manager_opened = False
                    elif "open calculator" in command:
                        subprocess.Popen("calc" if os.name == "nt" else "gnome-calculator", shell=True)
                    elif "open chrome" in command:
                        subprocess.Popen("start chrome" if os.name == "nt" else "google-chrome", shell=True)
                    elif "open vs code" in command or "open vscode" in command:
                        subprocess.Popen("code", shell=True)

                    elif "set brightness to" in command:
                        try:
                            level = int(command.split("set brightness to")[-1].strip().split()[0])
                            sbc.set_brightness(max(0, min(level, 100)))
                        except:
                            print("Couldn't parse brightness value.")

                    elif "set volume to" in command:
                        try:
                            level = int(command.split("set volume to")[-1].strip().split()[0])
                            vol_level = np.interp(level, [0, 100], [-65.25, 0])
                            self.volume.SetMasterVolumeLevel(vol_level, None)
                        except:
                            print("Couldn't parse volume value.")

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("Could not understand the audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

    def run(self):
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

            self.display_instructions(frame)
            cv2.imshow('Gesture Control System', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    system = GestureControlSystem()
    system.run()
