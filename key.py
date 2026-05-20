import cv2
import mediapipe as mp
import math
import numpy as np
from pynput.keyboard import Controller, Key
import time

# Modern MediaPipe Tasks API imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize keyboard controllers for OS typing events
keyboard = Controller()

# WebCam Setup
cap = cv2.VideoCapture(0)  
cap.set(3, 1280) # Width
cap.set(4, 720)  # Height

# Keyboard Layout Configuration Grid
keys_layout = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
    ["SPACE"]
]

class KeyboardKey:
    def __init__(self, text, pos, size=(80, 80)):
        self.text = text
        self.pos = pos
        self.size = size

# Build a downsized, compact layout grid positions dynamically
keyboard_keys = []
start_x = 240  # Centers the smaller keyboard horizontally on your 1280 width screen
start_y = 150  # Positions it at a comfortable height

for row_idx, row in enumerate(keys_layout):
    for col_idx, key_text in enumerate(row):
        if key_text == "SPACE":
            # Compact spacebar styled to match the new grid scale
            keyboard_keys.append(KeyboardKey(key_text, (430, 375), (320, 55)))
        else:
            x_pos = start_x + col_idx * 73  # Tightened horizontal spacing (down from 100)
            y_pos = start_y + row_idx * 73  # Tightened vertical row spacing (down from 95)
            # Sets individual key bounding sizes to a smaller 60x60 square
            keyboard_keys.append(KeyboardKey(key_text, (x_pos, y_pos), size=(60, 60)))

last_pressed_time = 0
cooldown_period = 0.5  # Half-second cooldown to avoid duplicate characters

def draw_keyboard(img, keys, active_key=None, clicked_key=None):
    """Renders a clean, semi-transparent visual keyboard onto the camera frame."""
    overlay = img.copy()
    for key in keys:
        x, y = key.pos
        w, h = key.size
        
        # Determine background and text colors based on state
        if key == clicked_key:
            bg_color = (0, 255, 0)       # Green highlight when typed
            text_color = (0, 0, 0)
        elif key == active_key:
            bg_color = (200, 200, 200)   # Light gray highlight on hover
            text_color = (0, 0, 0)
        else:
            bg_color = (40, 40, 40)      # Default charcoal background
            text_color = (255, 255, 255)
            
        cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, cv2.FILLED)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        font_scale = 1 if key.text != "SPACE" else 0.8
        text_size = cv2.getTextSize(key.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        
        cv2.putText(overlay, key.text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
        
    alpha = 0.35  # Makes layout keys slightly clearer than before
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

# Configure Modern Hand Landmarker using VIDEO running mode
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # Optimizes tracker strictly for live frame feeds
    num_hands=1,
    min_hand_detection_confidence=0.5
)

# Open detection stream wrapper context
with vision.HandLandmarker.create_from_options(options) as detector:
    print("Application successfully opened. Click on the camera window and press 'q' to exit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)  # Mirror frame horizontally so left/right movements feel normal
        h_img, w_img, _ = frame.shape
        
        # Calculate current timestamp in milliseconds required by detect_for_video
        frame_timestamp_ms = int(time.time() * 1000)
        
        # Convert OpenCV BGR frame to MediaPipe RGB Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Run detection
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        
        hovered_key = None
        pressed_key = None
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Target Landmark 8 (Index Finger Tip) and Landmark 12 (Middle Finger Tip)
                idx_tip = hand_landmarks[8]
                mid_tip = hand_landmarks[12]
                
                # Convert normalized coordinates (0.0 - 1.0) to pixel screen space
                x_idx, y_idx = int(idx_tip.x * w_img), int(idx_tip.y * h_img)
                x_mid, y_mid = int(mid_tip.x * w_img), int(mid_tip.y * h_img)
                
                # Render a visual pink dot right on your tracking fingertip
                cv2.circle(frame, (x_idx, y_idx), 12, (255, 0, 255), cv2.FILLED)
                
                # Evaluate keyboard layout collision logic
                for key in keyboard_keys:
                    kx, ky = key.pos
                    kw, kh = key.size
                    
                    # Check if index finger tip is hovering inside a key boundary box
                    if kx < x_idx < kx + kw and ky < y_idx < ky + kh:
                        hovered_key = key
                        
                        # Measure distance between index tip and middle tip to detect click
                        distance = math.hypot(x_mid - x_idx, y_mid - y_idx)
                        
                        # When fingers pinch closely together (Pinch Click)
                        if distance < 45: 
                            current_time = time.time()
                            if current_time - last_pressed_time > cooldown_period:
                                pressed_key = key
                                
                                # Send direct keystroke string commands to your OS system
                                if key.text == "SPACE":
                                    keyboard.press(Key.space)
                                    keyboard.release(Key.space)
                                    print("Pressed: [SPACE]")
                                else:
                                    keyboard.press(key.text.lower())
                                    keyboard.release(key.text.lower())
                                    print(f"Pressed: {key.text}")
                                    
                                last_pressed_time = current_time

        # Update and render the structural keyboard over frames
        frame = draw_keyboard(frame, keyboard_keys, active_key=hovered_key, clicked_key=pressed_key)
        cv2.imshow('Virtual Keyboard', frame)
        
        # Quit safely if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
