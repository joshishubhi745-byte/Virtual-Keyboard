# ⌨️ AI-Powered Virtual Interactive Keyboard

A touchless, gesture-controlled virtual keyboard application that transforms a standard webcam stream into an interactive input peripheral. By leveraging real-time computer vision pipelines, the system tracks hand landmark coordinates, maps them onto a custom dynamic UI overlay, and injects native hardware-level keystrokes using intuitive finger-pinch gestures.

---

## ✨ Features

* **Zero-Lag Video Pipeline:** Fully integrated with the modern MediaPipe Tasks API utilizing native VIDEO context tracking and matching frames to monotonic system timestamps to completely eliminate inference latency.
* **Pinch-to-Click Spatial Mechanics:** Computes real-time mathematical Euclidean distances between tracking node landmarks (Index Tip & Middle Tip) to accurately register click events without physical surface contact.
* **Calibrated Compact UI Grid:** Features an optimized, downsized 60x60 pixel bounding-box key layout centered dynamically within a 720p frame space, leaving the screen open and ensuring stable tracking along the edge boundaries.
* **Keystroke Debounce Engine:** Built-in cool-down window to filter duplicate stroke events during sustained finger pinches.
* **Live Visual State Feedback:** Implements dynamic rendering color shifts (Default Charcoal -> Hover Light Gray -> Active Green Click) to guide user interaction smoothly.

---

## 🛠️ Technical Architecture & Dependencies

* **Core Language:** Python
* **OpenCV (opencv-python):** Handles image capture, horizontal frame mirroring, canvas draw operations, and UI alpha transparency rendering.
* **MediaPipe Tasks Framework (mediapipe):** Runs the specialized machine learning Hand Landmarker solution to track normalized coordinates.
* **Pynput (pynput):** Simulates low-level background hardware keyboard event injection directly into the operating system.

---

## 📥 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/joshishubhi745-byte/Virtual-Keyboard.git](https://github.com/joshishubhi745-byte/Virtual-Keyboard.git)
cd Virtual-Keyboard

2. Install Required Dependencies
Bash
pip install opencv-python mediapipe pynput numpy

3. Download the Pre-trained Model Asset
The system requires MediaPipe's modern machine learning task model package to evaluate hand landmarks.
1).Download the file from Google's official cloud storage: hand_landmarker.task
2).Place the downloaded file directly into your project's root directory (alongside your key.py script).

🚀 How to Run & Use
1).Click inside any active text field on your computer (such as Notepad, a browser address bar, or a blank document).
2).Launch the application from your terminal:
Bash
python key.py
3).Look at the camera window feed. Move your hand to hover your Index Finger Tip over an on-screen letter key.

4).Bring your Index Finger and Middle Finger tightly together (Pinch Gesture) to type the letter! The on-screen key will turn green and fire a keystroke to your active text field.

5).To close the program safely, bring focus back to the camera window and tap the q key on your physical keyboard.
