import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import winsound
import time
import threading

class PostureMonitorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("900x700")
        self.window.configure(bg="#0f172a") # Slate-900 background

        # --- Style Configuration ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#0f172a", foreground="#e2e8f0", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=10)
        style.configure("Header.TLabel", font=("Segoe UI", 24, "bold"), foreground="#38bdf8")
        style.configure("Status.TLabel", font=("Segoe UI", 12, "bold"), background="#1e293b", padding=5)

        # --- Header ---
        self.header_frame = tk.Frame(window, bg="#0f172a")
        self.header_frame.pack(pady=10)
        tk.Label(self.header_frame, text="ERGO", font=("Segoe UI", 28, "bold"), fg="#38bdf8", bg="#0f172a").pack(side=tk.LEFT)
        tk.Label(self.header_frame, text="PY", font=("Segoe UI", 28, "bold"), fg="#ffffff", bg="#0f172a").pack(side=tk.LEFT)

        # --- Video Canvas ---
        self.video_frame = tk.Frame(window, bg="#1e293b", bd=2, relief="groove")
        self.video_frame.pack(pady=10)
        
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480, bg="#000000", highlightthickness=0)
        self.canvas.pack()

        # --- HUD / Status ---
        self.status_frame = tk.Frame(window, bg="#0f172a")
        self.status_frame.pack(fill=tk.X, padx=50, pady=10)

        self.status_label = tk.Label(self.status_frame, text="WAITING FOR CAMERA...", font=("Segoe UI", 14, "bold"), fg="#94a3b8", bg="#1e293b", padx=20, pady=10)
        self.status_label.pack(pady=5)

        self.health_label = tk.Label(self.status_frame, text="POSTURE HEALTH: 100%", font=("Consolas", 10), fg="#4ade80", bg="#0f172a")
        self.health_label.pack()

        # --- Controls ---
        self.btn_frame = tk.Frame(window, bg="#0f172a")
        self.btn_frame.pack(pady=20)

        self.btn_calibrate = tk.Button(self.btn_frame, text="SET BASELINE", command=self.calibrate, 
                                       bg="#0284c7", fg="white", font=("Segoe UI", 11, "bold"), 
                                       activebackground="#0369a1", activeforeground="white", relief="flat", padx=20, pady=10)
        self.btn_calibrate.pack(side=tk.LEFT, padx=10)

        self.btn_reset = tk.Button(self.btn_frame, text="RESET", command=self.reset, 
                                   bg="#334155", fg="white", font=("Segoe UI", 11), 
                                   activebackground="#1e293b", activeforeground="white", relief="flat", padx=20, pady=10)
        self.btn_reset.pack(side=tk.LEFT, padx=10)

        # --- State Variables ---
        self.running = True
        self.calibrated = False
        self.baseline_landmarks = None
        self.posture_health = 100
        self.last_alert_time = 0
        
        # --- MediaPipe Setup ---
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # --- Camera Setup ---
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.status_label.config(text="ERROR: CAMERA NOT FOUND", fg="#f87171")
        else:
            self.update_video_loop()

    def update_video_loop(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            # 1. Preprocess
            frame = cv2.flip(frame, 1) # Mirror effect
            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 2. Pose Detection
            results = self.pose.process(rgb_frame)

            # 3. Drawing & Logic
            status_color = (255, 255, 255) # Default white
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Convert normalized coords to pixels for logic
                nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

                # --- Core Logic ---
                if self.calibrated and self.baseline_landmarks:
                    is_good, issues = self.check_posture(landmarks)
                    
                    if is_good:
                        status_color = (0, 255, 0) # Green (BGR -> RGB later)
                        self.status_label.config(text="PERFECT POSTURE", fg="#4ade80")
                        self.update_health(0.1) # Heal
                    else:
                        status_color = (0, 0, 255) # Red
                        issue_text = " + ".join(issues).upper()
                        self.status_label.config(text=f"⚠️ {issue_text}", fg="#f87171")
                        self.update_health(-0.5) # Damage
                        self.trigger_alert()
                
                else:
                    status_color = (255, 255, 0) # Yellow/Cyan
                    self.status_label.config(text="READY TO CALIBRATE", fg="#38bdf8")

                # Custom Drawing
                self.draw_skeleton(frame, landmarks, w, h, status_color)
            
            # 4. Display on Tkinter Canvas
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk # Keep reference

        self.window.after(15, self.update_video_loop)

    def draw_skeleton(self, frame, landmarks, w, h, color_rgb):
        # CV2 uses BGR, so flip color tuple
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        
        connections = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.LEFT_SHOULDER), # Approximation for neck visual
            (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        ]

        # Draw connections
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            cv2.line(frame, 
                     (int(start.x * w), int(start.y * h)), 
                     (int(end.x * w), int(end.y * h)), 
                     color_bgr, 3)

        # Draw key points
        needed_points = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_EYE, self.mp_pose.PoseLandmark.RIGHT_EYE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]
        
        for idx in needed_points:
            lm = landmarks[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, color_bgr, -1)

        # Draw Baseline Ghost if exists
        if self.calibrated and self.baseline_landmarks:
            base_l_shoulder = self.baseline_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            base_r_shoulder = self.baseline_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Draw a simple horizontal line for baseline shoulder height
            avg_y = (base_l_shoulder.y + base_r_shoulder.y) / 2
            y_px = int(avg_y * h)
            cv2.line(frame, (0, y_px), (w, y_px), (0, 255, 255), 1) # Yellow line
            cv2.putText(frame, "BASELINE", (10, y_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def check_posture(self, current_landmarks):
        # Thresholds (Normalized 0.0 to 1.0)
        SLOUCH_THRESHOLD = 0.05  # Dropping down
        LEAN_THRESHOLD = 0.08    # Leaning left/right
        NECK_THRESHOLD = 0.05    # Nose dropping

        issues = []
        
        # 1. Slouching (Shoulders Drop)
        # Y increases downwards in computer vision
        curr_y = (current_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
                  current_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
        
        base_y = (self.baseline_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
                  self.baseline_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
        
        if (curr_y - base_y) > SLOUCH_THRESHOLD:
            issues.append("Slouching")

        # 2. Leaning (Shoulder Center X)
        curr_x = (current_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x + 
                  current_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
        
        base_x = (self.baseline_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x + 
                  self.baseline_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2

        if abs(curr_x - base_x) > LEAN_THRESHOLD:
            issues.append("Leaning")

        # 3. Tech Neck (Nose position relative to shoulders)
        # If nose drops significantly relative to baseline nose
        curr_nose = current_landmarks[self.mp_pose.PoseLandmark.NOSE].y
        base_nose = self.baseline_landmarks[self.mp_pose.PoseLandmark.NOSE].y
        
        if (curr_nose - base_nose) > NECK_THRESHOLD:
            issues.append("Head Down")

        return len(issues) == 0, issues

    def calibrate(self):
        # Capture current frame landmarks as baseline
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Store a snapshot of the landmarks
                self.baseline_landmarks = results.pose_landmarks.landmark
                self.calibrated = True
                self.btn_calibrate.config(text="RE-CALIBRATE", bg="#059669") # Emerald color
                self.status_label.config(text="BASELINE SET", fg="#34d399")
            else:
                self.status_label.config(text="NO POSE DETECTED", fg="#fbbf24")

    def reset(self):
        self.calibrated = False
        self.baseline_landmarks = None
        self.btn_calibrate.config(text="SET BASELINE", bg="#0284c7")
        self.status_label.config(text="RESET COMPLETE", fg="#94a3b8")
        self.posture_health = 100

    def update_health(self, amount):
        self.posture_health = max(0, min(100, self.posture_health + amount))
        
        color = "#4ade80" # Green
        if self.posture_health < 70: color = "#facc15" # Yellow
        if self.posture_health < 40: color = "#f87171" # Red
        
        self.health_label.config(text=f"POSTURE HEALTH: {int(self.posture_health)}%", fg=color)

    def trigger_alert(self):
        current_time = time.time()
        # Debounce audio alerts (every 2 seconds max)
        if current_time - self.last_alert_time > 2:
            # Play standard Windows warning sound
            threading.Thread(target=winsound.MessageBeep, args=(winsound.MB_ICONWARNING,)).start()
            self.last_alert_time = current_time

    def cleanup(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PostureMonitorApp(root, "ErgoPy - Desktop Posture Monitor")
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()
