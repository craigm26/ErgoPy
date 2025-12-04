import os
import warnings
import tkinter as tk
from tkinter import ttk
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple
import ctypes

# Keep TensorFlow/MediaPipe noisy logs out of the console.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")
from absl import logging as absl_logging

absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.use_absl_handler()

import cv2
import mediapipe as mp
from PIL import Image, ImageTk

# winsound is Windows-only; guard the import so the app still runs elsewhere.
try:
    import winsound  # type: ignore
except ImportError:
    winsound = None


@dataclass
class PostureThresholds:
    """Tunables for posture deviation sensitivity."""

    slouch: float = 0.05   # Shoulder drop tolerance
    lean: float = 0.08     # Sideways lean tolerance
    head_drop: float = 0.035  # Nose drop tolerance (stricter)
    head_forward: float = 0.12  # Nose depth change (more negative z -> forward)
    head_tilt: float = 0.03  # Eye height asymmetry (roll)
    shoulder_tilt: float = 0.04  # Shoulder height asymmetry


class AudioNotifier:
    """Plays alert sounds with debouncing and platform fallback."""

    def __init__(self, debounce_seconds: float = 2.0) -> None:
        self.debounce_seconds = debounce_seconds
        self.last_alert_time = 0.0

    def warn(self) -> None:
        now = time.time()
        if now - self.last_alert_time < self.debounce_seconds:
            return
        self.last_alert_time = now
        threading.Thread(target=self._play_sound, daemon=True).start()

    @staticmethod
    def _play_sound() -> None:
        if winsound:
            winsound.MessageBeep(winsound.MB_ICONWARNING)
        else:
            # Fallback to terminal bell; safe to ignore if suppressed.
            print("\a", end="", flush=True)


class PostureAnalyzer:
    """Computes posture issues against a calibrated baseline."""

    def __init__(self, thresholds: PostureThresholds) -> None:
        self.thresholds = thresholds
        self.baseline_landmarks: List = []

    @property
    def calibrated(self) -> bool:
        return bool(self.baseline_landmarks)

    def set_baseline(self, landmarks: List) -> None:
        self.baseline_landmarks = list(landmarks)

    def reset(self) -> None:
        self.baseline_landmarks = []

    def analyze(self, current_landmarks: List) -> Tuple[bool, List[str]]:
        if not self.baseline_landmarks:
            return False, ["Not calibrated"]

        t = self.thresholds
        issues: List[str] = []

        base = self.baseline_landmarks
        mp_pose = mp.solutions.pose.PoseLandmark

        # Slouching: shoulders drop.
        curr_y = (current_landmarks[mp_pose.LEFT_SHOULDER].y +
                  current_landmarks[mp_pose.RIGHT_SHOULDER].y) / 2
        base_y = (base[mp_pose.LEFT_SHOULDER].y + base[mp_pose.RIGHT_SHOULDER].y) / 2
        if (curr_y - base_y) > t.slouch:
            issues.append("Slouching")

        # Leaning: shoulders center shifts sideways.
        curr_x = (current_landmarks[mp_pose.LEFT_SHOULDER].x +
                  current_landmarks[mp_pose.RIGHT_SHOULDER].x) / 2
        base_x = (base[mp_pose.LEFT_SHOULDER].x + base[mp_pose.RIGHT_SHOULDER].x) / 2
        if abs(curr_x - base_x) > t.lean:
            issues.append("Leaning")

        # Shoulder tilt: uneven height vs baseline.
        curr_delta = current_landmarks[mp_pose.LEFT_SHOULDER].y - current_landmarks[mp_pose.RIGHT_SHOULDER].y
        base_delta = base[mp_pose.LEFT_SHOULDER].y - base[mp_pose.RIGHT_SHOULDER].y
        if abs(curr_delta - base_delta) > t.shoulder_tilt:
            issues.append("Shoulder Tilt")

        # Head drop: nose lowers relative to baseline.
        curr_nose = current_landmarks[mp_pose.NOSE].y
        base_nose = base[mp_pose.NOSE].y
        if (curr_nose - base_nose) > t.head_drop:
            issues.append("Head Down")

        # Head forward: nose moves closer to camera (more negative z).
        curr_nose_z = current_landmarks[mp_pose.NOSE].z
        base_nose_z = base[mp_pose.NOSE].z
        if (base_nose_z - curr_nose_z) > t.head_forward:
            issues.append("Head Forward")

        # Head tilt (roll): eye height difference vs baseline.
        curr_eye_delta = current_landmarks[mp_pose.LEFT_EYE].y - current_landmarks[mp_pose.RIGHT_EYE].y
        base_eye_delta = base[mp_pose.LEFT_EYE].y - base[mp_pose.RIGHT_EYE].y
        if abs(curr_eye_delta - base_eye_delta) > t.head_tilt:
            issues.append("Head Tilt")

        return len(issues) == 0, issues


class PostureMonitorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("900x700")
        self.window.configure(bg="#0f172a")  # Slate-900 background
        self.window.resizable(False, False)
        self.window.update_idletasks()
        self.hwnd = self.window.winfo_id()

        self.thresholds = PostureThresholds()
        self.analyzer = PostureAnalyzer(self.thresholds)
        self.notifier = AudioNotifier()
        self.latest_landmarks: List = []
        self.camera_index_var = tk.IntVar(value=0)
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.last_posture_state = "INIT"
        self.monitoring_enabled = True
        self.settings_modal = None

        # --- Style Configuration ---
        style = ttk.Style()
        style.theme_use("clam")
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
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW)

        # --- HUD / Status ---
        self.status_frame = tk.Frame(window, bg="#0f172a")
        self.status_frame.pack(fill=tk.X, padx=50, pady=10)

        self.status_label = tk.Label(self.status_frame, text="WAITING FOR CAMERA...", font=("Segoe UI", 14, "bold"), fg="#94a3b8", bg="#1e293b", padx=20, pady=10)
        self.status_label.pack(pady=5)
        self.status_label.bind("<Button-1>", self.on_status_click)

        self.health_label = tk.Label(self.status_frame, text="POSTURE HEALTH: 100%", font=("Consolas", 10), fg="#4ade80", bg="#0f172a")
        self.health_label.pack()

        self.fps_label = tk.Label(self.status_frame, text="FPS: --", font=("Consolas", 9), fg="#94a3b8", bg="#0f172a")
        self.fps_label.pack()

        # --- Instructions ---
        self.instructions = tk.Label(
            window,
            text=(
                "Setup: 1) Make sure your camera is allowed and not used elsewhere. "
                "2) Sit upright in neutral posture. 3) Click SET BASELINE. "
                "Tip: Adjust sensitivity sliders if alerts feel too strict/loose."
            ),
            wraplength=760,
            justify="left",
            font=("Segoe UI", 10),
            fg="#cbd5e1",
            bg="#0f172a",
        )
        self.instructions.pack(padx=20, pady=(0, 10))

        # --- Controls ---
        self.btn_frame = tk.Frame(window, bg="#0f172a")
        self.btn_frame.pack(pady=20)

        self.btn_calibrate = tk.Button(
            self.btn_frame,
            text="SET BASELINE",
            command=self.calibrate,
            bg="#0284c7",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            activebackground="#0369a1",
            activeforeground="white",
            relief="flat",
            padx=20,
            pady=10,
        )
        self.btn_calibrate.pack(side=tk.LEFT, padx=10)

        self.btn_reset = tk.Button(
            self.btn_frame,
            text="RESET",
            command=self.reset,
            bg="#334155",
            fg="white",
            font=("Segoe UI", 11),
            activebackground="#1e293b",
            activeforeground="white",
            relief="flat",
            padx=20,
            pady=10,
        )
        self.btn_reset.pack(side=tk.LEFT, padx=10)

        self.btn_rebaseline = tk.Button(
            self.btn_frame,
            text="REBASELINE",
            command=self.calibrate,
            bg="#059669",
            fg="white",
            font=("Segoe UI", 11),
            activebackground="#047857",
            activeforeground="white",
            relief="flat",
            padx=20,
            pady=10,
        )
        self.btn_rebaseline.pack(side=tk.LEFT, padx=10)

        self.btn_start = tk.Button(
            self.btn_frame,
            text="START",
            command=self.start_monitoring,
            bg="#16a34a",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            activebackground="#15803d",
            activeforeground="white",
            relief="flat",
            padx=20,
            pady=10,
        )
        self.btn_start.pack(side=tk.LEFT, padx=10)

        self.btn_stop = tk.Button(
            self.btn_frame,
            text="STOP",
            command=self.stop_monitoring,
            bg="#b91c1c",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            activebackground="#991b1b",
            activeforeground="white",
            relief="flat",
            padx=20,
            pady=10,
        )
        self.btn_stop.pack(side=tk.LEFT, padx=10)

        # --- Camera Controls ---
        self.camera_frame = tk.Frame(window, bg="#0f172a")
        self.camera_frame.pack(pady=5)
        tk.Label(self.camera_frame, text="Camera Index:", font=("Segoe UI", 10), fg="#e2e8f0", bg="#0f172a").pack(side=tk.LEFT, padx=(0, 6))
        self.camera_spin = tk.Spinbox(
            self.camera_frame,
            from_=0,
            to=10,
            width=4,
            textvariable=self.camera_index_var,
            bg="#1e293b",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            relief="flat",
            justify="center",
        )
        self.camera_spin.pack(side=tk.LEFT)
        tk.Button(
            self.camera_frame,
            text="Reconnect",
            command=self.change_camera,
            bg="#475569",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            activebackground="#334155",
            activeforeground="white",
            relief="flat",
            padx=10,
            pady=6,
        ).pack(side=tk.LEFT, padx=(8, 0))

        # --- Sensitivity Controls ---
        self.sensitivity_frame = tk.Frame(window, bg="#0f172a")
        self.sensitivity_frame.pack(pady=5)
        tk.Label(self.sensitivity_frame, text="Sensitivity (lower = stricter)", font=("Segoe UI", 10, "bold"), fg="#e2e8f0", bg="#0f172a").pack()

        tk.Button(
            self.sensitivity_frame,
            text="Advanced Settings",
            command=self.open_settings_modal,
            bg="#475569",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            activebackground="#334155",
            activeforeground="white",
            relief="flat",
            padx=12,
            pady=8,
        ).pack(pady=4)

        # --- State Variables ---
        self.running = True
        self.posture_health = 100

        # --- MediaPipe Setup ---
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.mp_drawing = mp.solutions.drawing_utils

        # --- Camera Setup ---
        self.cap = None
        if not self.open_camera(self.camera_index_var.get()):
            self.status_label.config(text="ERROR: CAMERA NOT FOUND", fg="#f87171")
        else:
            self.log_event("Camera opened", {"index": self.camera_index_var.get()})
            self.update_video_loop()

    def update_video_loop(self):
        if not self.running:
            return

        if not self.cap or not self.cap.isOpened():
            self.status_label.config(text="CAMERA NOT READY", fg="#f87171")
            self.log_event("Camera not ready")
            self.window.after(500, self.update_video_loop)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="ERROR: CAMERA FEED LOST", fg="#f87171")
            self.log_event("Camera feed lost")
            self.window.after(500, self.update_video_loop)
            return

        frame = cv2.flip(frame, 1)  # Mirror effect
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb_frame)
        self.latest_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else []

        status_color = (255, 255, 255)

        if not self.monitoring_enabled:
            self.status_label.config(text="PAUSED - CLICK START", fg="#cbd5e1", cursor="hand2")
            self.last_posture_state = "PAUSED"
            self.draw_skeleton(frame, self.latest_landmarks, w, h, status_color) if self.latest_landmarks else None
        elif self.latest_landmarks:
            if self.analyzer.calibrated:
                is_good, issues = self.analyzer.analyze(self.latest_landmarks)
                if is_good:
                    status_color = (0, 255, 0)
                    self.status_label.config(text="PERFECT POSTURE", fg="#4ade80", cursor="arrow")
                    self.update_health(0.15)
                    if self.last_posture_state != "GOOD":
                        self.log_event("Posture good")
                        self.last_posture_state = "GOOD"
                else:
                    status_color = (0, 0, 255)
                    issue_text = " + ".join(issues).upper()
                    self.status_label.config(text=f"ALERT: {issue_text}", fg="#f87171", cursor="arrow")
                    self.update_health(-0.5)
                    self.trigger_alert()
                    if self.last_posture_state != issue_text:
                        self.log_event("Posture issue", {"issues": issue_text})
                        self.last_posture_state = issue_text
            else:
                status_color = (255, 255, 0)
                self.status_label.config(text="READY TO CALIBRATE (click status or SET BASELINE)", fg="#38bdf8", cursor="hand2")
                if self.last_posture_state != "UNCALIBRATED":
                    self.log_event("Waiting for calibration")
                    self.last_posture_state = "UNCALIBRATED"

            self.draw_skeleton(frame, self.latest_landmarks, w, h, status_color)
        else:
            self.status_label.config(text="NO PERSON DETECTED", fg="#fbbf24")
            if self.last_posture_state != "NO_PERSON":
                self.log_event("No person detected")
                self.last_posture_state = "NO_PERSON"
            self.status_label.config(cursor="arrow")
            self.draw_skeleton(frame, self.latest_landmarks, w, h, status_color) if self.latest_landmarks else None

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.itemconfig(self.canvas_image, image=imgtk)
        self.canvas.imgtk = imgtk  # Keep reference to avoid GC

        # Update FPS every ~30 frames
        self.frame_counter += 1
        if self.frame_counter % 30 == 0:
            now = time.time()
            elapsed = now - self.last_fps_time
            if elapsed > 0:
                fps = 30 / elapsed
                self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.last_fps_time = now

        self.window.after(15, self.update_video_loop)

    def draw_skeleton(self, frame, landmarks, w, h, color_rgb):
        # CV2 uses BGR, so flip color tuple
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        connections = [
            # Upper body
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.LEFT_SHOULDER),  # Approx neck
            (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            # Torso/hips for better posture context
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            # Legs (to show alignment if visible)
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
        ]

        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            cv2.line(
                frame,
                (int(start.x * w), int(start.y * h)),
                (int(end.x * w), int(end.y * h)),
                color_bgr,
                3,
            )

        needed_points = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_EYE,
            self.mp_pose.PoseLandmark.RIGHT_EYE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_EAR,
            self.mp_pose.PoseLandmark.RIGHT_EAR,
        ]

        for idx in needed_points:
            lm = landmarks[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, color_bgr, -1)

        if self.analyzer.calibrated:
            base_l_shoulder = self.analyzer.baseline_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            base_r_shoulder = self.analyzer.baseline_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            avg_y = (base_l_shoulder.y + base_r_shoulder.y) / 2
            y_px = int(avg_y * h)
            cv2.line(frame, (0, y_px), (w, y_px), (0, 255, 255), 1)  # Yellow line
            cv2.putText(frame, "BASELINE", (10, y_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def open_camera(self, index: int) -> bool:
        try:
            idx = int(index)
        except ValueError:
            return False

        if self.cap and self.cap.isOpened():
            self.cap.release()

        self.cap = cv2.VideoCapture(idx)
        if self.cap.isOpened():
            self.camera_index_var.set(idx)
            return True
        return False

    def change_camera(self):
        idx = self.camera_index_var.get()
        if self.open_camera(idx):
            self.status_label.config(text="CAMERA CONNECTED", fg="#34d399")
            self.log_event("Camera reconnected", {"index": idx})
        else:
            self.status_label.config(text="CAMERA NOT FOUND", fg="#f87171")
            self.log_event("Camera reconnect failed", {"index": idx})

    def on_slouch_change(self, value: str) -> None:
        self.thresholds.slouch = float(value)

    def on_lean_change(self, value: str) -> None:
        self.thresholds.lean = float(value)

    def on_head_change(self, value: str) -> None:
        self.thresholds.head_drop = float(value)

    def on_tilt_change(self, value: str) -> None:
        self.thresholds.shoulder_tilt = float(value)

    def on_head_tilt_change(self, value: str) -> None:
        self.thresholds.head_tilt = float(value)

    def on_head_forward_change(self, value: str) -> None:
        self.thresholds.head_forward = float(value)

    def flash_taskbar(self):
        # Flash the taskbar icon on Windows as a subtle notification.
        if os.name != "nt":
            return
        if not self.hwnd:
            return
        FLASHW_TRAY = 0x00000002
        FLASHW_TIMERNOFG = 0x0000000C

        class FLASHWINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_uint),
                ("hwnd", ctypes.c_void_p),
                ("dwFlags", ctypes.c_uint),
                ("uCount", ctypes.c_uint),
                ("dwTimeout", ctypes.c_uint),
            ]

        info = FLASHWINFO(
            ctypes.sizeof(FLASHWINFO),
            ctypes.c_void_p(self.hwnd),
            FLASHW_TRAY | FLASHW_TIMERNOFG,
            2,  # flash twice per alert debounce
            0,
        )
        try:
            ctypes.windll.user32.FlashWindowEx(ctypes.byref(info))
        except Exception:
            # Best-effort: ignore if flashing fails.
            pass

    def calibrate(self):
        # Prefer the latest detected landmarks to avoid hiccups when clicking.
        if self.latest_landmarks:
            self.analyzer.set_baseline(self.latest_landmarks)
            self.btn_calibrate.config(text="RE-CALIBRATE", bg="#059669")  # Emerald color
            self.status_label.config(text="BASELINE SET", fg="#34d399")
            self.log_event("Baseline set", {"source": "live_landmarks"})
            self.status_label.config(cursor="arrow")
            return

        # Fallback: capture a fresh frame to attempt calibration.
        if not self.cap or not self.cap.isOpened():
            self.status_label.config(text="CAMERA NOT READY", fg="#f87171")
            self.log_event("Calibration failed", {"reason": "camera_not_ready"})
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="CAMERA UNAVAILABLE", fg="#f87171")
            self.log_event("Calibration failed", {"reason": "camera_unavailable"})
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            self.analyzer.set_baseline(results.pose_landmarks.landmark)
            self.btn_calibrate.config(text="RE-CALIBRATE", bg="#059669")
            self.status_label.config(text="BASELINE SET", fg="#34d399")
            self.log_event("Baseline set", {"source": "captured_frame"})
            self.status_label.config(cursor="arrow")
        else:
            self.status_label.config(text="NO POSE DETECTED", fg="#fbbf24")
            self.log_event("Calibration failed", {"reason": "no_pose"})

    def reset(self):
        self.analyzer.reset()
        self.btn_calibrate.config(text="SET BASELINE", bg="#0284c7")
        self.status_label.config(text="RESET COMPLETE", fg="#94a3b8")
        self.posture_health = 100

    def update_health(self, amount):
        self.posture_health = max(0, min(100, self.posture_health + amount))

        color = "#4ade80"  # Green
        if self.posture_health < 70:
            color = "#facc15"  # Yellow
        if self.posture_health < 40:
            color = "#f87171"  # Red

        self.health_label.config(text=f"POSTURE HEALTH: {int(self.posture_health)}%", fg=color)

    def trigger_alert(self):
        self.notifier.warn()
        self.flash_taskbar()

    def log_event(self, message: str, data: dict | None = None) -> None:
        payload = f"[ErgoPy] {message}"
        if data:
            payload += f" | {data}"
        print(payload, flush=True)

    def on_status_click(self, event):
        # Allow users to click the status banner to calibrate when prompted.
        if not self.analyzer.calibrated:
            self.log_event("Status clicked for calibration")
            self.calibrate()

    def start_monitoring(self):
        if not self.monitoring_enabled:
            self.monitoring_enabled = True
            self.status_label.config(text="RESUMED - READY", fg="#38bdf8", cursor="arrow")
            self.log_event("Monitoring started/resumed")
            if not self.running:
                self.running = True
                self.update_video_loop()

    def stop_monitoring(self):
        self.monitoring_enabled = False
        self.status_label.config(text="PAUSED - CLICK START", fg="#cbd5e1", cursor="hand2")
        self.log_event("Monitoring paused")

    def open_settings_modal(self):
        if self.settings_modal and tk.Toplevel.winfo_exists(self.settings_modal):
            self.settings_modal.lift()
            return

        modal = tk.Toplevel(self.window)
        modal.title("Advanced Settings")
        modal.configure(bg="#0f172a")
        modal.resizable(False, False)
        self.settings_modal = modal

        tk.Label(modal, text="Sensitivity (lower = stricter)", font=("Segoe UI", 11, "bold"), fg="#e2e8f0", bg="#0f172a").pack(pady=(10, 6), padx=10)

        def scale(parent, label, frm, to, resolution, value, command):
            scale_widget = tk.Scale(
                parent,
                from_=frm,
                to=to,
                resolution=resolution,
                orient=tk.HORIZONTAL,
                length=260,
                label=label,
                bg="#0f172a",
                fg="#e2e8f0",
                troughcolor="#1e293b",
                highlightthickness=0,
                command=command,
            )
            scale_widget.set(value)
            scale_widget.pack(pady=4, padx=12)
            return scale_widget

        scale(modal, "Slouch Threshold", 0.02, 0.12, 0.005, self.thresholds.slouch, self.on_slouch_change)
        scale(modal, "Lean Threshold", 0.04, 0.16, 0.005, self.thresholds.lean, self.on_lean_change)
        scale(modal, "Head Drop Threshold", 0.015, 0.10, 0.003, self.thresholds.head_drop, self.on_head_change)
        scale(modal, "Head Forward Threshold", 0.05, 0.25, 0.01, self.thresholds.head_forward, self.on_head_forward_change)
        scale(modal, "Shoulder Tilt Threshold", 0.02, 0.10, 0.005, self.thresholds.shoulder_tilt, self.on_tilt_change)
        scale(modal, "Head Tilt Threshold", 0.01, 0.08, 0.002, self.thresholds.head_tilt, self.on_head_tilt_change)

        tk.Button(
            modal,
            text="Close",
            command=modal.destroy,
            bg="#334155",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            activebackground="#1e293b",
            activeforeground="white",
            relief="flat",
            padx=14,
            pady=8,
        ).pack(pady=(6, 12))

        def on_close():
            self.settings_modal = None
            modal.destroy()

        modal.protocol("WM_DELETE_WINDOW", on_close)

    def cleanup(self):
        self.running = False
        if hasattr(self, "pose"):
            self.pose.close()
        if getattr(self, "cap", None) and self.cap.isOpened():
            self.cap.release()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PostureMonitorApp(root, "ErgoPy - Desktop Posture Monitor")
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()
