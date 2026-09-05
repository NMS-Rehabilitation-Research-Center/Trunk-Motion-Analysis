"""Desktop frontal-video tool for right and left trunk lateral flexion."""

import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import mediapipe as mp
import numpy as np
import queue
import threading
import time
from PIL import Image, ImageTk

# ============================
# Main application
# ============================
class SpineAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Frontal Trunk Analysis - Right / Left Lateral Flexion")
        self.root.geometry("1100x750")
        self.root.configure(bg='#ffffff')
        
        # Playback state
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.is_paused = False
        self.current_frame = 0
        self.fps = 30
        self.total_frames = 0
        self.analysis_thread = None
        self.stop_analysis = False
        self.ui_queue = queue.Queue(maxsize=3)
        self.rotation_degrees = 0
        self.rotation_choice = tk.StringVar(value="0°")
        self.view_choice = tk.StringVar(value="Normal")
        self.view_factor = 1
        
        # Trunk measurements
        self.spine_angle = 0
        self.right_flexion = 0
        self.left_flexion = 0
        self.right_flexion_max = 0
        self.left_flexion_max = 0
        self.rom_min = 180
        self.rom_max = 180
        self.pose_detected = False
        self.analysis_data = []
        
        # Recent values shown in the chart
        self.angle_history = []
        self.max_history = 150
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Build the interface
        self.create_widgets()
        self.root.after(15, self.process_ui_queue)
        
    # ============================
    # Interface
    # ============================
    def create_widgets(self):
        # Main layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Give the video more space than the measurement panel.
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left column: video
        video_frame = ttk.LabelFrame(main_frame, text="Video Preview", padding=10)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.video_label = ttk.Label(video_frame, background='#ffffff')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Right column: measurements
        info_frame = ttk.LabelFrame(main_frame, text="Frontal Trunk Analysis", padding=15)
        info_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        info_frame.columnconfigure(0, weight=1)
        
        title = ttk.Label(info_frame, text="Trunk Status", font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Main inclination value
        angle_frame = ttk.Frame(info_frame)
        angle_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(angle_frame, text="Frontal angle:", font=('Arial', 12)).pack(side=tk.LEFT)
        self.angle_label = ttk.Label(angle_frame, text="0°", font=('Arial', 20, 'bold'), foreground='#00ff88')
        self.angle_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Separator(info_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Right and left lateral flexion
        flex_frame = ttk.Frame(info_frame)
        flex_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(flex_frame, text="Right lateral flexion:", font=('Arial', 11)).pack(side=tk.LEFT)
        self.right_flexion_label = ttk.Label(flex_frame, text="0°", font=('Arial', 16, 'bold'), foreground='#0077aa')
        self.right_flexion_label.pack(side=tk.LEFT, padx=10)
        
        ext_frame = ttk.Frame(info_frame)
        ext_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(ext_frame, text="Left lateral flexion:", font=('Arial', 11)).pack(side=tk.LEFT)
        self.left_flexion_label = ttk.Label(ext_frame, text="0°", font=('Arial', 16, 'bold'), foreground='#c23b4d')
        self.left_flexion_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Separator(info_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Peak values
        max_frame = ttk.LabelFrame(info_frame, text="Peak Values", padding=10)
        max_frame.pack(fill=tk.X, pady=5)
        
        max_flex_frame = ttk.Frame(max_frame)
        max_flex_frame.pack(fill=tk.X, pady=2)
        ttk.Label(max_flex_frame, text="Peak right:", font=('Arial', 10)).pack(side=tk.LEFT)
        self.max_right_label = ttk.Label(max_flex_frame, text="0°", font=('Arial', 11, 'bold'), foreground='#0077aa')
        self.max_right_label.pack(side=tk.LEFT, padx=10)
        
        max_ext_frame = ttk.Frame(max_frame)
        max_ext_frame.pack(fill=tk.X, pady=2)
        ttk.Label(max_ext_frame, text="Peak left:", font=('Arial', 10)).pack(side=tk.LEFT)
        self.max_left_label = ttk.Label(max_ext_frame, text="0°", font=('Arial', 11, 'bold'), foreground='#c23b4d')
        self.max_left_label.pack(side=tk.LEFT, padx=10)
        
        # Observed range
        rom_frame = ttk.Frame(max_frame)
        rom_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rom_frame, text="Observed range:", font=('Arial', 10)).pack(side=tk.LEFT)
        self.rom_label = ttk.Label(rom_frame, text="0° - 0°", font=('Arial', 11, 'bold'), foreground='#ffdd00')
        self.rom_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Separator(info_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Status
        status_frame = ttk.Frame(info_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(status_frame, text="Status:", font=('Arial', 12)).pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, text="Neutral", font=('Arial', 14, 'bold'), foreground='#00ff88')
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Live chart
        chart_frame = ttk.LabelFrame(info_frame, text="Live Trunk Angle", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.chart_canvas = tk.Canvas(chart_frame, height=150, bg='#ffffff', highlightthickness=0)
        self.chart_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bottom toolbar
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)
        
        btn_load = ttk.Button(control_frame, text="Open Video", command=self.load_video)
        btn_load.pack(side=tk.LEFT, padx=5)
        
        self.btn_play = ttk.Button(control_frame, text="Play", command=self.toggle_play, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = ttk.Button(control_frame, text="Stop", command=self.stop_video, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        btn_reset = ttk.Button(control_frame, text="Reset", command=self.reset_data)
        btn_reset.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Rotate:").pack(side=tk.LEFT, padx=(12, 4))
        self.rotation_menu = ttk.Combobox(
            control_frame,
            textvariable=self.rotation_choice,
            values=("0°", "90° CW", "180°", "270° CW"),
            state="readonly",
            width=9,
        )
        self.rotation_menu.pack(side=tk.LEFT)
        self.rotation_menu.bind("<<ComboboxSelected>>", self.on_rotation_changed)

        ttk.Label(control_frame, text="View:").pack(side=tk.LEFT, padx=(12, 4))
        self.view_menu = ttk.Combobox(
            control_frame,
            textvariable=self.view_choice,
            values=("Normal", "Mirrored"),
            state="readonly",
            width=10,
        )
        self.view_menu.pack(side=tk.LEFT)
        self.view_menu.bind("<<ComboboxSelected>>", self.on_view_changed)

        ttk.Button(control_frame, text="Export CSV", command=self.export_data).pack(
            side=tk.LEFT, padx=(12, 5)
        )
        
        # Playback progress
        self.progress = ttk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.time_label = ttk.Label(control_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.LEFT, padx=5)
        
    # ============================
    # Playback
    # ============================
    def rotate_frame(self, frame):
        """Rotate a frame to the orientation selected in the toolbar."""
        if self.rotation_degrees == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation_degrees == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotation_degrees == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def on_rotation_changed(self, _event=None):
        """Refresh the paused preview immediately after changing rotation."""
        self.rotation_degrees = {
            "0°": 0,
            "90° CW": 90,
            "180°": 180,
            "270° CW": 270,
        }[self.rotation_choice.get()]
        if self.cap and self.cap.isOpened() and (not self.is_playing or self.is_paused):
            position = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            preview_index = max(0, int(position) - 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, preview_index)
            ret, frame = self.cap.read()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            if ret:
                self.display_frame(self.rotate_frame(frame))

    def on_view_changed(self, _event=None):
        """Correct anatomical side labels when the video is mirrored."""
        self.view_factor = 1 if self.view_choice.get() == "Normal" else -1

    def load_video(self):
        """Open a local video and show its first frame."""
        file_path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if file_path:
            if self.cap:
                self.cap.release()
            self.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)
            
            if not self.cap.isOpened():
                messagebox.showerror("Video Error", "The selected video could not be opened.")
                return
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0
            self.progress['to'] = self.total_frames
            self.reset_data()
            
            # Enable playback controls.
            self.btn_play.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.NORMAL)
            
            # Show the first frame.
            self.update_frame()
    
    def toggle_play(self):
        """Start, pause, or resume playback."""
        if self.cap is None or not self.cap.isOpened():
            return
            
        if not self.is_playing:
            self.is_playing = True
            self.is_paused = False
            self.btn_play.config(text="Pause")
            self.stop_analysis = False
            
            if self.analysis_thread is None or not self.analysis_thread.is_alive():
                self.analysis_thread = threading.Thread(target=self.process_video)
                self.analysis_thread.daemon = True
                self.analysis_thread.start()
        else:
            if self.is_paused:
                self.is_paused = False
                self.btn_play.config(text="Pause")
            else:
                self.is_paused = True
                self.btn_play.config(text="Resume")
    
    def stop_video(self):
        """Stop playback and return to the first frame."""
        self.is_playing = False
        self.is_paused = False
        self.stop_analysis = True
        self.btn_play.config(text="Play")
        
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.update_frame()
            self.progress.set(0)
            self.time_label.config(text="00:00 / 00:00")
    
    def reset_data(self):
        """Clear peak values and chart history."""
        self.right_flexion_max = 0
        self.left_flexion_max = 0
        self.rom_min = 180
        self.rom_max = 180
        self.pose_detected = False
        self.analysis_data = []
        self.angle_history = []
        
        self.max_right_label.config(text="0°")
        self.max_left_label.config(text="0°")
        self.rom_label.config(text="0° - 0°")
        self.chart_canvas.delete("all")
    
    def process_video(self):
        """Process video frames on the playback thread."""
        if self.cap is None:
            return
            
        while self.is_playing and not self.stop_analysis:
            if self.is_paused:
                time.sleep(0.05)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                self.is_playing = False
                self.enqueue_ui_event(("play_state", "Play"))
                break
                
            self.current_frame += 1
            frame = self.rotate_frame(frame)

            # Analyze the same rotated frame that will be displayed.
            analyzed_frame = self.analyze_frame(frame)
            current_time = self.current_frame / self.fps
            total_time = self.total_frames / self.fps
            self.analysis_data.append({
                "Frame": self.current_frame,
                "Time (s)": f"{current_time:.3f}",
                "Trunk angle (deg)": f"{self.spine_angle:.2f}" if self.pose_detected else "",
                "Right lateral flexion (deg)": f"{self.right_flexion:.2f}" if self.pose_detected else "",
                "Left lateral flexion (deg)": f"{self.left_flexion:.2f}" if self.pose_detected else "",
                "Pose detected": "Yes" if self.pose_detected else "No",
                "Rotation (deg)": self.rotation_degrees,
                "Video view": self.view_choice.get(),
            })
            self.enqueue_ui_event(
                (
                    "frame",
                    analyzed_frame.copy(),
                    self.current_frame,
                    current_time,
                    total_time,
                )
            )
            
            time.sleep(1/self.fps)

    def enqueue_ui_event(self, event):
        """Queue a worker result, dropping the oldest result if rendering falls behind."""
        try:
            self.ui_queue.put_nowait(event)
        except queue.Full:
            try:
                self.ui_queue.get_nowait()
            except queue.Empty:
                pass
            self.ui_queue.put_nowait(event)

    def process_ui_queue(self):
        """Render worker results on Tkinter's main thread."""
        try:
            latest_frame = None
            try:
                while True:
                    event = self.ui_queue.get_nowait()
                    if event[0] == "frame":
                        latest_frame = event
                    elif event[0] == "play_state":
                        self.btn_play.config(text=event[1])
            except queue.Empty:
                pass

            if latest_frame is not None:
                _, frame, frame_index, current_time, total_time = latest_frame
                self.update_gui()
                self.display_frame(frame)
                self.progress.set(frame_index)
                self.time_label.config(
                    text=f"{self.format_time(current_time)} / {self.format_time(total_time)}"
                )
        finally:
            try:
                self.root.after(15, self.process_ui_queue)
            except tk.TclError:
                pass
    
    def analyze_frame(self, frame):
        """Run pose estimation and calculate the trunk angle for one frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            self.pose_detected = True
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            # Draw the detected pose.
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Calculate trunk inclination.
            self.calculate_spine_angle(landmarks, w, h)
            
            # Draw the measurement overlay.
            self.draw_info_on_frame(frame)
        else:
            self.pose_detected = False
            cv2.putText(frame, "No person detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def calculate_spine_angle(self, landmarks, w, h):
        """Calculate right and left lateral trunk flexion in the frontal plane."""
        # Required landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Bilateral shoulder and hip midpoints
        mid_shoulder = np.array([
            (left_shoulder.x + right_shoulder.x) / 2 * w,
            (left_shoulder.y + right_shoulder.y) / 2 * h
        ])
        
        mid_hip = np.array([
            (left_hip.x + right_hip.x) / 2 * w,
            (left_hip.y + right_hip.y) / 2 * h
        ])
        
        # Trunk segment vector
        spine_vec = mid_shoulder - mid_hip
        
        # Directed frontal-plane angle: upright is 180 degrees. In a normal
        # frontal view, image-left is the subject's anatomical right.
        if np.linalg.norm(spine_vec) > 0:
            signed_tilt = np.degrees(np.arctan2(spine_vec[0], -spine_vec[1]))
            anatomical_tilt = signed_tilt * self.view_factor
            self.spine_angle = (180 + anatomical_tilt) % 360
        else:
            self.spine_angle = 180
        
        if self.spine_angle < 180:
            self.right_flexion = 180 - self.spine_angle
            self.left_flexion = 0
        elif self.spine_angle > 180:
            self.right_flexion = 0
            self.left_flexion = self.spine_angle - 180
        else:
            self.right_flexion = 0
            self.left_flexion = 0
        
        # Peak values
        if self.right_flexion > self.right_flexion_max:
            self.right_flexion_max = self.right_flexion
        if self.left_flexion > self.left_flexion_max:
            self.left_flexion_max = self.left_flexion
        
        # Observed range
        if self.spine_angle < self.rom_min:
            self.rom_min = self.spine_angle
        if self.spine_angle > self.rom_max:
            self.rom_max = self.spine_angle
        
        # Chart history
        self.angle_history.append(self.spine_angle)
        if len(self.angle_history) > self.max_history:
            self.angle_history.pop(0)
        
    def update_gui(self):
        """Refresh measurement labels and the live chart."""
        self.angle_label.config(text=f"{int(self.spine_angle)}°")
        
        self.right_flexion_label.config(text=f"{int(self.right_flexion)}°")
        self.left_flexion_label.config(text=f"{int(self.left_flexion)}°")
        
        self.max_right_label.config(text=f"{int(self.right_flexion_max)}°")
        self.max_left_label.config(text=f"{int(self.left_flexion_max)}°")
        
        self.rom_label.config(text=f"{int(self.rom_min)}° - {int(self.rom_max)}°")
        
        if self.right_flexion > 30:
            self.status_label.config(text="High right lateral flexion", foreground='#ff6b6b')
        elif self.left_flexion > 30:
            self.status_label.config(text="High left lateral flexion", foreground='#ff6b6b')
        elif self.right_flexion > 15:
            self.status_label.config(text="Moderate right lateral flexion", foreground='#b8860b')
        elif self.left_flexion > 15:
            self.status_label.config(text="Moderate left lateral flexion", foreground='#b8860b')
        else:
            self.status_label.config(text="Neutral", foreground='#00ff88')
        
        self.update_chart()
    
    def update_chart(self):
        """Redraw the recent trunk-angle trace."""
        self.chart_canvas.delete("all")
        
        width = self.chart_canvas.winfo_width()
        height = self.chart_canvas.winfo_height()
        
        if width < 10 or height < 10 or len(self.angle_history) < 2:
            return
        
        # Axes
        self.chart_canvas.create_line(30, 10, 30, height-10, fill="#444466", width=2)
        self.chart_canvas.create_line(30, height-10, width-10, height-10, fill="#444466", width=2)
        
        # Upright reference, quarter-turn guides, and neutral band.
        y_180 = 10 + (height - 20) * 0.5
        self.chart_canvas.create_line(30, y_180, width-10, y_180, fill="#666666", width=1, dash=(3,3))
        self.chart_canvas.create_text(18, y_180, text="180°", fill="#555555", font=('Arial', 8))
        y_90 = 10 + (height - 20) * 0.75
        y_270 = 10 + (height - 20) * 0.25
        self.chart_canvas.create_line(30, y_90, width-10, y_90, fill="#dddddd", width=1, dash=(2,2))
        self.chart_canvas.create_line(30, y_270, width-10, y_270, fill="#dddddd", width=1, dash=(2,2))
        y_165 = 10 + (height - 20) * (1 - (165 / 360))
        y_195 = 10 + (height - 20) * (1 - (195 / 360))
        self.chart_canvas.create_rectangle(30, y_195, width-10, y_165,
                                          fill="#e6f6ee", outline="")
        
        # Trace points
        points = []
        history_len = len(self.angle_history)
        for i, angle in enumerate(self.angle_history):
            x = 30 + (i / history_len) * (width - 40)
            # Canvas y coordinates increase downward.
            y = 10 + (height - 20) * (1 - (angle / 360))
            points.append((x, y))
        
        # Colored trace segments
        if len(points) > 1:
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i+1]
                
                angle = self.angle_history[i]
                deviation = abs(angle - 180)
                if deviation > 30:
                    color = "#ff6b6b"
                elif deviation > 15:
                    color = "#b8860b"
                else:
                    color = "#00875a"
                
                self.chart_canvas.create_line(x1, y1, x2, y2, fill=color, width=2)
        
        # Current point and value
        if points:
            x, y = points[-1]
            self.chart_canvas.create_oval(x-4, y-4, x+4, y+4, fill="#ffdd00", outline="#ffffff", width=2)
            
            last_angle = self.angle_history[-1]
            self.chart_canvas.create_text(x, y-15, text=f"{int(last_angle)}°", 
                                         fill="#ffffff", font=('Arial', 8))
        
        # Guide labels
        self.chart_canvas.create_text(15, 15, text="360°", fill="#555555", font=('Arial', 8))
        self.chart_canvas.create_text(15, height-15, text="0°", fill="#555555", font=('Arial', 8))
        self.chart_canvas.create_text(width//2, 5, text="Right lateral  |  Left lateral",
                                     fill="#555555", font=('Arial', 9))
    
    def draw_info_on_frame(self, frame):
        """Draw English measurements and status on the video frame."""
        h, w, _ = frame.shape
        
        # Translucent information panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        y = 30
        cv2.putText(frame, "FRONTAL TRUNK ANALYSIS", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 25
        
        cv2.putText(frame, f"Angle: {int(self.spine_angle)} deg", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20
        
        cv2.putText(frame, f"Right lateral: {int(self.right_flexion)} deg", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)
        y += 20
        
        cv2.putText(frame, f"Left lateral: {int(self.left_flexion)} deg", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        y += 20
        
        status = "Neutral"
        color = (0, 255, 0)
        if self.right_flexion > 30:
            status = "High right lateral"
            color = (0, 0, 255)
        elif self.left_flexion > 30:
            status = "High left lateral"
            color = (0, 0, 255)
        elif self.right_flexion > 15:
            status = "Moderate right lateral"
            color = (0, 255, 255)
        elif self.left_flexion > 15:
            status = "Moderate left lateral"
            color = (0, 255, 255)
            
        cv2.putText(frame, f"Status: {status}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Direction indicator
        if self.right_flexion > 5:
            for i in range(3):
                x = w - 50 - i*20
                y_pos = h//2 - 20 + i*15
                cv2.arrowedLine(frame, (x, y_pos), (x-30, y_pos+20), (0, 200, 255), 2, tipLength=0.3)
            cv2.putText(frame, "RIGHT LATERAL", (w-165, h//2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        
        if self.left_flexion > 5:
            for i in range(3):
                x = w - 50 - i*20
                y_pos = h//2 - 20 + i*15
                cv2.arrowedLine(frame, (x, y_pos), (x+30, y_pos-20), (255, 100, 100), 2, tipLength=0.3)
            cv2.putText(frame, "LEFT LATERAL", (w-155, h//2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)

    def export_data(self):
        """Export all processed frames and their measurements to CSV."""
        if not self.analysis_data:
            messagebox.showinfo("Export Data", "Play the video to collect measurements first.")
            return

        default_name = "lateral_trunk_analysis.csv"
        if self.video_path:
            video_name = self.video_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
            default_name = f"{video_name}_lateral_trunk_analysis.csv"
        file_path = filedialog.asksaveasfilename(
            title="Export analysis data",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv")],
        )
        if not file_path:
            return

        try:
            rows = list(self.analysis_data)
            with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            messagebox.showinfo("Export Complete", f"Data exported to:\n{file_path}")
        except OSError as exc:
            messagebox.showerror("Export Error", f"Could not write the CSV file:\n{exc}")
    
    def display_frame(self, frame):
        """Scale and display a BGR frame in the Tkinter preview."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w, _ = frame_rgb.shape
        aspect_ratio = w / h
        
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        
        if label_width > 10 and label_height > 10:
            if label_width / label_height > aspect_ratio:
                new_height = label_height
                new_width = int(label_height * aspect_ratio)
            else:
                new_width = label_width
                new_height = int(label_width / aspect_ratio)
        else:
            new_width = 640
            new_height = 480
        
        img = cv2.resize(frame_rgb, (new_width, new_height))
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)
    
    def update_frame(self):
        """Read and display the next frame using the selected rotation."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(self.rotate_frame(frame))
    
    def format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def run(self):
        self.root.mainloop()

# ============================
# Run the application
# ============================
if __name__ == "__main__":
    root = tk.Tk()
    app = SpineAnalysisApp(root)
    app.run()
