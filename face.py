import cv2
import numpy as np
import os
import threading
import winsound
import time
import sys
import sqlite3
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

# --- CONSTANTS & CONFIG ---
DB_FILE = "face_database.db"
DATASET_DIR = "dataset"
TRAINER_FILE = "trainer.yml"
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Ensure directories exist
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def add_user(self, name):
        self.cursor.execute("INSERT INTO users (name) VALUES (?)", (name,))
        self.conn.commit()
        return self.cursor.lastrowid

    def delete_user(self, user_id):
        self.cursor.execute("DELETE FROM users WHERE id=?", (user_id,))
        self.conn.commit()

    def get_user_name(self, user_id):
        self.cursor.execute("SELECT name FROM users WHERE id=?", (user_id,))
        result = self.cursor.fetchone()
        return result[0] if result else "Unknown"

    def get_all_users(self):
        self.cursor.execute("SELECT id, name, created_at FROM users")
        return self.cursor.fetchall()

class FaceAlarmApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Face Security System")
        self.root.geometry("1000x600")
        self.root.configure(bg="#2c3e50")

        # Initialize CV & DB
        self.db = DatabaseManager()
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        self.check_opencv_contrib()
        
        self.cap = None
        self.is_running = False
        self.mode = "IDLE"  # IDLE, REGISTER, MONITOR
        self.alarm_active = False
        self.register_count = 0
        self.register_max = 30
        self.current_user_id = None
        
        self.setup_ui()

    def check_opencv_contrib(self):
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            # Load trainer if exists
            if os.path.exists(TRAINER_FILE):
                self.recognizer.read(TRAINER_FILE)
        except AttributeError:
            messagebox.showerror("Critical Error", "OpenCV 'face' module missing.\nPlease run: pip install opencv-contrib-python")
            sys.exit(1)

    def setup_ui(self):
        # --- LEFT PANEL (CONTROLS) ---
        left_panel = tk.Frame(self.root, bg="#34495e", width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)

        tk.Label(left_panel, text="SECURITY CONTROL", font=("Arial", 16, "bold"), bg="#34495e", fg="white", pady=20).pack()

        # Registration Section
        reg_frame = tk.LabelFrame(left_panel, text="New User Registration", bg="#34495e", fg="white", padx=10, pady=10)
        reg_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(reg_frame, text="Name:", bg="#34495e", fg="white").pack(anchor="w")
        self.entry_name = tk.Entry(reg_frame)
        self.entry_name.pack(fill=tk.X, pady=5)

        self.btn_register = tk.Button(reg_frame, text="Start Registration", bg="#3498db", fg="white", command=self.start_registration)
        self.btn_register.pack(fill=tk.X, pady=5)

        # Security Section
        sec_frame = tk.LabelFrame(left_panel, text="Security Monitor", bg="#34495e", fg="white", padx=10, pady=10)
        sec_frame.pack(fill=tk.X, padx=10, pady=10)

        self.btn_monitor = tk.Button(sec_frame, text="START SECURITY MODE", bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), command=self.toggle_security)
        self.btn_monitor.pack(fill=tk.X, pady=10)

        # Database View Section
        db_frame = tk.LabelFrame(left_panel, text="Database", bg="#34495e", fg="white", padx=10, pady=10)
        db_frame.pack(fill=tk.X, padx=10, pady=10, expand=True)
        
        self.user_list = tk.Listbox(db_frame, height=10)
        self.user_list.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.btn_delete = tk.Button(db_frame, text="Delete Selected User", bg="#c0392b", fg="white", command=self.delete_selected_user)
        self.btn_delete.pack(fill=tk.X)

        self.refresh_user_list()

        # --- RIGHT PANEL (VIDEO) ---
        right_panel = tk.Frame(self.root, bg="black")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(right_panel, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        self.status_label = tk.Label(right_panel, text="System Idle", bg="#2c3e50", fg="white", font=("Arial", 14), pady=10)
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)

    def refresh_user_list(self):
        self.user_list.delete(0, tk.END)
        users = self.db.get_all_users()
        for u in users:
            self.user_list.insert(tk.END, f"ID: {u[0]} | {u[1]}")

    def delete_selected_user(self):
        selection = self.user_list.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a user to delete.")
            return

        # Confirm Deletion
        if not messagebox.askyesno("Confirm Delete", "Are you sure? This will delete the user and their face data."):
            return

        try:
            # Parse ID from listbox "ID: 1 | Name"
            item_text = self.user_list.get(selection)
            user_id = int(item_text.split("|")[0].replace("ID:", "").strip())

            # 1. Delete from DB
            self.db.delete_user(user_id)

            # 2. Delete Images
            files_deleted = 0
            for f in os.listdir(DATASET_DIR):
                if f.startswith(f"User.{user_id}."):
                    os.remove(os.path.join(DATASET_DIR, f))
                    files_deleted += 1

            # 3. Retrain system to remove ID from model
            self.status_label.config(text="Updating System...", bg="#e67e22")
            self.train_system(is_deletion=True)
            
            messagebox.showinfo("Success", f"User deleted. System updated.")
            self.refresh_user_list()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete: {e}")

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.process_video_loop()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image="")

    def start_registration(self):
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showwarning("Input Error", "Please enter a name first.")
            return
        
        self.current_user_id = self.db.add_user(name)
        self.register_count = 0
        self.mode = "REGISTER"
        self.status_label.config(text=f"Registering: {name} (Look at Camera)", bg="#f1c40f")
        self.start_camera()

    def toggle_security(self):
        if self.mode == "MONITOR":
            # Stop Monitoring
            self.mode = "IDLE"
            self.btn_monitor.config(text="START SECURITY MODE", bg="#e74c3c")
            self.status_label.config(text="System Idle", bg="#2c3e50")
            self.stop_camera()
        else:
            # Start Monitoring
            if not os.path.exists(TRAINER_FILE):
                messagebox.showerror("Error", "No training data found. Please register a face first.")
                return
            
            # Reload recognizer to ensure latest data
            self.recognizer.read(TRAINER_FILE)
            
            self.mode = "MONITOR"
            self.btn_monitor.config(text="STOP SECURITY MODE", bg="#27ae60")
            self.status_label.config(text="Scanning for Imposters...", bg="#e74c3c")
            self.start_camera()

    def train_system(self, is_deletion=False):
        # Gather data from dataset folder
        face_samples = []
        ids = []
        
        image_paths = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)]
        
        for image_path in image_paths:
            try:
                pil_img = Image.open(image_path).convert('L') # Convert to gray
                img_numpy = np.array(pil_img, 'uint8')
                
                # Extract ID from filename: User.1.5.jpg
                id = int(os.path.split(image_path)[-1].split(".")[1])
                
                faces = self.face_cascade.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(id)
            except Exception as e:
                pass
        
        if len(face_samples) > 0:
            self.recognizer.train(face_samples, np.array(ids))
            self.recognizer.write(TRAINER_FILE)
            if not is_deletion:
                messagebox.showinfo("Success", "Training Complete! Face Registered.")
        else:
            # If no data left (all users deleted), reset the trainer
            if os.path.exists(TRAINER_FILE):
                os.remove(TRAINER_FILE)
            self.recognizer = cv2.face.LBPHFaceRecognizer_create() # Reset memory
            if not is_deletion:
                messagebox.showerror("Error", "Could not train system. No sufficient data.")

        self.refresh_user_list()
        self.stop_camera()
        self.mode = "IDLE"
        self.status_label.config(text="System Ready", bg="#2c3e50")

    def play_alarm(self):
        if not self.alarm_active:
            self.alarm_active = True
            try:
                winsound.Beep(2500, 500)
            except:
                pass
            self.alarm_active = False

    def process_video_loop(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # Logic based on Mode
            if self.mode == "REGISTER":
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    self.register_count += 1
                    # Save image to dataset
                    filename = f"User.{self.current_user_id}.{self.register_count}.jpg"
                    cv2.imwrite(f"{DATASET_DIR}/{filename}", gray[y:y+h, x:x+w])
                    
                    cv2.putText(frame, f"Capturing: {self.register_count}/{self.register_max}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if self.register_count >= self.register_max:
                    self.train_system()
                    return # Stop loop here to train

            elif self.mode == "MONITOR":
                for (x, y, w, h) in faces:
                    try:
                        id, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                        
                        # Confidence: 0 is perfect match, 100+ is bad match
                        if confidence < 55:
                            name = self.db.get_user_name(id)
                            color = (0, 255, 0)
                            msg = f"ACCESS GRANTED: {name}"
                            self.status_label.config(text=msg, bg="green")
                        else:
                            name = "IMPOSTER"
                            color = (0, 0, 255)
                            msg = "WARNING: IMPOSTER DETECTED"
                            self.status_label.config(text=msg, bg="red")
                            threading.Thread(target=self.play_alarm).start()
                            
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        
                    except Exception as e:
                        pass

            # Convert to Tkinter Image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.process_video_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAlarmApp(root)
    root.mainloop()