# Smart Face Security System

An AI-powered Smart Face Security System developed using Python, OpenCV, Tkinter, and SQLite.
The system performs real-time face detection and face recognition for user authentication and security monitoring.

It supports:

* Face registration
* Real-time recognition
* Intruder detection
* Alarm alerts
* User database management

---

# Features

* Real-time face detection using webcam
* Face recognition using LBPH algorithm
* User registration with image dataset creation
* Intruder/imposter detection system
* Security alarm for unauthorized access
* SQLite database integration
* User management system (add/delete users)
* Automatic model training and retraining
* GUI-based desktop application using Tkinter

---

# Technologies Used

* Python
* OpenCV
* NumPy
* Tkinter
* SQLite3
* PIL (Pillow)
* Threading
* Haar Cascade Classifier
* LBPH Face Recognizer

---

# Project Structure

```bash
Smart-Face-Security-System/
│
├── dataset/                  # Stored face images
├── trainer.yml               # Trained face recognition model
├── face_database.db          # SQLite database
├── face.py                   # Main application file
├── README.md                 # Project documentation
├── .gitignore                # Ignored files
└── requirements.txt          # Required libraries
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/your-username/smart-face-security-system.git
```

---

## 2. Navigate to the project folder

```bash
cd smart-face-security-system
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---


# How to Run

```bash
python face.py
```

---

# How the System Works

## User Registration

1. Enter user name
2. Click “Start Registration”
3. System captures multiple face samples
4. Images are stored in dataset folder
5. Model automatically trains the recognizer

---

## Security Monitoring

1. Start Security Mode
2. Webcam continuously scans faces
3. Authorized users get access
4. Unknown users trigger:

   * Intruder warning
   * Alarm sound
   * Red detection alert

---

# Face Recognition Method

The system uses:

* Haar Cascade for face detection
* LBPH (Local Binary Pattern Histogram) for face recognition

LBPH is lightweight, fast, and suitable for real-time applications.

---

# Database Management

SQLite database is used to:

* store user information
* manage registered users
* retrieve authentication details

Users can also be deleted from the interface, which automatically updates the trained model.

---

# GUI Interface

The application includes:

* Registration panel
* Security monitoring controls
* Live webcam feed
* User database viewer
* Real-time status display

Built using Tkinter.

---

# Applications

* Smart Attendance Systems
* Office Security
* Home Security
* Authentication Systems
* AI Surveillance Projects
* College Mini/Major Projects

---

# Future Improvements

* Deep Learning-based face recognition
* Cloud database integration
* Email/SMS alerts
* Multi-camera support
* Web application deployment
* Face mask detection
* Mobile app integration

---


# Author

Sumit Asnora
BCA Graduate | AI & Full Stack Enthusiast

---

# License

This project is licensed under the MIT License.
