# 🧠 Driver Drowsiness Detection (Deep Learning Model)

This project implements a **Driver Drowsiness Detection System** using a **trained deep learning model** built on TensorFlow and OpenCV.  
The model identifies drowsy states in drivers based on eye and facial landmarks in real-time.

---

## 🚀 Key Highlights
- 🤖 Deep Learning model (`best_ddd_model_realistic.h5`) trained on a real-world dataset.  
- 🎥 Real-time video feed analysis using OpenCV.  
- ⏰ Detects drowsiness and can trigger alerts.  
- 🔧 TensorFlow-based inference pipeline for high accuracy.

---

## 🧩 Project Structure
```

📦 DDD-Trained-Model
┣ 📜 DrowsinessDetection.py       # Main detection script
┣ 📜 best_ddd_model_realistic.h5  # Trained Keras model
┣ 📜 requirements.txt             # Dependencies
┗ 📜 README.md                    # Documentation

````

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/Trained-Driver-Drowsiness-Detection.git
cd Trained-Driver-Drowsiness-Detection
````

### 2️⃣ Create and Activate a Virtual Environment

```bash
python -m venv venv
```

#### 🪟 Windows

```bash
venv\Scripts\activate
```

#### 🐧 macOS/Linux

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the detection script:

```bash
python DrowsinessDetection.py
```

Make sure your webcam is connected.
When the model detects drowsiness, it will print or trigger the respective alert in real-time.

---

## 🧠 Requirements

| Library       | Version   | Purpose                                   |
| ------------- | --------- | ----------------------------------------- |
| opencv-python | 4.10.0.84 | Real-time image capture and preprocessing |
| numpy         | 2.1.3     | Matrix operations                         |
| tensorflow    | 2.20.0    | Deep learning inference                   |

---

## 💡 Future Enhancements

* Integrate sound alerts or seat vibration.
* Extend model for yawning detection.
* Optimize for mobile and edge devices (TensorFlow Lite).
* Add GUI dashboard using Tkinter or PyQt.

---