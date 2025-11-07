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
┣ 📜 best_ddd_model_realistic.h5  # Trained Keras model (download separately)
┣ 📜 requirements.txt             # Dependencies
┗ 📜 README.md                    # Documentation

````

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Ravikumar-2016/Trained-Model-DDD-System.git
cd DDD-Trained-Model
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

## 📥 Download the Trained Model

This project uses a pre-trained model file:
`best_ddd_model_realistic.h5`

### 🔗 Download Link

👉 [Download Model from Google Drive](https://drive.google.com/file/d/1fgwikiY9UOI9dOvhbswGxay9L7qCf1HO/view?usp=drive_link)

After downloading:

1. Place the file in the same folder as `DrowsinessDetection.py`.
2. The final structure should look like:

   ```
   DDD-Trained-Model/
   ┣ DrowsinessDetection.py
   ┣ best_ddd_model_realistic.h5
   ┣ requirements.txt
   ┗ README.md
   ```

> 💡 Tip: If you want automatic download, you can add this code snippet inside your Python file:

```python
import os, gdown

model_path = "best_ddd_model_realistic.h5"
url = "https://drive.google.com/file/d/1fgwikiY9UOI9dOvhbswGxay9L7qCf1HO/view?usp=drive_link"

if not os.path.exists(model_path):
    print("Downloading trained model...")
    gdown.download(url, model_path, quiet=False)
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
| gdown         | latest    | Download model from Google Drive          |

---

## 💡 Future Enhancements

* 🔊 Integrate sound alerts or seat vibration.
* 😴 Extend model for yawning detection.
* 📱 Optimize for mobile and edge devices (TensorFlow Lite).
* 🧭 Add GUI dashboard using Tkinter or PyQt.

---