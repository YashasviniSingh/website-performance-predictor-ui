# 🌐 Website Performance Predictor

A Machine Learning based web application that analyzes a website URL and predicts its performance using multiple behavioral and engagement features.

---

## 📌 Overview

This project combines **Machine Learning + Web Development** to evaluate how well a website performs.

User ek website URL input deta hai, aur system:

* Relevant features generate karta hai
* ML model se prediction karta hai
* Final performance score show karta hai

---

## ⚡ Features

* 🔍 Website URL analysis
* 🤖 ML-based prediction system
* 📊 Feature-driven evaluation
* 🎯 Clean and responsive UI
* 🟢 Performance classification (Good / Poor)

---

## 🧠 How It Works

1. User enters website URL
2. System generates feature values
3. Features ML model ko diye jaate hain
4. Model performance predict karta hai
5. Result UI me show hota hai

---

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS
* **Backend:** Python (Flask)
* **ML:** Scikit-learn, NumPy

---

## 📁 Project Structure

* `app.py` – Flask backend
* `model.pkl` – Trained ML model
* `feature_means.json` – Feature reference values
* `templates/` – HTML files
* `static/` – CSS files
* `requirements.txt` – Dependencies

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Open: http://localhost:10000

---

## 📊 Output

* Low score → Poor performance
* High score → Good performance

---

## 👩‍💻 Author

Yashasvini Singh 
Tanishq Nain

## 🌐 Live Demo
https://website-performance-predictor-web.onrender.com
