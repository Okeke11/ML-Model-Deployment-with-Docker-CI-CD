# 📧 Spam Classifier API (MLOps Docker Project)

A production-ready **Machine Learning Inference API** capable of detecting spam emails in real-time. This project demonstrates an end-to-end MLOps workflow: training a Naive Bayes model, exposing it via **FastAPI**, and containerizing the application with **Docker** for consistent deployment across environments.

## 🚀 Key Features
* **Machine Learning:** Scikit-learn pipeline using CountVectorizer and Multinomial Naive Bayes.
* **API:** High-performance REST API built with FastAPI.
* **Containerization:** Fully dockerized application (Linux/Alpine based) for reproducibility.
* **Documentation:** Automatic interactive API documentation via Swagger UI.

## 🛠️ Tech Stack
* **Python 3.9**
* **FastAPI** (Web Framework)
* **Scikit-learn** (ML Library)
* **Pandas** (Data Manipulation)
* **Docker** (Containerization)
* **Uvicorn** (ASGI Server)

---

## ⚙️ Setup & Installation

### Option A: Run Locally (Python)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/mlops-spam-api.git](https://github.com/YOUR_USERNAME/mlops-spam-api.git)
   cd mlops-spam-api

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Train the model:**
   This script generates the `spam_model.pkl` artifact.
   ```bash
   python train_model.py

4. **Running the Server**

 **Option A: Run Locally**

Run the server using Uvicorn:

```bash
uvicorn main:app --reload
