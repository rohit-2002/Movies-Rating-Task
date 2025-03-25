# 🎬 IMDb Movie Rating Predictor

## 📌 Objective
Build a machine learning model to predict IMDb movie ratings based on metadata like genre, director, actors, duration, and more.

---
## 🌐 Live Demo

The project is deployed and live!  
👉 **[Click here to use the IMDb Movie Rating Predictor](https://movies-rating.streamlit.app/)**  

---
## 🛠️ Features Used
- Year of release
- Duration (in minutes)
- Number of votes
- Director's average rating
- Genre’s average rating
- Frequency of appearance of director, genre, actors

---

## 🧹 Preprocessing
- Removed rows with missing `Rating`, `Genre`, `Director`, etc.
- Extracted numeric values from `Year`, `Duration`
- Cleaned commas from `Votes` and converted to int
- Feature engineered average ratings and frequencies

---

## 📈 Model Used
- **RandomForestRegressor** from scikit-learn
- Evaluation Metrics:
  - **MAE**: `x.xxx`
  - **RMSE**: `x.xxx`
  - **R²**: `x.xxx`

---

## 🚀 How to Run
1. Install dependencies:  
   `pip install -r requirements.txt`
2. Train model:  
   `python train_model.py`
3. Launch app:  
   `streamlit run app.py`

---

## 📂 Files
- `IMDb Movies India.csv` – Raw dataset
- `train_model.py` – Model training script
- `model.pkl` – Trained model
- `features.pkl` – Features used in training
- `app.py` – Streamlit app UI
- `README.md` – Project documentation

---

## 📊 Output
User can select movie details and get predicted IMDb rating instantly via an interactive Streamlit UI.
