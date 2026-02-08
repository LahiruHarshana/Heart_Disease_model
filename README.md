# HeartGuard AI — Heart Disease Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pCpwELdg-8gX6irQyQ4zYIFH-FyzirLJ?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)

## 📌 Project Overview
**HeartGuard AI** is a deep learning-based system designed to predict the presence of heart disease. This project is developed to answer the challenge posed by the **Kaggle Playground Series - Season 6, Episode 2: Predicting Heart Disease**.

![Kaggle Competition](images/kaggle_competition.png)
*Figure 1: Kaggle Playground Prediction Competition Overview*

The goal is to predict the likelihood of heart disease using a dataset of clinical features. We implement a comprehensive pipeline including:
1.  **Exploratory Data Analysis (EDA)**.
2.  **Data Preprocessing & Scaling**.
3.  **Deep Learning Classification** (TensorFlow/Keras).
4.  **Prediction Web Application** (Streamlit).

## 🚀 Features
-   **Google Colab Ready**: Optimization to run directly in the cloud environment.
-   **Deep Learning Classifier**: Custom Neural Network trained on clinical data.
-   **Interactive Dashboard**: A user-friendly Streamlit app for real-time risk assessment.
-   **Visual Insights**: Comprehensive charts analyzing feature correlations.

## 📸 Visuals & Results

### Data Analysis & Charts
*Below are visualizations of the dataset distributions and correlation matrices generated during the analysis phase.*

![Data Charts](images/charts.png)
*(Run the notebook to generate these visualizations)*

### Application Frontend
*The Streamlit interface allows users to input patient vitals and receive a risk prediction.*

![Streamlit Frontend](images/frontend.png)
*(Screenshot of the HeartGuard AI Web App)*

## 📂 Project Structure
```
├── Heart_Disease_Assignment.ipynb  # Main notebook (Run this in Colab)
├── images/                         # Project screenshots and diagrams
│   ├── kaggle_competition.png
│   ├── charts.png
│   └── frontend.png
├── heart_model.keras               # Saved Model (Generated)
├── scaler.pkl                      # Saved Scaler (Generated)
├── app.py                          # Streamlit App (Generated)
└── README.md                       # Documentation
```

## 🛠️ Installation & Usage (Google Colab)

This project is designed to be easy to run on Google Colab.

### 1. Launch in Colab
Click the "Open in Colab" badge at the top of this README or upload the `Heart_Disease_Assignment.ipynb` file to your Google Drive.

### 2. Run the Analysis
Execute the cells in the notebook. This will:
-   Download/Load the dataset.
-   Perform EDA and display charts.
-   Train the classifier.
-   Save `heart_model.keras` and `scaler.pkl`.
-   Generate the `app.py` file.

### 3. Running the App
The notebook generates an `app.py` file. To run the Streamlit app:

**Option A: Local Execution (Recommended for App)**
1.  Download `app.py`, `heart_model.keras`, and `scaler.pkl` from Colab files.
2.  Install requirements locally: `pip install streamlit pandas numpy tensorflow scikit-learn`
3.  Run:
    ```bash
    streamlit run app.py
    ```

**Option B: Within Colab**
You can run Streamlit inside Colab using a tunnel:
```python
!pip install streamlit -q
!wget -q -O - ipv4.icanhazip.com
!streamlit run app.py & npx localtunnel --port 8501
```
*(Copy the IP address from the output and paste it into the tunnel URL)*

## 📊 Dataset Features
The model operates on standard clinical features provided in the competition dataset:
-   **Age, Sex, Chest Pain Type**
-   **Resting BP, Cholesterol, Fasting Blood Sugar**
-   **ECG Results, Max Heart Rate, Exercise Angina**
-   **ST Depression, Slope, Major Vessels, Thallium Stress Test**

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
