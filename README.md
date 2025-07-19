# 🧠 Employee Salary Prediction App

An intelligent web application built using **Machine Learning** and **Streamlit** to predict whether an employee earns **more than $50K** or not based on key demographic and work-related factors.

---

## 🔍 Problem Statement

In many organizations, identifying high-earning employees plays a critical role in decision-making processes such as promotions, taxation, and resource allocation. However, manually evaluating income ranges based on multiple employee attributes can be time-consuming and error-prone. This project focuses on automating this classification task using a trained machine learning model that predicts if an individual earns more than $50K per year based on selected features.

---

## 🛠️ System Development Approach

### ✅ System Requirements:
- Python 3.x
- Streamlit
- Scikit-learn
- Pickle
- Pandas, NumPy

### ✅ Required Libraries:
- `streamlit`
- `pandas`
- `scikit-learn`
- `numpy`
- `pickle`

These libraries were essential for data processing, model building, and creating the interactive user interface for real-time predictions.

---

## 🔄 Algorithm & Deployment (Step-by-Step Procedure)

1. **Data Collection**: The UCI Adult Income Dataset was used as the base dataset.
2. **Data Preprocessing**: Irrelevant columns were removed; necessary columns were encoded.
3. **Model Training**: A `Gradient Boosting Classifier` was trained on the processed dataset.
4. **Model Saving**: The trained model and encoders were saved using `pickle`.
5. **App Interface**: A responsive and color-themed web app was created using Streamlit.
6. **Deployment**: The app was run locally with `streamlit run app.py`, and is ready for deployment to cloud platforms like Streamlit Sharing or Heroku.

---

## 📸 Result

Below are screen captures of the working app:

### 🖥️ Streamlit UI:
[INSERT IMAGE HERE]

### 📊 Prediction Result:
[INSERT IMAGE HERE]

### 🧠 Model Training Output:
[INSERT IMAGE HERE]

### 🟢 App Running View:
[INSERT IMAGE HERE]

---

## 📈 Conclusion

This Employee Salary Prediction system offers a fast, accurate, and user-friendly tool for classifying individuals into income groups. It simplifies the HR decision-making process and showcases the power of machine learning when integrated into modern web interfaces. While building this project, handling categorical variables and UI design posed challenges, which were addressed through feature encoding and Streamlit layout optimizations.

---

## 🔭 Future Scope

- Host the web app on Streamlit Cloud or AWS for public access.
- Add more predictive features like job level, years of experience, and company rating.
- Create a REST API to integrate this system into larger HR dashboards.
- Include user authentication and admin-level analytics for decision-makers.

---

## 📚 References

- [UCI Machine Learning Repository - Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pickle Python Docs](https://docs.python.org/3/library/pickle.html)

---

⭐ If you found this helpful, give this project a star and feel free to fork or contribute!
