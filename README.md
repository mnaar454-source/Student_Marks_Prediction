# 🎓 Student Marks Prediction

> An end-to-end Machine Learning project for predicting students' Mathematics scores from demographic and academic performance data, with an interactive Streamlit web application.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.2-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)

---

## 🚀 Live Demo

### 🌐 Streamlit Application
https://studentmarksprediction-tfdrm6iwem2fcekwhk6jbr.streamlit.app/

The deployed application provides an interactive interface for entering student information and generating a Mathematics score prediction using the trained Machine Learning model.

---

## 📌 Project Overview

**Student Marks Prediction** is an end-to-end Machine Learning project developed as part of a **B.Tech CSE (AI & ML)** academic project.

The project uses the Kaggle `exams.csv` dataset to build a regression-based prediction system for estimating a student's **Mathematics score** from demographic information and other academic scores.

The complete workflow covers:

- Dataset loading and inspection
- Data cleaning
- Exploratory Data Analysis (EDA)
- Feature selection
- Train/test splitting
- Leakage-safe preprocessing
- Machine Learning model training
- Model comparison
- Model evaluation
- Model serialization using Joblib
- Student score prediction
- Interactive Streamlit deployment

---

## 🎯 Objective

The main objective of this project is to develop a reusable Machine Learning pipeline that can:

1. Process real student examination data.
2. Prepare numerical and categorical features for Machine Learning.
3. Train and compare regression models.
4. Evaluate model performance using standard regression metrics.
5. Save the trained model for reuse.
6. Provide predictions through an interactive web application.

---

## 📊 Dataset

The project uses the Kaggle `exams.csv` dataset.

The dataset contains the following attributes:

| Feature | Description |
|---|---|
| `gender` | Student gender |
| `race/ethnicity` | Student race/ethnicity group |
| `parental level of education` | Parent/guardian education level |
| `lunch` | Lunch program/type |
| `test preparation course` | Test preparation course status |
| `math score` | Mathematics examination score |
| `reading score` | Reading examination score |
| `writing score` | Writing examination score |

### Target Variable

```text
math score
