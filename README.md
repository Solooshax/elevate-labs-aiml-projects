# Elevate Labs - AI/ML Internship Projects

This repository contains the complete implementation, datasets, model pipelines, and interactive web dashboards for the **2 projects** completed as part of the Elevate Labs AI/ML Internship Phase.

---

## 📂 Repository Structure

```text
elevate-labs-aiml-projects/
├── Project1_FakeNewsClassifier/       # Task 1: Fake/Real News Classification (NLP)
│   ├── app.py                         # Streamlit Interactive Web Application
│   ├── train.py                       # Preprocessing, Vectorization & Model Training
│   ├── requirements.txt               # Dependencies for Project 1
│   ├── model.pkl                      # Serialized Trained Model
│   ├── vectorizer.pkl                 # Fitted TF-IDF Vectorizer
│   └── fake_or_real_news.csv          # Training Dataset
│
├── Project2_MovieRecommender/          # Task 2: Content-Based Movie Recommendation System
│   ├── app.py                         # Streamlit Interactive Web Application
│   ├── recommender.py                 # Feature Tagging, Cosine Similarity & Serialization
│   ├── requirements.txt               # Dependencies for Project 2
│   ├── movies_list.pkl                # Preprocessed Movies Dataset
│   └── similarity.pkl                 # Pairwise Cosine Similarity Matrix
│
└── README.md                          # Comprehensive Project Guide & Setup