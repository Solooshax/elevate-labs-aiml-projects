import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

print("1. Loading dataset...")
df = pd.read_csv('fake_or_real_news.csv')

# Drop NA rows
df = df.dropna(subset=['title', 'label'])

# Training on both Title and Text with higher weight to Title
df['combined'] = df['title'].astype(str) + " " + df['title'].astype(str) + " " + df['text'].astype(str).str[:500]
df['clean'] = df['combined'].apply(clean_text)

X = df['clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("2. Fitting TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2, max_features=30000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("3. Training Multinomial Naive Bayes Classifier...")
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vec, y_train)

acc = accuracy_score(y_test, model.predict(X_test_vec))
print(f"✅ Accuracy: {acc * 100:.2f}%")

print("4. Saving model and vectorizer...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Training finished successfully!")