import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame([
    {"id": 1, "title": "The Dark Knight", "genres": "Action Crime Drama", "keywords": "batman joker vigilante superhero gotham justice dark"},
    {"id": 2, "title": "Inception", "genres": "Action Sci-Fi Thriller", "keywords": "dream heist mind subconscious reality cobb architect spinning totem"},
    {"id": 3, "title": "Interstellar", "genres": "Adventure Drama Sci-Fi", "keywords": "space black hole wormhole gravity time relativity cooper daughter astronaut"},
    {"id": 4, "title": "The Shawshank Redemption", "genres": "Drama", "keywords": "prison escape hope friendship justice andy dufresne red banker innocence"},
    {"id": 5, "title": "The Matrix", "genres": "Action Sci-Fi", "keywords": "simulation cyberpunk hacker reality neo morpheus trinity red pill virtual"},
    {"id": 6, "title": "Pulp Fiction", "genres": "Crime Drama", "keywords": "gangster hitman nonlinear dialogue vincent jules briefcase dance diner"}
])

movies['tags'] = movies['genres'] + " " + movies['keywords']
cv = TfidfVectorizer(stop_words='english')
matrix = cv.fit_transform(movies['tags'])
similarity = cosine_similarity(matrix)

with open('movies_list.pkl', 'wb') as f:
    pickle.dump(movies, f)
with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

print("Project 2: Movies dataset & similarity matrix saved!")