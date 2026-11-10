import streamlit as st
import pickle

st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")

try:
    movies = pickle.load(open('movies_list.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
except FileNotFoundError:
    st.error("Please run `python recommender.py` first!")
    st.stop()

selected_movie = st.selectbox("Select a Movie:", movies['title'].values)

def recommend(movie_title):
    index = movies[movies['title'] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    return [(movies.iloc[i[0]]['title'], movies.iloc[i[0]]['genres']) for i in distances[1:4]]

if st.button("Get Recommendations"):
    recs = recommend(selected_movie)
    st.subheader(f"Movies similar to '{selected_movie}':")
    for title, genres in recs:
        st.write(f"🎥 **{title}** — *{genres}*")