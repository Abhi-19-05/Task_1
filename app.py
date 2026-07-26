import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------
# Load Pickle Model
# ---------------------------
@st.cache_resource
def load_model():
    with open("movie_recommender.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

user_movie_matrix = model["user_movie_matrix"]
user_similarity_df = model["user_similarity_df"]
item_similarity_df = model["item_similarity_df"]
movie_titles = model["movie_titles"]

# ---------------------------
# Recommendation Function
# ---------------------------
def recommend_movies(user_id, top_n=5):

    if user_id not in user_similarity_df.index:
        return pd.DataFrame()

    # Top similar users
    similar_users = (
        user_similarity_df.loc[user_id]
        .sort_values(ascending=False)
        .iloc[1:11]
    )

    # Movies already watched
    watched_movies = (
        user_movie_matrix.loc[user_id]
        [user_movie_matrix.loc[user_id] > 0]
        .index
    )

    recommendations = {}

    for sim_user, similarity in similar_users.items():

        sim_ratings = user_movie_matrix.loc[sim_user]

        for movie_id, rating in sim_ratings.items():

            if rating > 0 and movie_id not in watched_movies:

                recommendations[movie_id] = (
                    recommendations.get(movie_id, 0)
                    + rating * similarity
                )

    recommendations = sorted(
        recommendations.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    result = []

    for movie_id, score in recommendations:

        result.append({
            "Movie ID": movie_id,
            "Movie Title": movie_titles.get(movie_id, "Unknown"),
            "Score": round(score, 3)
        })

    return pd.DataFrame(result)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎬 Movie Recommendation System")

st.markdown(
"""
This application recommends movies using **User-Based Collaborative Filtering**
with **Cosine Similarity**.
"""
)

st.sidebar.header("Settings")

user_id = st.sidebar.number_input(
    "User ID",
    min_value=int(user_movie_matrix.index.min()),
    max_value=int(user_movie_matrix.index.max()),
    value=1
)

top_n = st.sidebar.slider(
    "Top N Recommendations",
    min_value=1,
    max_value=20,
    value=5
)

if st.sidebar.button("Recommend"):

    recommendations = recommend_movies(user_id, top_n)

    if recommendations.empty:
        st.error("No recommendations found.")
    else:
        st.subheader(f"Top {top_n} Recommendations for User {user_id}")
        st.dataframe(recommendations, use_container_width=True, hide_index=True)

# ---------------------------
# Footer
# ---------------------------
st.divider()

col1, col2 = st.columns(2)

col1.metric("Users", len(user_movie_matrix.index))
col2.metric("Movies", len(user_movie_matrix.columns))

st.caption("Built using Python, Pandas, Scikit-learn and Streamlit")