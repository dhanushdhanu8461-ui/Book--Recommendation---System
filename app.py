# Streamlit app for Book Recommendation (hybrid user-user + item-item)
# Place this file at the repo root. Launch with: streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import time

st.set_page_config(page_title="Book Recommender", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(books_path="Books.csv", ratings_path="Ratings.csv", users_path="Users.csv"):
    # Read CSVs (use latin1 encoding as in notebooks)
    books = pd.read_csv(books_path, encoding="latin1")
    ratings = pd.read_csv(ratings_path, encoding="latin1")
    users = pd.read_csv(users_path, encoding="latin1", engine="python", on_bad_lines="skip")

    # Basic cleaning consistent with notebooks
    # remove rows with missing book meta (very few)
    books = books.dropna()
    # remove ratings with missing fields (should be none)
    ratings = ratings.dropna()
    # fill user age with median
    if "Age" in users.columns:
        median_age = users["Age"].median()
        users["Age"] = users["Age"].fillna(median_age)

    # Merge title into ratings so we carry a book title
    ratings_up = ratings.merge(books[["ISBN", "Book-Title"]], on="ISBN", how="left")
    # drop ratings with missing titles
    ratings_up = ratings_up.dropna(subset=["Book-Title"]).reset_index(drop=True)

    return books, ratings_up, users

@st.cache_resource(show_spinner=False)
def prepare_models(ratings_up):
    """
    Build categorical encodings, sparse matrices and fit KNN models.
    Returns mapping dicts and trained NearestNeighbors instances.
    """
    # create category codes
    ratings_up = ratings_up.copy()
    ratings_up["user_index"] = ratings_up["User-ID"].astype("category").cat.codes
    ratings_up["book_index"] = ratings_up["Book-Title"].astype("category").cat.codes

    # mappings
    user_id_mapping = dict(enumerate(ratings_up["User-ID"].astype("category").cat.categories))
    user_index_mapping = {v: k for k, v in user_id_mapping.items()}

    book_title_mapping = dict(enumerate(ratings_up["Book-Title"].astype("category").cat.categories))
    book_index_mapping = {v: k for k, v in book_title_mapping.items()}

    # Build sparse user-book and book-user matrices
    user_book_matrix = csr_matrix(
        (ratings_up["Book-Rating"].astype(float), (ratings_up["user_index"], ratings_up["book_index"]))
    )
    book_user_matrix = csr_matrix(
        (ratings_up["Book-Rating"].astype(float), (ratings_up["book_index"], ratings_up["user_index"]))
    )

    # Fit KNN models (cosine)
    user_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=6, n_jobs=-1)
    user_knn.fit(user_book_matrix)

    book_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=6, n_jobs=-1)
    book_knn.fit(book_user_matrix)

    return {
        "ratings_up": ratings_up,
        "user_book_matrix": user_book_matrix,
        "book_user_matrix": book_user_matrix,
        "user_knn": user_knn,
        "book_knn": book_knn,
        "user_id_mapping": user_id_mapping,
        "user_index_mapping": user_index_mapping,
        "book_title_mapping": book_title_mapping,
        "book_index_mapping": book_index_mapping,
    }

def get_similar_users(user_index, user_knn, user_book_matrix, user_id_mapping, n=5):
    distances, indices = user_knn.kneighbors(user_book_matrix[user_index], n_neighbors=n+1)
    # indices includes the user itself at position 0
    sim_indices = [i for i in indices.flatten() if i != user_index][:n]
    return [user_id_mapping[i] for i in sim_indices]

def recommend_from_similar_users(target_user_id, similar_user_ids, ratings_up, min_rating=5, top_n=10):
    sim_books = ratings_up[ratings_up["User-ID"].isin(similar_user_ids)]
    target_books = set(ratings_up[ratings_up["User-ID"] == target_user_id]["Book-Title"])
    # select books liked by similar users and not read by target user
    recs = (
        sim_books[sim_books["Book-Rating"] >= min_rating]
        .loc[~sim_books["Book-Title"].isin(target_books)]
        .groupby("Book-Title")["Book-Rating"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return recs

def recommend_similar_items(target_user_books, book_knn, book_user_matrix, book_index_mapping, book_title_mapping, ratings_up, top_n=10):
    similar_books_set = set()
    for book in target_user_books:
        if book in book_index_mapping:
            book_id = book_index_mapping[book]
            distances, indices = book_knn.kneighbors(book_user_matrix[book_id], n_neighbors=4)
            for idx in indices.flatten()[1:]:
                similar_books_set.add(book_title_mapping.get(idx))
    # Rank them by average rating in dataset
    if not similar_books_set:
        return pd.Series(dtype=float)
    similar_books_ratings = (
        ratings_up[ratings_up["Book-Title"].isin(similar_books_set)]
        .groupby("Book-Title")["Book-Rating"]
        .mean()
        .sort_values(ascending=False)
    )
    return similar_books_ratings.head(top_n)

def hybrid_recommend(target_user_id, models, top_n=10):
    ratings_up = models["ratings_up"]
    user_knn = models["user_knn"]
    book_knn = models["book_knn"]
    user_book_matrix = models["user_book_matrix"]
    book_user_matrix = models["book_user_matrix"]
    user_id_mapping = models["user_id_mapping"]
    user_index_mapping = models["user_index_mapping"]
    book_title_mapping = models["book_title_mapping"]
    book_index_mapping = models["book_index_mapping"]

    if target_user_id not in user_index_mapping:
        return {"error": "User not found in dataset."}

    target_user_index = user_index_mapping[target_user_id]
    similar_users = get_similar_users(target_user_index, user_knn, user_book_matrix, user_id_mapping, n=5)

    target_user_data = ratings_up[ratings_up["User-ID"] == target_user_id][["Book-Title", "Book-Rating"]]
    target_user_books = set(target_user_data["Book-Title"])

    rec_user = recommend_from_similar_users(target_user_id, similar_users, ratings_up, min_rating=5, top_n=top_n)
    rec_items = recommend_similar_items(target_user_books, book_knn, book_user_matrix, book_index_mapping, book_title_mapping, ratings_up, top_n=top_n)

    # Combine (union) leaving ordering by user-based rec first then items
    combined = list(rec_user.index) + [b for b in rec_items.index if b not in rec_user.index]
    return {
        "similar_users": similar_users,
        "target_user_books": target_user_data,
        "user_cf": rec_user,
        "item_cf": rec_items,
        "combined": combined[:top_n],
    }

# ---- Streamlit UI ----
st.title("Book Recommendation (Hybrid CF)")
st.write("This demo uses a hybrid approach (user–user + item–item) similar to the notebooks in the repo. Building models on large datasets may take time and memory. The app caches models between runs.")

with st.sidebar:
    st.header("Data & Settings")
    use_sample = st.checkbox("Load sample of users (fast)", value=False)
    top_n = st.slider("Number of recommendations", 1, 25, 10)
    st.write("If model build is slow, try the 'sample' option.")

# Load data
with st.spinner("Loading data..."):
    try:
        books, ratings_up, users = load_data()
    except FileNotFoundError as e:
        st.error(f"Could not find CSV files. Ensure Books.csv, Ratings.csv and Users.csv are in the app folder. {e}")
        st.stop()

# optionally sample a subset (faster startup)
if use_sample:
    # choose top users by number of ratings
    top_users = ratings_up["User-ID"].value_counts().nlargest(5000).index
    ratings_up = ratings_up[ratings_up["User-ID"].isin(top_users)].reset_index(drop=True)
    st.info("Using a sample: top 5k most active users (faster).")

# Prepare models (cached)
with st.spinner("Preparing recommendation models (one-time)... this can take a minute on large datasets"):
    t0 = time.time()
    models = prepare_models(ratings_up)
    t1 = time.time()
st.success(f"Models ready (built in {t1-t0:.1f}s).")

# Select user
all_user_ids = list(models["user_id_mapping"].values())
# show a subset for quick select in UI
sample_user_ids = all_user_ids[:5000] if len(all_user_ids) > 5000 else all_user_ids
selected_user = st.selectbox("Select User-ID (or paste an ID)", sample_user_ids)

if selected_user is None:
    st.stop()

if st.button("Get recommendations"):
    with st.spinner("Generating recommendations..."):
        res = hybrid_recommend(target_user_id=selected_user, models=models, top_n=top_n)
    if "error" in res:
        st.error(res["error"])
    else:
        st.subheader(f"Similar users to {selected_user}")
        st.write(res["similar_users"])

        st.subheader(f"Books read by user {selected_user}")
        if res["target_user_books"].empty:
            st.write("User has not rated any books yet.")
        else:
            st.dataframe(res["target_user_books"].reset_index(drop=True))

        st.subheader("Top recommendations (User–User CF)")
        if res["user_cf"].empty:
            st.write("No high-rated books from similar users.")
        else:
            st.table(res["user_cf"].reset_index().rename(columns={"Book-Rating":"avg_rating"}))

        st.subheader("Top recommendations (Item–Item CF)")
        if res["item_cf"].empty:
            st.write("No similar items found.")
        else:
            st.table(res["item_cf"].reset_index().rename(columns={"Book-Rating":"avg_rating"}))

        st.subheader("Final combined recommendations")
        if not res["combined"]:
            st.write("No recommendations available.")
        else:
            st.write(pd.DataFrame(res["combined"], columns=["Book-Title"]))

st.markdown("---")
st.write("Notes:")
st.write("- This app builds KNN models in-memory. For production, precompute models and persist them (joblib/pickle) or use a dedicated recommender backend.")
st.write("- If your dataset is very large, run the sampled option or precompute with an offline job and load artifacts here.")
