import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

# what file does
# 1. reads in cleaned movie dataset
# 2. builds tf-idf features from movie text descriptions
# 3. runs nmf topic modeling to discover high-level themes
# 4. assigns each movie a dominant topic
# 5. generates explanation sentences for recommendations


df = pd.read_csv("cleaned_imdb_dataset.csv")

custom_stopwords = [
    "plot", "add", "short", "wraps", "untitled", "summary",
    "young", "man", "woman", "girl", "boy", "life", "story", "tells", "follows",
    "finds", "look", "new", "past", "years", "time", "day", "people", "lives",
    "series", "film", "movie", "based"
]
all_stopwords = list(ENGLISH_STOP_WORDS.union(custom_stopwords))


# tf-idf converts converts movie text into numerical vectors
docs = df["document_text_v2"].fillna("")

vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.60,
    stop_words=all_stopwords
)

tfidf = vectorizer.fit_transform(docs)

# nmf groups words into hidden topics
nmf = NMF(
    n_components=10,
    random_state=42,
    init="nndsvda",
    max_iter=500
)

W = nmf.fit_transform(tfidf) #movie-topic matrix, given a score
H = nmf.components_ # topic-word matrix, each matrix gets important keywords

feature_names = vectorizer.get_feature_names_out() #list of tf-idf vocabilary words


# assign topic to each movie
df["dominant_topic"] = W.argmax(axis=1)
df["topic_strength"] = W.max(axis=1)

# examples for now
topic_names = {
    0: "drama / school / biography",
    1: "comedy / stand-up",
    2: "animated action / adventure",
    3: "documentary / biography",
    4: "romance / relationship",
    5: "reality TV / competition",
    6: "family / fantasy",
    7: "sci-fi / horror / mystery",
    8: "crime / thriller / mystery",
    9: "history / war / biography"
}


# explanation sentence generator
def make_explanation(seed_title, rec_title, shared_genres, shared_terms, topic_name, shared_cast):
    pieces = []

    if shared_genres:
        pieces.append(f"genres like {', '.join(shared_genres[:2])}")
    if shared_terms:
        pieces.append(f"similar terms such as {', '.join(shared_terms[:4])}")
    if topic_name:
        pieces.append(f"a broader {topic_name.lower()} topic")
    if shared_cast:
        pieces.append(f"actors like {', '.join(shared_cast[:2])}")
    if not pieces:
        return f"{rec_title} is recommended because it is textually similar to {seed_title}."

    return (
        f"{rec_title} is recommended because it shares "
        + "; ".join(pieces)
        + f" with {seed_title}."
    )


def get_shared_genres(seed_movie, rec_movie):
    seed_genres = set(str(seed_movie["genre"]).split(", "))
    rec_genres = set(str(rec_movie["genre"]).split(", "))
    return sorted(seed_genres & rec_genres)


# find overlapping important tf-idf words
def get_shared_terms(seed_idx, rec_idx, top_n=5):
    shared = tfidf[seed_idx].multiply(tfidf[rec_idx])
    if shared.nnz == 0:
        return []

    coo = shared.tocoo()
    ranked = sorted(
        zip(coo.col, coo.data),
        key=lambda x: x[1],
        reverse=True
    )

    terms = []
    for i, score in ranked:
        word = feature_names[i]
        if len(word) < 4:
            continue
        if word in {
            "said","life","story","world","family","young",
            "people","finds","years","time","day"
        }:
            continue
        terms.append(word)

        if len(terms) >= top_n:
            break

    return terms

def get_shared_cast(seed_movie, rec_movie):
    seed_cast = set(str(seed_movie.get("clean_stars", "")).split())
    rec_cast = set(str(rec_movie.get("clean_stars", "")).split())
    return list(seed_cast & rec_cast)


# generate results
def recommend_with_explanations(seed_title, top_k=5):
    matches = df[df["title"].str.contains(seed_title, case=False, na=False)]
    if len(matches) == 0:
        print(f"No movie found for: {seed_title}")
        return

    seed_idx = matches.index[0]
    seed_movie = df.loc[seed_idx]

    similarities = cosine_similarity(tfidf[seed_idx], tfidf).ravel()
    similarities[seed_idx] = -1
    seed_title_clean = str(seed_movie["title"]).strip().lower()

    for i, title in enumerate(df["title"]):
        if str(title).strip().lower() == seed_title_clean:
            similarities[i] = -1

    top_indices = similarities.argsort()[::-1][:top_k]

    print(f"\n=== Recommendations for {seed_movie['title']} ===")

    for rec_idx in top_indices:
        rec_movie = df.loc[rec_idx]

        shared_genres = get_shared_genres(seed_movie, rec_movie)
        shared_terms = get_shared_terms(seed_idx, rec_idx)
        shared_cast = get_shared_cast(seed_movie, rec_movie)

        seed_topic = int(seed_movie["dominant_topic"])
        rec_topic = int(rec_movie["dominant_topic"])

        topic_name = None
        if seed_topic == rec_topic:
            topic_name = topic_names.get(seed_topic, f"topic {seed_topic}")

        explanation = make_explanation(
            seed_movie["title"],
            rec_movie["title"],
            shared_genres,
            shared_terms,
            topic_name,
            shared_cast
        )

        print(f"\n{rec_movie['title']} | similarity={similarities[rec_idx]:.4f}")
        print(f"Genres: {rec_movie['genre']}")
        print(f"Shared terms: {shared_terms}")
        print(f"Seed topic: {seed_topic}, Recommendation topic: {rec_topic}")
        print(f"Explanation: {explanation}")


print("=== Top Terms Per Topic ===")

for topic_id, topic_weights in enumerate(H):
    top_indices = topic_weights.argsort()[::-1][:10]
    top_terms = [feature_names[i] for i in top_indices]
    print(f"Topic {topic_id}: {', '.join(top_terms)}")

print("\n=== Example Movies Per Topic ===")

for topic_id in range(10):
    topic_movies = df[df["dominant_topic"] == topic_id].nlargest(5, "topic_strength")
    print(f"\nTopic {topic_id}: {topic_names.get(topic_id, '')}")
    print(topic_movies[["title", "genre"]].to_string(index=False))

print("\n=== Topic Distribution ===")
print(df["dominant_topic"].value_counts().sort_index())


# me testing it recommendations
recommend_with_explanations("Breaking Bad", top_k=5)