import argparse
from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CUSTOM_STOPWORDS = {
    "plot", "add", "short", "wraps", "untitled", "summary",
    "young", "man", "woman", "girl", "boy", "life", "story",
    "tells", "follows", "finds", "look", "new", "past", "years",
    "time", "day", "people", "lives", "series", "film", "movie",
    "based"
}
DEFAULT_STOPWORDS = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS))


@dataclass
class TopicModel:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    nmf: NMF
    tfidf_matrix: object
    topic_matrix: object
    feature_names: object
    topic_keywords: dict
    topic_names: dict

def load_movies(data_file):
    df = pd.read_csv(data_file)
    df = df.reset_index(drop=True)
    return df

def train_topic_model(
    data_file="cleaned_imdb_dataset.csv",
    text_col="document_text_v2",           
    n_topics=10,
    max_features=5000,
    min_df=5,
    max_df=0.60,
    random_state=42
):
    df = load_movies(data_file)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in dataset.")

    docs = df[text_col].fillna("").astype(str)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=DEFAULT_STOPWORDS
    )

    tfidf_matrix = vectorizer.fit_transform(docs)
    nmf = NMF(
        n_components=n_topics,
        random_state=random_state,
        init="nndsvda",
        max_iter=500
    )

    topic_matrix = nmf.fit_transform(tfidf_matrix)
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = extract_topic_keywords(
        nmf_model=nmf,
        feature_names=feature_names,
        top_n=10
    )

    df["dominant_topic"] = topic_matrix.argmax(axis=1)
    df["topic_strength"] = topic_matrix.max(axis=1)
    topic_names = auto_topic_names(topic_keywords)

    return TopicModel(
        df=df,
        vectorizer=vectorizer,
        nmf=nmf,
        tfidf_matrix=tfidf_matrix,
        topic_matrix=topic_matrix,
        feature_names=feature_names,
        topic_keywords=topic_keywords,
        topic_names=topic_names
    )


def extract_topic_keywords(nmf_model, feature_names, top_n=10):
    topic_keywords = {}
    for topic_id, weights in enumerate(nmf_model.components_):
        top_indices = weights.argsort()[::-1][:top_n]
        topic_keywords[topic_id] = [feature_names[i] for i in top_indices]

    return topic_keywords


def auto_topic_names(topic_keywords, words_per_name=3):
    topic_names = {}

    for topic_id, words in topic_keywords.items():
        topic_names[topic_id] = " / ".join(words[:words_per_name])

    return topic_names


def get_movie_topic_info(topic_model, movie_idx):
    df = topic_model.df

    topic_id = int(df.loc[movie_idx, "dominant_topic"])
    topic_strength = float(df.loc[movie_idx, "topic_strength"])

    return {
        "topic_id": topic_id,
        "topic_name": topic_model.topic_names.get(topic_id, f"topic {topic_id}"),
        "topic_keywords": topic_model.topic_keywords.get(topic_id, []),
        "topic_strength": topic_strength,
    }


def get_topic_similarity(topic_model, seed_idx, rec_idx):
    seed_vec = topic_model.topic_matrix[seed_idx].reshape(1, -1)
    rec_vec = topic_model.topic_matrix[rec_idx].reshape(1, -1)

    return float(cosine_similarity(seed_vec, rec_vec)[0][0])


def get_pair_topic_info(topic_model, seed_idx, rec_idx):
    seed_topic = get_movie_topic_info(topic_model, seed_idx)
    rec_topic = get_movie_topic_info(topic_model, rec_idx)

    similarity = get_topic_similarity(topic_model, seed_idx, rec_idx)
    same_topic = seed_topic["topic_id"] == rec_topic["topic_id"]

    if same_topic:
        shared_topic_keywords = seed_topic["topic_keywords"][:5]
    else:
        shared_topic_keywords = []

    return {
        "seed_topic": seed_topic,
        "rec_topic": rec_topic,
        "same_topic": same_topic,
        "topic_similarity": similarity,
        "shared_topic_keywords": shared_topic_keywords,
    }


def print_topics(topic_model):
    print("\n=== Top Terms Per Topic ===")
    for topic_id, words in topic_model.topic_keywords.items():
        topic_name = topic_model.topic_names.get(topic_id, f"topic {topic_id}")
        print(f"Topic {topic_id} ({topic_name}): {', '.join(words)}")


def print_topic_examples(topic_model, examples_per_topic=5):
    df = topic_model.df

    print("\n=== Example Movies Per Topic ===")
    for topic_id in sorted(df["dominant_topic"].unique()):
        topic_movies = df[df["dominant_topic"] == topic_id].nlargest(
            examples_per_topic,
            "topic_strength"
        )

        topic_name = topic_model.topic_names.get(topic_id, f"topic {topic_id}")
        print(f"\nTopic {topic_id}: {topic_name}")
        print(topic_movies[["title", "genre"]].to_string(index=False))


def print_topic_distribution(topic_model):
    print("\n=== Topic Distribution ===")
    print(topic_model.df["dominant_topic"].value_counts().sort_index())


def save_topic_assignments(topic_model, output_file="movies_with_topics.csv"):
    topic_model.df.to_csv(output_file, index=False)
    print(f"\nSaved topic assignments to: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="cleaned_imdb_dataset.csv")
    parser.add_argument("--text-col", default="document_text_v2")
    parser.add_argument("--topics", type=int, default=10)
    parser.add_argument("--output", default="movies_with_topics.csv")
    parser.add_argument("--show-examples", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    topic_model = train_topic_model(
        data_file=args.data,
        text_col=args.text_col,
        n_topics=args.topics
    )

    print_topics(topic_model)
    print_topic_distribution(topic_model)
    if args.show_examples:
        print_topic_examples(topic_model)

    save_topic_assignments(topic_model, args.output)


if __name__ == "__main__":
    main()