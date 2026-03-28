import argparse
import csv
import re
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# just hardcoded for milestone 1
DATA_FILE = Path("cleaned_imdb_dataset.csv")
TEXT_COLUMN = "clean_description"
TOP_K = 5


def load_movies(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"no records found {csv_path}")

    return rows


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = [token for token in text.split() if token.isalpha() and token not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def build_index(movies, text_column):
    docs = [movie.get(text_column, "") or "" for movie in movies]
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.85)
    tfidf_matrix = vectorizer.fit_transform(docs)
    return vectorizer, tfidf_matrix


def top_shared_terms(vector_a, vector_b, feature_names, top_n=5):
    shared = vector_a.multiply(vector_b)
    if shared.nnz == 0:
        return []

    coo = shared.tocoo()
    ranked = sorted(zip(coo.col, coo.data), key=lambda item: item[1], reverse=True)[:top_n]
    return [(feature_names[idx], score) for idx, score in ranked]


def find_movie_index(movies, seed_title):
    seed_title_lower = seed_title.strip().lower()

    exact_matches = [
        idx for idx, movie in enumerate(movies) if (movie.get("title", "").strip().lower() == seed_title_lower)
    ]
    if exact_matches:
        return exact_matches[0]

    partial_matches = [
        idx for idx, movie in enumerate(movies) if seed_title_lower in movie.get("title", "").strip().lower()
    ]
    if partial_matches:
        return partial_matches[0]

    raise ValueError(f"couldn't find a movie title matching '{seed_title}'.")


def format_results(header, results):
    print(f"\n{header}")
    print("=" * len(header))
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']} ({result['year']}) | score={result['score']:.4f}")
        print(f"   genre: {result['genre']}")
        print(f"   description: {result['description']}")
        if result["shared_terms"]:
            shared_terms = ", ".join(f"{term} ({score:.3f})" for term, score in result["shared_terms"])
            print(f"   top shared TF-IDF terms: {shared_terms}")
        print()


def collect_results(movies, similarities, source_vector, tfidf_matrix, vectorizer, top_k, skip_idx=None):
    feature_names = vectorizer.get_feature_names_out()
    ranked_indices = similarities.argsort()[::-1][:top_k]
    results = []

    for idx in ranked_indices:
        if skip_idx is not None and idx == skip_idx:
            continue
        movie = movies[idx]
        results.append(
            {
                "title": movie.get("title", "Unknown"),
                "year": movie.get("year", "Unknown"),
                "genre": movie.get("genre", ""),
                "description": movie.get("description", ""),
                "score": similarities[idx],
                "shared_terms": top_shared_terms(source_vector, tfidf_matrix[idx], feature_names),
            }
        )
        if len(results) == top_k:
            break

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query")
    mode.add_argument("--seed-title")

    return parser.parse_args()


def main():
    args = parse_args()
    movies = load_movies(DATA_FILE)
    vectorizer, tfidf_matrix = build_index(movies, TEXT_COLUMN)

    print(f"Loaded {len(movies)} movies from {DATA_FILE}")
    print(f"Indexed text column: {TEXT_COLUMN}")

    if args.query:
        cleaned_query = clean_text(args.query)
        if not cleaned_query:
            raise ValueError("query is empty after preproccesing")

        query_vector = vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).ravel()
        results = collect_results(movies, similarities, query_vector, tfidf_matrix, vectorizer, TOP_K)

        print(f"Original query: {args.query}")
        print(f"Cleaned query: {cleaned_query}")
        format_results("Top query matches", results)
    else:
        seed_idx = find_movie_index(movies, args.seed_title)
        seed_movie = movies[seed_idx]
        similarities = cosine_similarity(tfidf_matrix[seed_idx], tfidf_matrix).ravel()
        similarities[seed_idx] = -1
        results = collect_results(
            movies,
            similarities,
            tfidf_matrix[seed_idx],
            tfidf_matrix,
            vectorizer,
            TOP_K,
            skip_idx=seed_idx,
        )

        print(f"Seed movie: {seed_movie.get('title', 'Unknown')} ({seed_movie.get('year', 'Unknown')})")
        format_results("Top seed-based recommendations", results)


if __name__ == "__main__":
    main()
