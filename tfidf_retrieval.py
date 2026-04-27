import argparse
import ast
import csv
import re
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# just hardcoded for milestone 1
DATA_FILE = Path("cleaned_imdb_dataset.csv")
TEXT_COLUMN = "document_text_v3"
TOP_K = 5
SEARCH_INDEX_CONFIGS = {
    "query": {
        "label": "plot",
        "weight": 1.0,
        "vectorizer_kwargs": {
            "max_features": 10000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.85,
        },
    },
    "title": {
        "label": "title",
        "weight": 1.0,
        "vectorizer_kwargs": {"max_features": 5000, "ngram_range": (1, 2), "min_df": 1},
    },
    "genre": {
        "label": "genre",
        "weight": 0.85,
        "vectorizer_kwargs": {"max_features": 3000, "ngram_range": (1, 2), "min_df": 1},
    },
    "cast": {
        "label": "cast",
        "weight": 0.9,
        "vectorizer_kwargs": {
            "max_features": 10000,
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 0.95,
        },
    },
}


def load_movies(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"no records found {csv_path}")

    return rows


def clean_content_text(text):
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = [token for token in text.split() if token.isalpha() and token not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def normalize_metadata_text(text):
    text = text.casefold()
    text = re.sub(r"[_]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def format_cast(raw_cast):
    if not raw_cast:
        return ""

    try:
        cast_members = ast.literal_eval(raw_cast)
    except (SyntaxError, ValueError):
        cast_members = None

    if isinstance(cast_members, list):
        cleaned_members = [str(member).strip(" ,") for member in cast_members if str(member).strip(" ,")]
        if cleaned_members:
            return ", ".join(cleaned_members)

    cleaned_text = raw_cast.replace("[", "").replace("]", "").replace("'", "")
    cleaned_text = re.sub(r"\s*,\s*", ", ", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text.strip(" ,")


def build_index(docs, **vectorizer_kwargs):
    vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    tfidf_matrix = vectorizer.fit_transform(docs)
    return vectorizer, tfidf_matrix


def build_search_indexes(movies):
    index_inputs = {
        "query": [movie.get(TEXT_COLUMN, "") or "" for movie in movies],
        "title": [normalize_metadata_text(movie.get("title", "") or "") for movie in movies],
        "genre": [normalize_metadata_text(movie.get("genre", "") or "") for movie in movies],
        "cast": [normalize_metadata_text(format_cast(movie.get("stars", "") or "")) for movie in movies],
    }

    indexes = {}
    for field_name, config in SEARCH_INDEX_CONFIGS.items():
        vectorizer, tfidf_matrix = build_index(index_inputs[field_name], **config["vectorizer_kwargs"])
        indexes[field_name] = {
            "label": config["label"],
            "weight": config["weight"],
            "vectorizer": vectorizer,
            "matrix": tfidf_matrix,
        }

    return indexes


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
        if result["cast"]:
            print(f"   cast: {result['cast']}")
        print(f"   description: {result['description']}")
        if result["field_scores"]:
            field_scores = ", ".join(
                f"{field_name}={score:.3f}" for field_name, score in sorted(
                    result["field_scores"].items(), key=lambda item: item[1], reverse=True
                )
            )
            print(f"   matched fields: {field_scores}")
        if result["matched_terms"]:
            for field_name, terms in result["matched_terms"].items():
                formatted_terms = ", ".join(f"{term} ({score:.3f})" for term, score in terms)
                print(f"   shared {field_name} terms: {formatted_terms}")
        print()


def collect_results(movies, similarities, field_matches, top_k, skip_idx=None):
    ranked_indices = similarities.argsort()[::-1][:top_k]
    results = []

    for idx in ranked_indices:
        if skip_idx is not None and idx == skip_idx:
            continue
        movie = movies[idx]
        field_scores = {}
        matched_terms = {}

        for field_name, field_match in field_matches.items():
            field_score = field_match["similarities"][idx]
            if field_score <= 0:
                continue

            field_scores[field_name] = field_score
            shared_terms = top_shared_terms(
                field_match["query_vector"],
                field_match["matrix"][idx],
                field_match["vectorizer"].get_feature_names_out(),
            )
            if shared_terms:
                matched_terms[field_name] = shared_terms

        results.append(
            {
                "title": movie.get("title", "Unknown"),
                "year": movie.get("year", "Unknown"),
                "genre": movie.get("genre", ""),
                "cast": format_cast(movie.get("stars", "")),
                "description": movie.get("description", ""),
                "score": similarities[idx],
                "field_scores": field_scores,
                "matched_terms": matched_terms,
            }
        )
        if len(results) == top_k:
            break

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="plot/description search text")
    parser.add_argument("--title", help="search by movie title")
    parser.add_argument("--genre", help="search by genre")
    parser.add_argument("--cast", help="search by actor or cast member")
    parser.add_argument("--seed-title", help="find recommendations similar to a seed title")

    args = parser.parse_args()

    search_args = [args.query, args.title, args.genre, args.cast]
    if args.seed_title and any(search_args):
        parser.error("use either --seed-title or search arguments, not both")
    if not args.seed_title and not any(search_args):
        parser.error("provide --seed-title or at least one of --query, --title, --genre, --cast")

    return args


def build_search_requests(args):
    requests = {}

    if args.query:
        requests["query"] = clean_content_text(args.query)
    if args.title:
        requests["title"] = normalize_metadata_text(args.title)
    if args.genre:
        requests["genre"] = normalize_metadata_text(args.genre)
    if args.cast:
        requests["cast"] = normalize_metadata_text(args.cast)

    cleaned_requests = {field_name: text for field_name, text in requests.items() if text}
    if not cleaned_requests:
        raise ValueError("all search inputs are empty after preprocessing")

    return cleaned_requests


def run_field_search(movies, indexes, search_requests, top_k):
    combined_scores = None
    total_weight = 0.0
    field_matches = {}

    for field_name, cleaned_text in search_requests.items():
        index = indexes[field_name]
        query_vector = index["vectorizer"].transform([cleaned_text])
        similarities = cosine_similarity(query_vector, index["matrix"]).ravel()

        weighted_similarities = similarities * index["weight"]
        combined_scores = weighted_similarities if combined_scores is None else combined_scores + weighted_similarities
        total_weight += index["weight"]
        field_matches[field_name] = {
            "vectorizer": index["vectorizer"],
            "matrix": index["matrix"],
            "query_vector": query_vector,
            "similarities": similarities,
        }

    combined_scores = combined_scores / total_weight
    if combined_scores.max() <= 0:
        return []

    return collect_results(movies, combined_scores, field_matches, top_k)


def main():
    args = parse_args()
    movies = load_movies(DATA_FILE)
    search_indexes = build_search_indexes(movies)
    plot_index = search_indexes["query"]

    print(f"Loaded {len(movies)} movies from {DATA_FILE}")
    print(f"Indexed text column: {TEXT_COLUMN}")

    if args.seed_title is None:
        search_requests = build_search_requests(args)
        results = run_field_search(movies, search_indexes, search_requests, TOP_K)

        for field_name, cleaned_text in search_requests.items():
            print(f"{SEARCH_INDEX_CONFIGS[field_name]['label'].title()} search: {cleaned_text}")

        if not results:
            print("\nNo matches found for the provided search fields.")
            return

        format_results("Top query matches", results)
    else:
        seed_idx = find_movie_index(movies, args.seed_title)
        seed_movie = movies[seed_idx]
        similarities = cosine_similarity(plot_index["matrix"][seed_idx], plot_index["matrix"]).ravel()
        similarities[seed_idx] = -1
        field_matches = {
            "plot": {
                "vectorizer": plot_index["vectorizer"],
                "matrix": plot_index["matrix"],
                "query_vector": plot_index["matrix"][seed_idx],
                "similarities": similarities,
            }
        }
        results = collect_results(
            movies,
            similarities,
            field_matches,
            TOP_K,
            skip_idx=seed_idx,
        )

        print(f"Seed movie: {seed_movie.get('title', 'Unknown')} ({seed_movie.get('year', 'Unknown')})")
        format_results("Top seed-based recommendations", results)


if __name__ == "__main__":
    main()
