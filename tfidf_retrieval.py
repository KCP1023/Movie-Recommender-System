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

CAST_NOISE = {"star", "stars", "actor", "actors"}
def format_clean_cast(movie):
    clean_cast = movie.get("clean_stars", "") or ""

    if clean_cast:
        seen = set()
        cleaned_tokens = []
        for token in clean_cast.split():
            token = token.lower().strip()
            if token in CAST_NOISE:
                continue
            if token not in seen:
                seen.add(token)
                cleaned_tokens.append(token)

        return ", ".join(cleaned_tokens)

    return format_cast(movie.get("stars", "") or "")

def format_cast(raw_cast):
    if not raw_cast:
        return ""

    try:
        cast_members = ast.literal_eval(raw_cast)
    except (SyntaxError, ValueError):
        cast_members = None

    if isinstance(cast_members, list):
        cleaned_members = []
        for member in cast_members:
            member = str(member)
            member = member.replace("|", " ")
            member = re.sub(r"\bStars?:\s*", " ", member, flags=re.IGNORECASE)
            member = re.sub(r"\s+", " ", member).strip(" ,")
            if member:
                cleaned_members.append(member)
        return ", ".join(cleaned_members)

    cleaned_text = str(raw_cast)
    cleaned_text = cleaned_text.replace("[", "").replace("]", "").replace("'", "")
    cleaned_text = cleaned_text.replace("|", ",")
    cleaned_text = re.sub(r"\bStars?:\s*", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\s*,\s*", ", ", cleaned_text)
    cleaned_text = re.sub(r"(,\s*)+", ", ", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)

    return cleaned_text.strip(" ,")

def save_results_csv(seed_title, results, output_path):
    rows = []

    for rank, rec in enumerate(results, start=1):
        shared_terms = []

        for field_name, items in rec.get("matched_terms", {}).items():
            for item in items:
                shared_terms.append(item["term"])

        # remove duplicates, preserve order
        seen = set()
        shared_terms_clean = []
        for term in shared_terms:
            if term not in seen:
                seen.add(term)
                shared_terms_clean.append(term)

        rows.append({
            "seed_title": seed_title,
            "rank": rank,
            "recommended_title": rec.get("title", ""),
            "year": rec.get("year", ""),
            "genre": rec.get("genre", ""),
            "score": rec.get("score", 0.0),
            "shared_terms": ", ".join(shared_terms_clean),
            "cast": rec.get("cast", ""),
            "description": rec.get("description", "")
        })

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved retrieval results to: {output_path}")

def build_index(docs, **vectorizer_kwargs):
    vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    tfidf_matrix = vectorizer.fit_transform(docs)
    return vectorizer, tfidf_matrix


def build_search_indexes(movies):
    index_inputs = {
        "query": [movie.get(TEXT_COLUMN, "") or "" for movie in movies],
        "title": [normalize_metadata_text(movie.get("title", "") or "") for movie in movies],
        "genre": [normalize_metadata_text(movie.get("clean_genre", "") or "") for movie in movies],
        "cast": [normalize_metadata_text(format_clean_cast(movie)) for movie in movies],
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
    
    NOISY_TERMS = {"star", "stars", "cast", "actor", "actors"}

    terms = []
    for idx, score in ranked:
        term = feature_names[idx]
        term_tokens = set(term.split())

        if term_tokens & NOISY_TERMS:
            continue
        terms.append((term, score))
        if len(terms) == top_n:
            break
    return terms


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
                formatted_terms = ", ".join(
                f"{item['term']} ({item['score']:.3f})"
                for item in terms
            )
            print(f"   shared {field_name} terms: {formatted_terms}")

def normalize_title(text):
    return str(text).strip().casefold()

def collect_results(movies, similarities, field_matches, top_k, skip_idx=None, seed_title=None):
    ranked_indices = similarities.argsort()[::-1]
    results = []

    seed_title_clean = normalize_title(seed_title) if seed_title else None

    for idx in ranked_indices:
        movie = movies[idx]
        movie_title = movie.get("title", "")

        if skip_idx is not None and idx == skip_idx:
            continue
        if similarities[idx] <= 0:
            continue

        if seed_title_clean and normalize_title(movie_title) == seed_title_clean:
          continue

        field_scores = {}
        matched_terms = {}

        for field_name, field_match in field_matches.items():
            field_score = field_match["similarities"][idx]
            if field_score <= 0:
                continue

            field_scores[field_name] = float(field_score)
            shared_terms = top_shared_terms(
                field_match["query_vector"],
                field_match["matrix"][idx],
                field_match["vectorizer"].get_feature_names_out(),
            )

            if shared_terms:
                matched_terms[field_name] = [
                    {"term": term, "score": float(score)}
                    for term, score in shared_terms
                ]

        results.append(
            {
                "idx": int(idx),
                "title": movie.get("title", "Unknown"),
                "year": movie.get("year", "Unknown"),
                "genre": movie.get("genre", ""),
                "cast": format_clean_cast(movie),
                "description": movie.get("description", ""),
                "score": float(similarities[idx]),
                "field_scores": field_scores,
                "matched_terms": matched_terms,
            }
        )

        if len(results) == top_k:
            break

    return results

def build_retrieval_engine(data_file=DATA_FILE):
    movies = load_movies(Path(data_file))
    search_indexes = build_search_indexes(movies)
    return movies, search_indexes

def get_seed_recommendations(seed_title, data_file=DATA_FILE, top_k=TOP_K):
    movies, search_indexes = build_retrieval_engine(data_file)
    plot_index = search_indexes["query"]

    seed_idx = find_movie_index(movies, seed_title)
    seed_movie = movies[seed_idx]

    similarities = cosine_similarity(
        plot_index["matrix"][seed_idx],
        plot_index["matrix"]
    ).ravel()

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
        top_k,
        skip_idx=seed_idx,
        seed_title=seed_movie.get("title", "")
    )   

    return {
        "mode": "seed",
        "seed_idx": seed_idx,
        "seed_title": seed_movie.get("title", "Unknown"),
        "seed_year": seed_movie.get("year", "Unknown"),
        "seed_movie": seed_movie,
        "results": results,
    }
    
def get_query_results(query=None, title=None, genre=None, cast=None, data_file=DATA_FILE, top_k=TOP_K):
    movies, search_indexes = build_retrieval_engine(data_file)

    class Args:
        pass

    args = Args()
    args.query = query
    args.title = title
    args.genre = genre
    args.cast = cast

    search_requests = build_search_requests(args)
    results = run_field_search(movies, search_indexes, search_requests, top_k)

    return {
        "mode": "query",
        "search_requests": search_requests,
        "results": results,
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="plot/description search text")
    parser.add_argument("--title", help="search by movie title")
    parser.add_argument("--genre", help="search by genre")
    parser.add_argument("--cast", help="search by actor or cast member")
    parser.add_argument("--seed-title", help="find recommendations similar to a seed title")
    parser.add_argument("--save-csv", default=None)

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
            seed_title=seed_movie.get("title", ""),
        )

        print(f"Seed movie: {seed_movie.get('title', 'Unknown')} ({seed_movie.get('year', 'Unknown')})")
        format_results("Top seed-based recommendations", results)
        if args.save_csv:
            save_results_csv(
                seed_title=seed_movie.get("title", ""),
                results=results,
                output_path=args.save_csv
            )


if __name__ == "__main__":
    main()
