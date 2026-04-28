# explanation_generator.py

import argparse
import re

from tfidf_retrieval import get_seed_recommendations
from topicmodeling_fin import train_topic_model, get_pair_topic_info
NOISY_TERMS = {"star", "stars", "actor", "actors", "movie", "film"}

def split_genres(genre_text):
    if not genre_text:
        return set()
    return {
        g.strip().lower()
        for g in str(genre_text).split(",")
        if g.strip()
    }

def split_cast(cast_text):
    if not cast_text:
        return set()
    # cast is currently displayed as comma-separated cleaned tokens
    tokens = [t.strip().lower() for t in str(cast_text).split(",") if t.strip()]
    tokens = [t for t in tokens if t not in NOISY_TERMS]

    return set(tokens)

def extract_cast_names(cast_text):
    if not cast_text:
        return set(), set()
    cast_text = str(cast_text).lower().strip()

    # Remove noise words
    cast_text = re.sub(r"\b(star|stars|actor|actors)\b", " ", cast_text)
    cast_text = re.sub(r"\s+", " ", cast_text).strip()

    # Case 1: comma-separated tokens like "bryan, cranston, aaron, paul"
    if "," in cast_text:
        tokens = [t.strip() for t in cast_text.split(",") if t.strip()]
    else:
        # Case 2: whitespace text like "bryan cranston aaron paul"
        tokens = cast_text.split()

    tokens = [t for t in tokens if t not in NOISY_TERMS]

    cast_names = set()
    # Pair first/last names
    for i in range(0, len(tokens) - 1, 2):
        cast_names.add(f"{tokens[i]} {tokens[i+1]}")

    cast_tokens = set(tokens)
    return cast_names, cast_tokens

def clean_matched_terms(matched_terms, max_terms=5):
    terms = []

    for field_name, items in matched_terms.items():
        for item in items:
            term = item["term"].lower().strip()
            term_tokens = set(term.split())

            if term_tokens & NOISY_TERMS:
                continue

            if len(term) < 3:
                continue

            terms.append(term)

    # remove duplicates while preserving order
    seen = set()
    cleaned = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            cleaned.append(term)

    return cleaned[:max_terms]


def format_list(items, max_items=3):
    items = list(items)[:max_items]

    if not items:
        return ""

    if len(items) == 1:
        return items[0]

    if len(items) == 2:
        return f"{items[0]} and {items[1]}"

    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_explanation(
    seed_title,
    rec_title,
    shared_genres=None,
    shared_cast=None,
    shared_terms=None,
    topic_info=None
):
    shared_genres = shared_genres or []
    shared_cast = shared_cast or []
    shared_terms = shared_terms or []
    topic_info = topic_info or {}

    parts = []

    if shared_genres:
        parts.append(f"shares genre overlap in {format_list(shared_genres)}")

    if shared_cast:
        parts.append(f"features shared cast members such as {format_list(shared_cast)}")
        
    if shared_terms:
        parts.append(f"contains similar descriptive terms such as {format_list(shared_terms)}")

    if topic_info:
        topic_similarity = topic_info.get("topic_similarity")
        same_topic = topic_info.get("same_topic")
        shared_topic_keywords = topic_info.get("shared_topic_keywords", [])
        rec_topic = topic_info.get("rec_topic", {})

        if same_topic and shared_topic_keywords:
            topic_words = format_list(shared_topic_keywords, max_items=4)
            parts.append(
                f"aligns with a broader topic involving {topic_words}"
            )
        elif topic_similarity is not None and topic_similarity >= 0.50:
            topic_name = rec_topic.get("topic_name", "a related theme")
            parts.append(
                f"has related topic-level similarity to {topic_name}"
            )

    if not parts:
        return (
            f"{rec_title} is recommended because it has overall text similarity "
            f"to {seed_title}."
        )

    return (
        f"{rec_title} is recommended because it "
        + "; ".join(parts)
        + f" with {seed_title}."
    )


def enrich_recommendations_with_explanations(
    seed_title,
    data_file="cleaned_imdb_dataset.csv",
    text_col="document_text_v2",
    n_topics=10,
    top_k=5
):
    retrieval_output = get_seed_recommendations(
        seed_title=seed_title,
        data_file=data_file,
        top_k=top_k
    )

    topic_model = train_topic_model(
        data_file=data_file,
        text_col=text_col,
        n_topics=n_topics
    )

    seed_idx = retrieval_output["seed_idx"]
    seed_movie = retrieval_output["seed_movie"]

    seed_genres = split_genres(seed_movie.get("genre", ""))
    seed_cast_text = (seed_movie.get("cast", "")
    or seed_movie.get("clean_stars", "")
    or seed_movie.get("stars", "")
    )
    seed_cast_names, seed_cast_tokens = extract_cast_names(seed_cast_text)
    final_results = []

    for rec in retrieval_output["results"]:
        rec_idx = rec["idx"]

        rec_genres = split_genres(rec.get("genre", ""))
        rec_cast_names, rec_cast_tokens = extract_cast_names(rec.get("cast", ""))

        shared_genres = sorted(seed_genres & rec_genres)
        shared_cast = sorted(seed_cast_names & rec_cast_names)
        shared_terms = clean_matched_terms(rec.get("matched_terms", {}))
        topic_info = get_pair_topic_info(topic_model, seed_idx, rec_idx)
        
        filtered_terms = []
        for term in shared_terms:
            term_clean = term.lower().strip()
            term_tokens = set(term_clean.split())

            # Move actor full names out of descriptive terms
            if term_clean in shared_cast:
                continue
            # Remove individual actor-name tokens too
            if term_tokens & seed_cast_tokens:
                continue
            filtered_terms.append(term)
        shared_terms = filtered_terms

        explanation = build_explanation(
            seed_title=retrieval_output["seed_title"],
            rec_title=rec["title"],
            shared_genres=shared_genres,
            shared_cast=shared_cast,
            shared_terms=shared_terms,
            topic_info=topic_info
        )

        final_results.append({
            "seed_title": retrieval_output["seed_title"],
            "recommended_title": rec["title"],
            "year": rec.get("year", ""),
            "genre": rec.get("genre", ""),
            "cast": rec.get("cast", ""),
            "description": rec.get("description", ""),
            "similarity_score": rec.get("score", 0.0),
            "field_scores": rec.get("field_scores", {}),
            "shared_terms": shared_terms,
            "shared_genres": shared_genres,
            "shared_cast": shared_cast,
            "topic_info": topic_info,
            "explanation": explanation,
        })

    return {
        "seed_title": retrieval_output["seed_title"],
        "seed_idx": seed_idx,
        "results": final_results
    }


def print_explained_results(output):
    print(f"\n=== Explained Recommendations for {output['seed_title']} ===")

    for i, rec in enumerate(output["results"], start=1):
        print(f"\n{i}. {rec['recommended_title']} ({rec['year']})")
        print(f"   Score: {rec['similarity_score']:.4f}")
        print(f"   Genre: {rec['genre']}")

        if rec["shared_genres"]:
            print(f"   Shared genres: {', '.join(rec['shared_genres'])}")

        if rec["shared_cast"]:
            print(f"   Shared cast signals: {', '.join(rec['shared_cast'])}")

        if rec["shared_terms"]:
            print(f"   Shared terms: {', '.join(rec['shared_terms'])}")

        topic_info = rec["topic_info"]
        print(
            f"   Topic similarity: {topic_info['topic_similarity']:.3f} "
            f"(same topic: {topic_info['same_topic']})"
        )
        print(f"   Explanation: {rec['explanation']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-title", required=True)
    parser.add_argument("--data", default="cleaned_imdb_dataset.csv")
    parser.add_argument("--text-col", default="document_text_v2")
    parser.add_argument("--topics", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    output = enrich_recommendations_with_explanations(
        seed_title=args.seed_title,
        data_file=args.data,
        text_col=args.text_col,
        n_topics=args.topics,
        top_k=args.top_k
    )

    print_explained_results(output)


if __name__ == "__main__":
    main()