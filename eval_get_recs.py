import pandas as pd
from tfidf_retrieval import get_query_results, get_seed_recommendations

df = pd.read_csv("Results\\eval_queries_and_seed_movies.csv")

queries = [query.strip() for query in df['query']]
seeds = [seed.strip() for seed in df['seed_title']]

output = []

for query in queries:
    query = query.strip()

    results = get_query_results(query=query, top_k=5)

    for i, result in enumerate(results["results"]):
        output.append({
            "input_type": "query",
            "input_text": query,
            "rank": i + 1,
            "title": result["title"],
            "year": result["year"],
            "genre": result["genre"],
            "score": result["score"],
            "description": result["description"],
            "relevant": ""
        })

for seed in seeds:
    seed = seed.strip()

    results = get_seed_recommendations(seed_title=seed, top_k=5)

    for i, result in enumerate(results["results"]):
        output.append({
            "input_type": "seed",
            "input_text": seed,
            "rank": i + 1,
            "title": result["title"],
            "year": result["year"],
            "genre": result["genre"],
            "score": result["score"],
            "description": result["description"],
            "relevant": ""
        })

pd.DataFrame(output).to_csv("Results\\eval_recommendations.csv", index=False)