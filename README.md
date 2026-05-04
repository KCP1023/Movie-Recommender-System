# Movie Recommender System
Movie recommendation system on a large IMDB dataset, using movie plot summaries as the primary matching and relevance criteria along with title, genre, and cast information metadata for better contextual information. The ssytem also utilizes topic modeling to better group movies with higher thematic similarity and generates natural language explanations that highlight reasoning behind the recommendations.

# Dataset
The dataset consists of IMDB movie entries with the following relevant fields:

`title`
`description`
`genre`
`stars`
`year`, `rating`, `votes`, `certificate`, `duration` (metadata)

# Data Cleaning and Preprocessing

This module prepares the IMDB movie dataset for downstream information retrieval and recommendation tasks. It includes data cleaning, preprocessing and construction of multiple document representations to support TF-IDF and semantic retrieval models.

The output of this step is a clean, structured dataset - `cleaned_imdb_dataset.csv` along with ready-to-use document text fields for retrieval. 
We construct multiple versions of document text and add them to our cleaned dataset to evaluate the contribution of different features:
`document_text_v1` : `title` + `description` which is our baseline textual representation
`document_text_v2` : `title` + `description` + `genre` which adds contextual category information
`document_text_v3` : `title` + `description` + `genre` + `actors` which is the most enriched representation (recommended for retrieval)
These fields are intended for direct use in TF-IDF and embedding-based models. We also export them as text files documents (each file contains one document per line) - `documents_v1.txt`, `documents_v2.txt`, `documents_v3.txt`. 

# Exploratory Data Analysis

Exploratory analysis was conducted to validate dataset quality and suitability:
1. Distribution of description lengths
2. Genre frequency distribution
3. Word frequency before preprocessing
4. Rating and year distributions

Key observations:
Plot descriptions are sufficiently detailed for text-based retrieval
Dataset contains diverse genres with mild imbalance
Raw text contains high-frequency noise words, justifying preprocessing

# TF-IDF Retrieval

This module sets up our initial retrieval pipeline to get the top 5 highest ranked recommendations for each seed movie or query. You can run our tf-idf retrieval script in two following ways: one for a natural language search query, or using a seed movie to find similar movie recommendations. We also extract shared descriptive terms, shared genre information, and shared cast signals between the seed movie and each recommendation to obtain actual contextual information that explain the recommendation which is then fed into our explanation generation pipeline.

Example natural language query search:
```
python tfidf_retrieval.py --query "a sponge makes friends underwater"

```

```
python tfidf_retrieval.py --query "a crime drama on sheby family prohibition"
```

Example seed-movie recommendations:
```
python tfidf_retrieval.py --seed-title "Spongebob"

```

```
python tfidf_retrieval.py --seed-title "Peaky Blinders"
```

# Topic Modeling and Building Topics Distribution

Our topic modeling script basically uses the column `document_text_v2` which contains title + description + genre information to group all our movies into a set of 10 topics, and assign each movie a topic similarity score of how similar each movie is to an overarching topic. This is then used to match seed movies to recommendations - if they both have similar dominant topics, and are in the same topic cluster, we get higher thematic similarity and better semantic understanding about our recommendations. We can run our topic modeling script as follows:

`--show-examples` flag gives the output of topic distribution, and the broad topics it has group your dataset under with 5 example movies for each cluster. 

```
python topicmodeling_fin.py --text-col document_text_v2 --topics 10 --show-examples 
```


# Explanation Generation for Recommendation

This module combines all the extracted signals from our tf-idf retrieval pipeline and the dominant topic selection from our topic modeling pipeline to generate explanations for why each movie has been recommended for a particular seed movie. It uses a natural language template and multiple shared signals to generate ranked recommendations with interpretable explanations for a final output that mitigates the "black box aspect" of other recommendation engines and fulfils our objective. We can run our explanation generator script as follows using a `--seed-title` flag for our seed movie. 

```
python explanation_generator.py --seed-title "Breaking Bad"
```



```
python explanation_generator.py --seed-title "Peaky Blinders"
```

# Evaluation

Our evaluation module checks relevance judgements for the recommendations and explanations with 5 seed movies and its recommendations and 5 query movies and their recommendations. 
These include metrics like Precision, MRR, and nDCG across all inputs as well as manual grading of whether the movie is relevant or irrelevant.