# Movie Recommender System
Movie recommendation system on a large IMDB dataset, using movie plot summaries as the primary matching and relevance criteria

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

Example query search:
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

```
python topicmodeling_fin.py --text-col document_text_v2 --topics 10 --show-examples 
```

# Explanation Generation for Recommendation

```
python explanation_generator.py --seed-title "Breaking Bad" --text-col document_text_v2 --topics 10 --top-k 5
```
