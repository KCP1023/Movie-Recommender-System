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

This module prepares the IMDB movie dataset for downstream information retrieval and recommendation tasks. It includes data cleaning, exploratory data analysis (EDA), and construction of multiple document representations to support TF-IDF and semantic retrieval models.

The output of this step is a clean, structured dataset - `cleaned_imdb_dataset.csv` along with ready-to-use document text fields for retrieval. The following columns are: `clean_title`, `clean_description`, `clean_genre`, `clean_actors`. 

We construct multiple versions of document text and add them to our cleaned dataset to evaluate the contribution of different features:
`document_text_v1` : `title` + `description` which is our baseline textual representation
`document_text_v2` : `title` + `description` + `genre` which adds contextual category information
`document_text_v3` : `title` + `description` + `genre` + `actors` which is the most enriched representation (recommended for retrieval)
These fields are intended for direct use in TF-IDF and embedding-based models. We also export them as text files documents (each file contains one document per line) - `documents_v1.txt`, `documents_v2.txt`, `documents_v3.txt`. 