# Hybrid-Movie-Recommender-System
# Overview
A hybrid movie recommendation engine that combines semantic search with popularity-based ranking, powered by Sentence Transformers and TMDB data. The system suggests films based on content similarity to user preferences while balancing recommendations with trending movies.

# Key Features
Hybrid Recommendation Approach: Blends content-based filtering with popularity ranking

Semantic Search: Uses all-mpnet-base-v2 Sentence Transformer for deep content understanding

TMDB Integration: Leverages comprehensive movie metadata from TMDB dataset

Interactive Interface: Gradio-powered UI for intuitive user experience

# Technical Implementation
  Data Processing Pipeline
  Data Extraction:
  
  Extracts genres, keywords, and overviews from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data 
  
  Creates unified movie representations: title + keywords + genres + overview

  Popularity score integration from TMDB metrics

  Recommendation Engine
  Content-Based Filtering:
  
  all-mpnet-base-v2 model generates 768-dimensional embeddings
  
  Cosine similarity for semantic matching

Popularity Bias:

  Weighted combination of semantic similarity and TMDB popularity
  
  Adjustable bias parameter

# Deployment
Gradio Interface:

User input: Movie title or description

Output: Top k recommendations based on user selection
