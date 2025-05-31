import torch
import pandas as pd
import ast
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

MOVIES_PATH = "/content/tmdb_5000_movies.csv"
movies_df = pd.read_csv(MOVIES_PATH)



def process_movie_data(df = movies_df):
    # Select relevant columns
    processed_df = movies_df.loc[:, ["original_title", "genres", "keywords", "overview", "popularity"]]
    
    # Drop rows with missing values
    processed_df.dropna(inplace=True)
    
    def extract_names(list_str):
        names = []
        for item in ast.literal_eval(list_str):
            names.append(item["name"])
        return names
    
    def concat_items(x):
        return " ".join(x)
    
    # Process genres and keywords columns
    for col in ["genres", "keywords"]:
        processed_df[col] = processed_df[col].apply(extract_names)
        processed_df[col] = processed_df[col].apply(concat_items)
    
    # Combine features into a single text column
    processed_df['text'] = (processed_df["original_title"] + " " + 
                           processed_df["keywords"] + " " + 
                           processed_df["genres"] + " " + 
                           processed_df["overview"])
    
    # Select final columns
    processed_df = processed_df.loc[:, ["original_title", "text", "popularity"]]
    
    return processed_df

processed_movies = process_movie_data(movies_df)

MODEL_PATH = "sentence-transformers/all-mpnet-base-v2"
Model  =  SentenceTransformer(MODEL_PATH)

def create_embeddings(model = Model, df = processed_movies):
    movie_texts = df['text'].tolist()
    movie_embeddings = model.encode(movie_texts, show_progress_bar=True)
    return movie_embeddings

movie_embeddings = create_embeddings()

def save_embeddings(df = processed_movies):
    with open('movie_embeddings.pkl', 'wb') as f:
        pickle.dump({'embeddings': movie_embeddings, 'titles': df['original_title']}, f)

def load_embeddings(pickle_path):
  with open(pickle_path, 'rb') as f:
    data = pickle.load(f)
    embeddings = data['embeddings']
    titles = data['titles']
  return embeddings, titles

def query_embedding(query):
  return Model.encode(query)

scaler = MinMaxScaler()

def semantic_search(query_embedding, top_k=5):
    embeddings, movie_titles = load_embeddings('movie_embeddings.pkl')
    sim_scores = cosine_similarity([query_embedding],embeddings)[0]
    top_indices = np.argsort(sim_scores)[-top_k:][::-1]
    results_df =processed_movies.iloc[top_indices][['original_title']]
    results_df['norm_pop'] = scaler.fit_transform(processed_movies.iloc[top_indices][['popularity']])
    results_df['sim_scores'] = sim_scores[top_indices]
    results_df['norm_sim'] = scaler.fit_transform(results_df[['sim_scores']])
    return results_df.loc[:,["original_title","norm_pop","norm_sim"]]


def hybrid_search(sem_results,top_k=5, popularity_weight=0.1):
  sem_results['hybrid_score'] = (1 - popularity_weight) * sem_results['norm_sim'] + (popularity_weight * sem_results['norm_pop'])
  top_indices = np.argsort(sem_results['hybrid_score'])[-top_k:][::-1]
  return sem_results.iloc[top_indices][['original_title']]


def recommend_movies(query, top_k=5):
    query_em = query_embedding(query)
    sem_res = semantic_search(query_em, top_k*3)
    results = hybrid_search(sem_res, top_k)
    return pd.DataFrame({"Recommended Movies": results["original_title"].tolist()})


iface = gr.Interface(
    fn=recommend_movies,
    inputs=[
        gr.Textbox(label="Describe a movie you'd like to see"),
        gr.Slider(1, 10, value=5,step=1, label="Number of recommendations")
    ],
    outputs=gr.Dataframe(
        headers=["Recommended Movies"],
        datatype=["object"]
    ),
    title="🎬 Movie Recommender System",
    description="Find similar movies based semantics and popularity"
)

iface.launch()