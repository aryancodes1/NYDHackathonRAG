import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import spacy
from groq import Groq
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()
model = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1")

client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
qdrant_client = QdrantClient(
    url="https://bbe512e4-6b6e-475e-bfb5-fe04f5797900.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=os.environ.get('QDRANT_API_KEY'),
)

def rewrite_query(query=""):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant trained to optimize user queries for Retrieval-Augmented Generation (RAG) systems, "
                    "specializing in retrieving precise and relevant verses from the Bhagavad Gita. Your task is to rewrite the given query "
                    "to make it more specific, contextually rich, and precise. Ensure the optimized query includes keywords, characters, concepts, "
                    "or terms specific to the Bhagavad Gita (e.g., Arjuna, Krishna, dharma, karma, moksha) when relevant to the original query. "
                    "Your response should strictly follow this format:\n\n"
                    "<Original Query>\n<Additional Keywords for Chunks>"
                ),
            },
            {
                "role": "user",
                "content": f"Rewrite and optimize the following query for effective retrieval of Bhagavad Gita verses: {query}",
            },
        ],
        model="llama3-8b-8192",
        max_tokens=1000,
    )
    return chat_completion.choices[0].message.content

def process_evaluation(query, collection_name, real):
    query = query.lower()
    query = rewrite_query(query).lower()
    
    query_embedding = model.encode(real)
    query_embedding1 = model.encode(query)
    
    answers = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding1.tolist(),
        limit=15,  
        with_vectors=True 
    )
    
    embeddings = [match.vector for match in answers if match.vector]
    
    similarities = [
        cosine_similarity([query_embedding], [embedding])[0][0] for embedding in embeddings
    ]
    return max(similarities)



data = pd.read_csv('/Users/arunkaul/Desktop/NYDRag/NYDHackathonRAG/dataset/Bhagwad_Gita_Verses_English_Questions.csv')
n = 50
data = data[:n]
d_query = list(data['question'])
d_real = list(data['translation'])

eval = []
for i in range(len(d_query)):
    a = process_evaluation(d_query[i],collection_name='Gita',real = d_real[i])
    print(a)
    eval.append(a)

print(sum(eval)/len(eval))

