import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import spacy
from groq import Groq
import pickle
import os

pc = Pinecone(
    api_key="pcsk_vDpvn_Saet8ExRKrRUYcdhuYrKFXD2oxPWGhLgoE1onf6jWJMY2DXuzRqDHdaSAPxKojh"
)


client = Groq(api_key="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")

model = SentenceTransformer("all-MiniLM-L6-v2")
from sklearn.metrics.pairwise import cosine_similarity

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
        model="gemma2-9b-it",
        max_tokens=1000,
    )
    return chat_completion.choices[0].message.content

def process_evaluation(query, namespace, real):
    query = query.lower()
    query = rewrite_query(query).lower()
    query_embedding = model.encode(real)
    query_embedding1 = model.encode(query)
    
    index = pc.Index("my-valid-index")
    answers = index.query(
        namespace=namespace,
        vector=query_embedding1.tolist(),
        top_k=15,
        include_values=True,
    )
    embeddings = []
    
    if "matches" in answers:
        for match in answers["matches"]:  
            if 'embedding' in match:  
                embeddings.append(match['embedding'])
            elif 'values' in match:  
                embeddings.append(match['values'])
    n = []
    for i in embeddings:
        sim = cosine_similarity([query_embedding],[i])
        n.append(sim)
    return max(n)[0][0]


data = pd.read_csv('/Users/arunkaul/Desktop/NYDRag/NYDHackathonRAG/dataset/Bhagwad_Gita_Verses_English_Questions.csv')
n = 10
data = data[:n]
d_query = list(data['question'])
d_real = list(data['translation'])

eval = []
for i in range(len(d_query)):
    a = process_evaluation(d_query[i],namespace='gita',real = d_real[i])
    print(a)
    eval.append(a)

print(sum(eval)/len(eval))

