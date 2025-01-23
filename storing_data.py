import pickle
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
import os

load_dotenv()

with open("NYDHackathonRAG/sentence_embeddings_yoga.pkl", "rb") as fp:
    sentence_embedings = pickle.load(fp)
with open("NYDHackathonRAG/enhanced_sentences_yoga.pkl", "rb") as f:
    enhanced_sentences = pickle.load(f)


qdrant_client = QdrantClient(
    url="https://bbe512e4-6b6e-475e-bfb5-fe04f5797900.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=os.environ.get('QDRANT_API_KEY'),
)

collection_name = "Yoga"
vector_size = len(sentence_embedings[0])  
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)  
)

print(enhanced_sentences[5])

print(len(sentence_embedings))

for i in range(len(sentence_embedings)):
    qdrant_client.upsert(
        collection_name="Yoga",
        points=[
            {
                "id": i,
                "vector": sentence_embedings[i],
                "payload": {} 
            }
        ]
    )
print("Data loaded successfully.")
