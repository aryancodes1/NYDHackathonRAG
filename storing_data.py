import pickle
from pinecone import Pinecone, ServerlessSpec
import pandas as pd

with open("sentence_embeddings_yoga.pkl", "rb") as fp:
    sentence_embedings = pickle.load(fp)
with open("enhanced_sentences_yoga.pkl", "rb") as f:
    enhanced_sentences = pickle.load(f)

print(enhanced_sentences[5])

pc = Pinecone(
    api_key="pcsk_vDpvn_Saet8ExRKrRUYcdhuYrKFXD2oxPWGhLgoE1onf6jWJMY2DXuzRqDHdaSAPxKojh"
)
print(len(sentence_embedings))
index = pc.Index("my-valid-index")

print(index.describe_index_stats())

for i in range(len(sentence_embedings)):
    index.upsert(
        vectors=[{"id": f"{i}", "values": sentence_embedings[i]}], namespace="yoga"
    )
