from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import spacy
from groq import Groq
import pickle
import os

translation = {"धृतराष्ट्र":"Dhritarashtra","सञ्जय":"Sanjay","अर्जुन":"Arjun","भगवान":"God","संजय": "Sanjay"}

client = Groq(api_key="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")
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(
    api_key="pcsk_vDpvn_Saet8ExRKrRUYcdhuYrKFXD2oxPWGhLgoE1onf6jWJMY2DXuzRqDHdaSAPxKojh"
)

"""
pc.create_index("my-valid-index", dimension=384, metric="cosine", spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"))"""


data = pd.read_csv("dataset/Patanjali_Yoga_Sutras_Verses_English_Questions.csv")
verses = data["translation"]

story = ""

for i in range(len(verses)):
    # used for bhagwad gita story = story + " " + "Chapter: " + str(data["chapter"][i]) + ", " + " Verse Number: " + str(data['verse'][i]) + " , "+ translation[data['speaker'][i]] + "," + " Says That " + " , " + verses[i]
    story = story + " " + "Chapter: " + str(data["chapter"][i]) + ", " + " Verse Number: " + str(data['verse'][i])  + " , " + verses[i] # used for yoga sutras

def text_to_sentences(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


sentences = text_to_sentences(story)
enhanced_sentences = sentences

c = 1

sentence_embeddings = model.encode(enhanced_sentences)

print(len(enhanced_sentences))
print(enhanced_sentences[0])
with open("sentence_embeddings_yoga.pkl", "wb") as fp:
    pickle.dump(sentence_embeddings, fp)
with open("enhanced_sentences_yoga.pkl", "wb") as f:
    pickle.dump(enhanced_sentences, f)
