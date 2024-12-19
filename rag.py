from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from groq import Groq
import pickle
import os
import json
from dotenv import load_dotenv


os.environ["TOKENIZERS_PARALLELISM"] = "false"


model = SentenceTransformer("all-MiniLM-L6-v2")

client = Groq(api_key="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")
pc = Pinecone(
    api_key="pcsk_vDpvn_Saet8ExRKrRUYcdhuYrKFXD2oxPWGhLgoE1onf6jWJMY2DXuzRqDHdaSAPxKojh"
)


with open("enhanced_sentences.pkl", "rb") as f:
    enhanced_sentences = pickle.load(f)

with open("enhanced_sentences_yoga.pkl", "rb") as f:
    enhanced_sentences_yoga = pickle.load(f)


def chain_of_thought(question=""):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a chain of thought creator"},
            {
                "role": "user",
                "content": f"{question} - create a chain of thought query to answer this specific question in the output only show the chain of thought query and nothing else at all. strictly follow the format",
            },
        ],
        model="llama3-8b-8192",
        max_tokens=100,
    )
    return chat_completion.choices[0].message.content


def get_bot_response(context="", question=""):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a Question answer Machine"},
            {
                "role": "user",
                "content": f"Using the provided context: {context}, answer the question: {question}. Ensure your response is entirely based on the context without adding any information not explicitly stated. Consider the reasoning outlined in {chain_of_thought(question)} before formulating your response. Provide a clear, answer with a good explanation make it easy to understand for the reader",
            },
        ],
        model="llama3-8b-8192",
        max_tokens=1000,
    )
    return chat_completion.choices[0].message.content


def check_valid(context=""):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a text classifier who only answers in 0 and 1"},
            {
                "role": "user",
                "content": f"{context} - Classify the sentence as 0 or 1: Output 0 if the sentence contains foul language or offensive words. Output 1 for all other sentences, including those that do not contain foul language or offensive content, regardless of whether they are related to any specific subject matter. Output only 0 or 1, nothing else. stricly follow the format only give 1 or 0",
            },
        ],
        model="llama3-8b-8192",
        max_tokens=3,
    )
    return chat_completion.choices[0].message.content


def check_type(context=""):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a text Classifier"},
            {
                "role": "user",
                "content": f"{context} - Classify this sentence as 0 or 1: If the sentence is related to the Patanjali Yoga Sutras—a foundational text of classical yoga philosophy focusing on meditation, self-discipline, and spiritual practice—or broadly discusses yoga principles, output 0. If the sentence is related to the Mahabharata—an ancient Indian epic—or the Bhagavad Gita, which is a sacred discourse within the Mahabharata that focuses on duty, dharma, and philosophy, output 1. Respond with only 0 or 1, nothing else.",
            },
        ],
        model="llama3-8b-8192",
        max_tokens=3,
    )
    return chat_completion.choices[0].message.content


def process_query(query):
    query_embedding = model.encode(query)

    index = pc.Index("my-valid-index")

    t = check_type(query)

    # print(t) t = 0 refers to yoga , t = 1 refers to gita
    if int(t) == 1:
        answers = index.query(
            namespace="gita",
            vector=query_embedding.tolist(),
            top_k=5,
            include_values=False,
        )
    else:
        answers = index.query(
            namespace="yoga",
            vector=query_embedding.tolist(),
            top_k=5,
            include_values=False,
        )

    idx = []
    for i in answers["matches"]:
        idx.append(i["id"])

    context = ""
    if int(t) == 1:
        for i in idx:
            context = context + " " + enhanced_sentences[int(i)]
    else:
        for i in idx:
            context = context + " " + enhanced_sentences_yoga[int(i)]

    response = get_bot_response(context=context, question=query)
    # print(context)
    output_json = {"query": query, "response": response}

    return json.dumps(output_json, indent=4)


query = input("Enter Your Query: ")

# checking if query is valid
check = check_valid(query.lower().strip())

for i in check:
    if int(i) == 1:
        print(process_query(query))
    else:
        print("Inappropriate Query Try Again")
