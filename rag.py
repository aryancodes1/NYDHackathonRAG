from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from groq import Groq
import pickle
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"


model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key="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")
pc = Pinecone(api_key="pcsk_vDpvn_Saet8ExRKrRUYcdhuYrKFXD2oxPWGhLgoE1onf6jWJMY2DXuzRqDHdaSAPxKojh")


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
                "content": f"Using the provided context: {context}, answer the question: {question}. Ensure your response is entirely based on the context without adding any information not explicitly stated. Consider the reasoning outlined in {chain_of_thought(question)} before formulating your response. Provide a clear, direct, and precise answer as if writing for an exam. Output only the final answer, with only one or two line commentary about the answer",
            },
        ],
        model="llama3-8b-8192",
        max_tokens=1000,
    )
    return chat_completion.choices[0].message.content

def check_valid(context=""):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a text validator"},
            {
                "role": "user",
                "content": f"{context} - Classify this sentence as 0 or 1: Output 0 if the sentence is: Inappropriate: contains offensive language, nonsensical statements, or irrelevant content. Unrelated to the Mahabharata, Bhagavad Gita, or Patanjali Yoga Sutras, including topics such as mind and body wellness, practices. Examples include sentences discussing unrelated subjects like technology, politics, or everyday activities not linked to the mentioned texts or yoga principles. Output 1 if the sentence is: Related to the Mahabharata, an ancient Indian epic exploring themes of dharma, relationships, morality, and the Kurukshetra War. Related to the Bhagavad Gita, a discourse within the Mahabharata discussing spiritual concepts like duty, selfless action, devotion, and philosophical guidance. Related to the Patanjali Yoga Sutras, which focus on meditation, mental discipline, spiritual growth, and mind-body wellness through yoga practices. Examples include sentences about dharma, yoga practices, meditation, the characters of the Mahabharata, or philosophical teachings from the Gita or Yoga Sutras. Respond with only 0 or 1, nothing else.",
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

    print(t)
    if(int(t) == 1):
        answers = index.query(
            namespace="gita", vector=query_embedding.tolist(), top_k=5, include_values=False
        )
    else:
        answers = index.query(
            namespace="yoga", vector=query_embedding.tolist(), top_k=5, include_values=False
        )

    idx = []
    for i in answers["matches"]:
        idx.append(i["id"])
    
    context = ""
    if(int(t) == 1):
        for i in idx:
            context = context + " " + enhanced_sentences[int(i)]
    else:
        for i in idx:
            context = context + " " + enhanced_sentences_yoga[int(i)]

    response = get_bot_response(context=context, question=query)
    print(context)
    output_json = {"query": query, "response": response}
    return json.dumps(output_json, indent=4)


query = input("Enter Your Query: ")

check = check_valid(query)
for i in check:
    if(int(i) == 1):
        print(process_query(query))
    else:
        print("Inappropriate Query Try Again")
