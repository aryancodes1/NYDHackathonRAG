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
            {"role": "system", "content": "You are a highly knowledgeable and detail-oriented assistant, specializing in providing precise and contextually accurate answers."},
            {
                "role": "user",
                "content": (
                    f"Given the following context:\n\n"
                    f"--- Context Start ---\n{context}\n--- Context End ---\n\n"
                    f"Answer the question: {question}\n\n"
                    f"Instructions:\n"
                    f"- Your response should rely strictly on the provided context.\n"
                    f"- Do not add any information or assumptions not explicitly mentioned in the context.\n"
                    f"- Use clear reasoning and explain your answer in a way that is simple and easy to understand.\n"
                    f"- Incorporate insights from the reasoning outlined in {chain_of_thought(question)} if applicable.\n"
                ),
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
            {"role": "system", "content": "You are a text Classifier who answers only in 0 and 1"},
            {
                "role": "user",
                "content": f"{context} - Patanjali Yoga Sutras Theme: Focuses on yoga philosophy, meditation, self-discipline, and achieving inner peace. Key Concepts: Ashtanga Yoga (Eightfold Path): Yama, Niyama, Asana, Pranayama, Pratyahara, Dharana, Dhyana, Samadhi. Spiritual Practices: Self-discipline (Tapas), surrender (Ishwarapranidhana), and mindfulness. Imagination and Contemplation: Techniques for focusing the mind and calming mental fluctuations (Chitta Vritti Nirodha). Ultimate Goal: Liberation (Kaivalya) through control over mind and senses. Context of Questions: Likely to involve topics about yoga, meditation, self-awareness, mental control, or achieving spiritual growth. Bhagavad Gita (Part of the Mahabharata) Theme: A sacred discourse delivered by Lord Krishna to Arjuna, addressing the nature of duty, righteousness (Dharma), and the moral dilemmas of life. Key Concepts: Karma Yoga: Path of action. Jnana Yoga: Path of knowledge. Bhakti Yoga: Path of devotion. Dharma: Performing one’s duty selflessly. Inner Conflict: Guidance on overcoming doubts and taking righteous actions. Spiritual and Ethical Teachings: Learning about the soul (Atman), the Supreme Being (Brahman), and life’s transient nature. Characters in Bhagavad Gita: Arjuna: The Pandava prince facing moral dilemmas. Krishna: Lord Vishnu incarnate and Arjuna's charioteer, delivering divine teachings. Duryodhana: Leader of the Kauravas, symbolizing greed and unrighteousness. Dhritarashtra: Blind king of Hastinapur, listening to the battle narration. Sanjaya: Dhritarashtra's advisor, granted divine vision to narrate the events. Classification Rules: If the topic involves yoga, meditation, spiritual practices, imagination, or self-discipline, classify as 0 (Yoga Sutras). If the topic involves duty, morality, dharma, inner conflicts, philosophical discourse on life, or any reference to the characters or teachings of the Gita, classify as 1 (Bhagavad Gita). Respond only with 0 or 1.",
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
    print(context)
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
