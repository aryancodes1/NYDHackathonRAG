import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
import json
import pickle
import os
import pandas as pd
import re 

os.environ["TOKENIZERS_PARALLELISM"] = "false"


model = SentenceTransformer("all-MiniLM-L6-v2")

client = Groq(api_key="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")
pc = Pinecone(api_key="pcsk_vDpvn_Saet8ExRKrRUYcdhuYrKFXD2oxPWGhLgoE1onf6jWJMY2DXuzRqDHdaSAPxKojh")

data_gita = pd.read_csv('dataset/Bhagwad_Gita_Verses_English_Questions.csv')
data_yoga = pd.read_csv('dataset/Patanjali_Yoga_Sutras_Verses_English_Questions.csv')

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
            {
                "role": "system",
                "content": "You are a highly knowledgeable and detail-oriented assistant, specializing in providing precise and contextually accurate answers.",
            },
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
                    f"- Incorporate insights from the reasoning outlined in {chain_of_thought(question)} if applicable. but do not show it in the output\n"
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
            {
                "role": "system",
                "content": "You are a text classifier who only answers in 0 and 1",
            },
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
            {
                "role": "system",
                "content": "You are a text Classifier who answers only in 0 and 1",
            },
            {
                "role": "user",
                "content": f"{context} - Patanjali Yoga Sutras Theme: Focuses on yoga philosophy, meditation, self-discipline, and achieving inner peace. Key Concepts: Ashtanga Yoga (Eightfold Path): Yama, Niyama, Asana, Pranayama, Pratyahara, Dharana, Dhyana, Samadhi. Spiritual Practices: Self-discipline (Tapas), surrender (Ishwarapranidhana), and mindfulness. Imagination and Contemplation: Techniques for focusing the mind and calming mental fluctuations (Chitta Vritti Nirodha). Ultimate Goal: Liberation (Kaivalya) through control over mind and senses. Context of Questions: Likely to involve topics about yoga, meditation, self-awareness, mental control, or achieving spiritual growth. Bhagavad Gita (Part of the Mahabharata) Theme: A sacred discourse delivered by Lord Krishna to Arjuna, addressing the nature of duty, righteousness (Dharma), and the moral dilemmas of life. Key Concepts: Karma Yoga: Path of action. Jnana Yoga: Path of knowledge. Bhakti Yoga: Path of devotion. Dharma: Performing one’s duty selflessly. Inner Conflict: Guidance on overcoming doubts and taking righteous actions. Spiritual and Ethical Teachings: Learning about the soul (Atman), the Supreme Being (Brahman), and life’s transient nature. Characters in Bhagavad Gita: Arjuna: The Pandava prince facing moral dilemmas. Krishna: Lord Vishnu incarnate and Arjuna's charioteer, delivering divine teachings. Duryodhana: Leader of the Kauravas, symbolizing greed and unrighteousness. Dhritarashtra: Blind king of Hastinapur, listening to the battle narration. Sanjaya: Dhritarashtra's advisor, granted divine vision to narrate the events , Bhishma ,  Classification Rules: If the topic involves yoga, meditation, spiritual practices, imagination, or self-discipline, classify as 0 (Yoga Sutras). If the topic involves duty, morality, dharma, inner conflicts, philosophical discourse on life, or any reference to the characters or teachings of the Gita, classify as 1 (Bhagavad Gita). Respond only with 0 or 1.",
            },
        ],
        model="llama3-8b-8192",
        max_tokens=3,
    )
    return chat_completion.choices[0].message.content


def get_sanskrit(data, chapter, verse):
    result = data.loc[(data['chapter'] == chapter) & (data['verse'] == verse), 'translation']
    if not result.empty:
        return result.iloc[0]  
    else:
        return None 

def get_chap_verse(context="", query=""):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a Chapter and Verse Extractor. Your task is to return only the chapter and verse numberS as integers in the format 'chapter : n, verse : n'. Do not include any additional text or explanation.",
            },
            {
                "role": "user",
                "content": f"Context: {context}\nQuestion: {query}\nProvide the most relevant chapter and verse number related to the question which can help the most in answering the question efficiently and is most related to the question in the exact format 'chapter : n, verse : n'. Only numbers should appear after 'chapter' and 'verse'.",
            },
        ],
        model="llama3-8b-8192",
        max_tokens=50,
    )
    
    response = chat_completion.choices[0].message.content
    match = re.search(r"chapter\s*:\s*(\d+),\s*verse\s*:\s*(\d+)", response, re.IGNORECASE)
    if match:
        chapter = int(match.group(1))
        verse = int(match.group(2))
        return chapter, verse
    else:
        raise ValueError("The response format was invalid. Ensure the LLM response matches 'chapter : n, verse : n'.")


def process_query(query):
    query_embedding = model.encode(query)
    namespace = "yoga" if check_type(query) == 0 else "gita"

    index = pc.Index("my-valid-index")
    answers = index.query(
        namespace=namespace,
        vector=query_embedding.tolist(),
        top_k=5,
        include_values=False,
    )

    idx = [match["id"] for match in answers["matches"]]
    context = " ".join(
        enhanced_sentences_yoga[int(i)] if namespace == "yoga" else enhanced_sentences[int(i)] for i in idx
    )

    response = get_bot_response(context, query)
    c, v = get_chap_verse(context, query)

    if namespace == "gita":
        translation = data_gita.loc[(data_gita['chapter'] == c) & (data_gita['verse'] == v), 'translation'].values[0]
    else:
        translation = data_yoga.loc[(data_yoga['chapter'] == c) & (data_yoga['verse'] == v), 'translation'].values[0]

    return {"query": query, "response": response, "chapter": c, "verse": v, "translation": translation}



st.set_page_config(page_title="Knowledge Assistant", page_icon=":robot:", layout="wide")

st.title("Chat with Knowledge Assistant 🤖")
st.markdown(
    "Ask questions related to **Yoga Sutras** or **Bhagavad Gita**, and get accurate answers!"
)


with st.sidebar:
    st.header("Select Your Query Type")
    query_type = st.radio(
        "Choose the type of question:",
        ("Yoga Sutras", "Bhagavad Gita"),
        index=0,
        help="Select the category your question belongs to.",
    )



st.title("Bhagavad Gita and Patanjali Yoga Sutras Query Assistant")
st.markdown("""
This app allows you to query from the Bhagavad Gita or Patanjali Yoga Sutras. Enter your query below to get the most relevant information.
""")

query = st.text_input("Enter your query:", "")

if query:
    if check_valid(query):
        with st.spinner("Processing your query..."):
            result = process_query(query)
        st.success("Query processed successfully!")
        st.json(result)
    else:
        st.error("Inappropriate Query. Please try again.")