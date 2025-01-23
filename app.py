import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
import json
import pickle
import os
import pandas as pd
import re 
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1")
client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
qdrant_client = QdrantClient(
    url="https://bbe512e4-6b6e-475e-bfb5-fe04f5797900.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=os.environ.get('QDRANT_API_KEY'),
)

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
                    f"- Incorporate insights from the reasoning outlined in {chain_of_thought(question)} if applicable. but never answer or show it in the output just use it to give a better answer\n"
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
                "content": (
                    "You are a strict text classifier that evaluates sentences. "
                    "Classify the input strictly as 0 or 1 based on the following rules: "
                    "Output 0 if the sentence contains foul language, offensive words, or is unrelated to the Bhagavad Gita or Yoga Sutras. "
                    "A sentence is considered related if it explicitly mentions concepts, teachings, verses, or philosophies found in the Bhagavad Gita or Yoga Sutras. "
                    "Examples of related content include discussions on dharma, karma, yoga, meditation, moksha, or any references to the texts themselves. "
                    "Output 1 that are directly or indirectly related to the Bhagavad Gita or Yoga Sutras. "
                    "Strictly follow the format and respond only with 0 or 1."
                ),
            },
            {
                "role": "user",
                "content": f"{context} - Classify the sentence as 0 or 1:",
            },
        ],
        model="llama3-8b-8192",
        max_tokens=3,
    )
    return chat_completion.choices[0].message.content



def check_valid_answer(q="", a=""):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict text classifier that evaluates answers. "
                    "Respond strictly with 1 or 0. "
                    "Output 1 if the answer is grammatically correct, directly responds to the question,and contains no hallucinations or fabricated information. "
                    "Output 0 if the answer fails to meet these criteria in any way, including incorrect grammar, irrelevance, or hallucinations. "
                    "Your response must only be 1 or 0, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {q} Answer: {a}. Classify the response:",
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
                "content": f"Rewrite and optimize the following query for effective retrieval of Bhagavad Gita verses: {query} in this format - ""<Original Query>\n<Additional Keywords for Chunks> ""  strictly follow it the additional keywords should be realted to the query and should help rag find out the correct chunks",
            },
        ],
        model="llama3-8b-8192",
        max_tokens=300,
    )
    return chat_completion.choices[0].message.content

def rewrite_yoga_sutras_query(query=""):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant trained to optimize user queries for Retrieval-Augmented Generation (RAG) systems, "
                    "specializing in retrieving precise and relevant sutras from the Yoga Sutras of Patanjali. Your task is to rewrite the given query "
                    "to make it more specific, contextually rich, and precise. Ensure the optimized query includes keywords, concepts, practices, or terms "
                    "specific to the Yoga Sutras (e.g., ashtanga, samadhi, yama, niyama, dharana, dhyana, pranayama) when relevant to the original query. "
                    "Your response should strictly follow this format:\n\n"
                    "<Original Query>\n<Additional Keywords for Chunks>"
                ),
            },
            {
                "role": "user",
                "content": f" Rewrite and optimize the following query for effective retrieval of Yoga Sutras Of Patanjali verses: {query} in this format - ""<Original Query>\n<Additional Keywords for Chunks> ""  strictly follow it the additional keywords should be realted to the query and should help rag find out the correct chunks only give the rewritten output nothing else",
            },
        ],
        model="llama3-8b-8192",
        max_tokens=300,
    )
    return chat_completion.choices[0].message.content



def get_sanskrit(data, chapter, verse):
    result = data.loc[(data['chapter'] == chapter) & (data['verse'] == verse), 'translation']
    if not result.empty:
        return result.iloc[0]  
    else:
        return None 

def get_chap_verse(context="", query=""):
    chapter_verse_list = []  
    while True:  
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a Chapter and Verse Extractor. Your task is to return only the chapter and verse number as integers in the format 'chapter : n, verse : n'. Do not include any additional text or explanation.",
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\nQuestion: {query}\nProvide the most relevant chapter and verse numbers related to the question which can help the most in answering the question efficiently and is most related to the question in the exact format 'chapter : n, verse : n'. Only numbers should appear after 'chapter' and 'verse' only return the most relevant one to the question",
                },
            ],
            model="llama3-8b-8192",
            max_tokens=50,
        )
        
        response = chat_completion.choices[0].message.content.strip()
        match = re.findall(r"chapter\s*:\s*(\d+),\s*verse\s*:\s*(\d+)", response, re.IGNORECASE)
        
        if match: 
            for m in match:
                chapter = int(m[0])
                verse = int(m[1])
                chapter_verse_list.append([chapter, verse])
            return chapter_verse_list

def process_query(query, collection_name):
    query = query.lower()
    
    if collection_name.lower() == 'yoga':
        query = rewrite_yoga_sutras_query(query).lower()
    else:
        query = rewrite_query(query).lower()
    
    print(query)
    
    query_embedding = model.encode(query)
    
    if collection_name.lower() == "yoga":
        answers = qdrant_client.search(
            collection_name="Yoga",
            query_vector=query_embedding.tolist(),
            limit=15  
        )
    else:
        answers = qdrant_client.search(
            collection_name="Gita",
            query_vector=query_embedding.tolist(),
            limit=15  
        )
    print("Query Retrieved")
    idx = [match.id for match in answers]
    context = " ".join(
        enhanced_sentences_yoga[int(i)] if collection_name == "yoga" else enhanced_sentences[int(i)] for i in idx
    )
    response = get_bot_response(context, query)
    chapter_verse_list = get_chap_verse(context, query)
    translations = []  
    for c, v in chapter_verse_list:
        if collection_name == "gita":
            translation = data_gita.loc[
                (data_gita['chapter'] == c) & (data_gita['verse'] == v), 'translation'
            ].values[0]
        else:
            translation = data_yoga.loc[
                (data_yoga['chapter'] == c) & (data_yoga['verse'] == v), 'translation'
            ].values[0]
        translations.append({"chapter": c, "verse": v, "translation": translation})
    
    return {"query": query, "response": response, "translations": translations}



st.set_page_config(page_title="Knowledge Assistant", page_icon=":robot:", layout="wide")

st.title("Chat with Knowledge Assistant 🤖")
st.markdown(
    "Ask questions related to **Yoga Sutras** or **Bhagavad Gita**, and get accurate answers!"
)


with st.sidebar:
    st.header("Select Your Query Type")
    query_type = st.radio(
        "Choose the type of question:",
        ("gita", "yoga"),
        help="Select the category your question belongs to.",
    )


st.title("Bhagavad Gita and Patanjali Yoga Sutras Query Assistant")
st.markdown("""
This app allows you to query from the Bhagavad Gita or Patanjali Yoga Sutras. Enter your query below to get the most relevant information.
""")
query = st.text_input("Enter your query:", "")

if query:
    t = 5
    print(check_valid(query))
    if int(check_valid(query)) == 1 or check_valid(query) == "1":
        with st.spinner("Processing your query..."):
            result = None
            while t>0: 
                result = process_query(query, collection_name=query_type)
                t-=1
                print(check_valid_answer(q=query, a=result))
                if int(check_valid_answer(q=query, a=result)) == 1 or check_valid_answer(q=query, a=result) == "1":
                    break  
            st.success("Query processed successfully!")
            st.json(result)

    elif(int(check_valid(query)) == 0):
        st.error("Inappropriate Query. Please try again.")