import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

import pandas as pd
import time
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# ==========================================
# Load Environment Variables
# ==========================================
load_dotenv()

# Load API key
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY not found!")
    st.stop()

# Gemini client
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-3-flash-preview")

# ==========================================
# Load FAISS
# ==========================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key)
vector_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True )

# ==========================================
# Quiz Loader
# ==========================================

def load_quiz(subject):

    files = {
        "Computer Networks":"mcqs/computer_network_mcq.csv",
        "Database Management System":"mcqs/dbms_mcq.csv",
        "Machine Learning":"mcqs/machine_learning_mcq.csv",
        "Cyber Security":"mcqs/cyber_security_mcq.csv",
        "Operating System":"mcqs/os_mcq.csv",
        "Data Structures":"mcqs/dsa_mcq.csv",
        "Object Oriented Programming":"mcqs/oops_mcq.csv"
    }

    if subject not in files:
        return None

    df = pd.read_csv(files[subject])
    return df.sample(5).reset_index(drop=True)

# ==========================================
# Streamlit UI
# ==========================================

st.set_page_config(
    page_title="Smart Study Assistant",
    page_icon="📚",
    layout="centered")

st.title("📚 Smart Study Assistant")
st.caption("AI-powered Study Assistant using TensorFlow + RAG + Gemini")
st.sidebar.title("📚 Supported Subjects")

st.sidebar.info("""
You can ask questions from:

• Computer Networks (CN)

• Database Management System (DBMS)

• Machine Learning (ML)

• Operating System (OS)

• Object Oriented Programming (OOP)

• Data Structures (DSA)

• Cyber Security
""")

st.sidebar.divider()
st.sidebar.success("💡 Example Questions")

st.sidebar.write("""
• Explain TCP 3-way Handshake
• What is Normalization?
• Explain CNN
• What is Deadlock?
• Difference between Class and Object
• Explain BFS Algorithm
• What is SQL Injection?
""")

prompt = st.chat_input("Ask me anything...")

# ==========================================
# User Question
# ==========================================

if prompt:

    # Show user message
    st.chat_message("user").write(prompt)

    # -----------------------------
    # Retrieve Context
    # -----------------------------

    docs = vector_db.similarity_search(prompt,k=5)
    subjects = []
    for doc in docs:
        if "subject" in doc.metadata:
            subjects.append(doc.metadata["subject"])
            
    from collections import Counter
    if subjects:
        predicted_subject = Counter(subjects).most_common(1)[0][0]
    else:
        predicted_subject = "Unknown"

    st.session_state.subject = predicted_subject
    st.success(f"Predicted Subject: {predicted_subject}")
    # Debug: Show retrieved chunks
    #st.subheader("📄 Retrieved Context (Debug)")
    #for i, doc in enumerate(docs):
        #st.write(f"### Chunk {i+1}")
        #st.write(doc.page_content)
        #st.divider()

    context = "\n\n".join([doc.page_content for doc in docs])

    st.session_state.context = context
    st.session_state.prompt = prompt

    # -----------------------------
    # Gemini Prompt
    # -----------------------------

    gemini_prompt = f"""
You are a Smart Study Assistant.

Answer naturally.

Use the retrieved notes as the PRIMARY source.

If the notes are incomplete,
use your own knowledge.

Never say:

'According to the notes'

or

'According to the uploaded PDF'

Study Notes:

{context}

Question:

{prompt}
"""

    # -----------------------------
    # Gemini Response
    # -----------------------------

    response = None
    for attempt in range(3):

        try:
            response = model.generate_content(gemini_prompt)

            break
        except Exception as e:
            if "503" in str(e):
                if attempt < 2:
                    time.sleep(5)
                else:
                    st.warning("⚠ Gemini servers are busy.\n\nPlease try again after a few seconds.")
            else:
                st.error(e)
                break

    if response:
        st.chat_message("assistant").write(response.text)
# ==========================================
# Flashcards
# ==========================================

st.divider()

col1, col2 = st.columns(2)
with col1:
    flashcard = st.button("📚 Flashcards")
with col2:
    quiz_clicked = st.button("📝 Quiz")

if flashcard:
    context = st.session_state.get("context", "")
    prompt = st.session_state.get("prompt", "")

    if context == "":
        st.warning("Please ask a question first.")
    else:
        try:
            flash_response = model.generate_content(f"""
You are an AI Study Assistant.
The user asked:
{prompt}
Study Notes:
{context}
Generate exactly 8 flashcards ONLY about the user's question.
Rules:
- Focus only on the topic asked by the user.
- Ignore unrelated topics from the study notes.
- If the study notes do not contain enough information, use your own knowledge.
- Do not generate flashcards about unrelated concepts.

Format:

Q:
A:
"""
)
            st.subheader("📚 Flashcards")
            st.write(flash_response.text)
        except Exception as e:
            if "429" in str(e):
                st.warning("⚠️ Gemini API quota exceeded. Please try again later or use a new API project.")
            else:
                st.error(e)

# ==========================================
# Quiz
# ==========================================
st.divider()
if quiz_clicked:
    if "quiz_df" in st.session_state:
        del st.session_state.quiz_df

    subject = st.session_state.get("subject", "")
    if subject == "":
        st.warning("Please ask a question first.")
    else:
        quiz_df = load_quiz(subject)
        if quiz_df is None:
            st.error("Quiz file not found.")
        else:
            st.session_state.quiz_df = quiz_df

# ==========================================
# Display Quiz
# ==========================================

if "quiz_df" in st.session_state:
    quiz_df = st.session_state.quiz_df
    subject = st.session_state.get("subject", "")
    st.subheader(f"📝 {subject} Quiz")
    score = 0

    for i, row in quiz_df.iterrows():
        st.write(f"### Q{i+1}. {row['question']}")

        options = {
            "A": row["option_a"],
            "B": row["option_b"],
            "C": row["option_c"],
            "D": row["option_d"]
        }

        answer = st.radio(
            label=f"Choose your answer for Question {i+1}",
            options=list(options.keys()),
            format_func=lambda x: f"{x}) {options[x]}",
            index=None,
            key=f"q{i}"
        )

        if answer == row["answer"]:
            score += 1

    st.divider()

    if st.button("✅ Submit Quiz"):
        st.subheader("📊 Result")
        st.success(f"Score : {score}/5")
        st.info(f"Percentage : {(score/5)*100:.0f}%")
        st.divider()

        for i, row in quiz_df.iterrows():
            user_answer = st.session_state.get(f"q{i}")
            options = {
                "A": row["option_a"],
                "B": row["option_b"],
                "C": row["option_c"],
                "D": row["option_d"]
            }

            st.write(f"### Question {i+1}")
            st.write(row["question"])

            if user_answer == row["answer"]:
                st.success("✅ Correct")
            else:
                st.error("❌ Incorrect")

            st.write(f"**Your Answer:** {user_answer if user_answer else 'Not Answered'}")

            st.write(f"**Correct Answer:** {row['answer']}) {options[row['answer']]}")
            st.divider()

        if score == 5:
            st.balloons()

        # Clear quiz so a new one is generated next time
        del st.session_state.quiz_df

st.divider()
st.caption("Developed by Pravart Singh | IBM Project | TensorFlow + FAISS + Gemini")