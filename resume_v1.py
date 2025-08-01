import streamlit as st
import fitz  # PyMuPDF
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import pandas as pd
import re

# === Helper Functions ===
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_skills_roles(llm, resume_text):
    prompt = f"""
    You are an AI trained to extract key information from resumes.
    Based on the resume below, list the candidate's primary skills and roles.

    Resume:
    {resume_text}

    Format:
    Skills:
    - Python
    - SQL
    - Machine Learning
    Roles:
    - Data Scientist
    - Software Engineer
    """
    return llm.invoke(prompt).strip()

def extract_skills_list(text):
    skills_match = re.search(r"Skills:\n((?:\s*-\s*.+\n?)+)", text)
    if skills_match:
        lines = skills_match.group(1).strip().split("\n")
        return [re.sub(r"^\s*-\s*", "", line.strip()) for line in lines if line.strip()]
    return []

def generate_summary(llm, resume_text):
    prompt = f"""
    Summarize the candidate's professional background in 3-5 sentences based on the following resume:

    Resume:
    {resume_text}
    """
    return llm.invoke(prompt).strip()

def generate_questions(llm, resume_text, question_type, difficulty, num_questions, selected_skills, external_context=""):
    skills_context = f"\nFocus only on the following skills or topics: {', '.join(selected_skills)}." if selected_skills else ""
    full_prompt = f"""
    You are a professional interviewer. Based on the resume below,
    generate {num_questions} {question_type} interview questions of {difficulty} difficulty.{skills_context}

    Do not provide answers. Emphasize the most recent and most extensive experience (longest duration or highest-level role).

    For each question, include the associated skill or role.

    Resume:
    {resume_text}

    Format:
    1. What is...? - [Skill: ...]
    2. How would you...? - [Skill: ...]
    ...
    """
    if external_context:
        full_prompt += f"Additional Context from CSV:{external_context}"
    return llm.invoke(full_prompt).strip()

def generate_followup(llm, question, answer):
    prompt = f"""
    You are a professional interviewer. Given the following question and candidate's answer, generate a relevant follow-up question or request for clarification.

    Question: {question}
    Answer: {answer}

    Follow-up:
    """
    return llm.invoke(prompt).strip()

def evaluate_answer(llm, question, answer):
    prompt = f"""
    Evaluate the candidate's answer to the following interview question on a scale of 1 to 10 and provide brief feedback.

    Question: {question}
    Answer: {answer}

    Format:
    Score: X/10
    Feedback: your comment here.
    """
    return llm.invoke(prompt).strip()

def convert_to_csv(text):
    lines = text.strip().split("\n")
    data = []
    for line in lines:
        if ". " in line:
            q_num, rest = line.split(". ", 1)
            if " - [" in rest:
                question, tag = rest.split(" - [", 1)
                tag = tag.strip("]")
            else:
                question, tag = rest, ""
            data.append({"Question #": q_num, "Question": question.strip(), "Tag": tag.strip()})
    return pd.DataFrame(data)

# === GitHub CSV Integration ===
def read_csv_from_github(url):
    try:
        df = pd.read_csv(url)
        st.sidebar.success("✅ Loaded external data from GitHub")
        return df
    except Exception as e:
        st.sidebar.warning(f"⚠️ Failed to load GitHub CSV: {e}")
        return pd.DataFrame()

# Add GitHub CSV info box
st.sidebar.markdown("### 📁 Optional: Load Context from GitHub")
github_csv_url = st.sidebar.text_input("Paste raw GitHub CSV URL")
external_context = ""
if github_csv_url:
    df_context = read_csv_from_github(github_csv_url)
    if not df_context.empty:
        # Combine all rows and columns to a text block
        external_context = df_context.astype(str).apply(lambda row: ' | '.join(row), axis=1).str.cat(sep='')

st.set_page_config(page_title="AI Resume Interview Generator", layout="centered")
st.title("🧠 Interactive Resume Interview Simulator")

uploaded_file = st.file_uploader("📄 Upload your Resume (PDF)", type="pdf")
question_type = st.selectbox("🎯 Question Type", ["Technical", "Behavioral", "Situational", "Coding", "Mixed"])
difficulty = st.selectbox("🔥 Difficulty Level", ["Easy", "Medium", "Hard"])
num_questions = st.slider("🔢 Number of Questions", min_value=1, max_value=10, value=5)

if uploaded_file:
    with st.spinner("🔍 Reading and analyzing resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        llm = OllamaLLM(model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF")

        # Step 1: Skill and role extraction
        extracted_info = extract_skills_roles(llm, resume_text)
        st.subheader("📌 Extracted Skills and Roles")
        st.text(extracted_info)

        # Step 2: Resume summary
        summary = generate_summary(llm, resume_text)
        st.subheader("🧾 Candidate Summary")
        st.write(summary)

        # Step 3: Select skills/topics
        skills_list = extract_skills_list(extracted_info)
        if skills_list:
            selected_skills = st.multiselect("📌 Select specific skills/topics to focus on", options=skills_list)
        else:
            selected_skills = []
            st.warning("⚠️ No skills detected. All topics will be considered.")

        # Step 4: Generate questions
        questions = generate_questions(llm, resume_text, question_type, difficulty, num_questions, selected_skills, external_context)
        st.subheader("🧪 Interactive Interview Mode")
        st.markdown("Answer each question below to receive follow-up questions and a score.")

        question_blocks = questions.strip().split("\n")
        for i, q_line in enumerate(question_blocks):
            if ". " in q_line:
                q_num, q_text = q_line.split(". ", 1)
                st.markdown(f"**{q_text}**")
                user_answer = st.text_area(f"✍️ Your Answer to Q{q_num}", key=f"answer_{i}")

                if user_answer:
                    follow_up = generate_followup(llm, q_text, user_answer)
                    st.markdown(f"🔁 **Follow-up Question:** {follow_up}")
                    score_feedback = evaluate_answer(llm, q_text, user_answer)
                    st.markdown(f"📈 **Evaluation:**\n{score_feedback}")

        # Step 5: Export as CSV
        df = convert_to_csv(questions)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Questions as CSV", data=csv, file_name="interview_questions.csv", mime="text/csv")

    st.success("✅ Interview simulation complete!")
