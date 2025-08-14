# app.py
import os
import re
import time
import base64
import random
import datetime
import io

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from streamlit_tags import st_tags

import psycopg2  # PostgreSQL
import fitz       # PyMuPDF for PDF text extraction
import spacy
from spacy.matcher import PhraseMatcher

# --- OPTIONAL: YouTube video title fetch ---
import pafy
import youtube_dl  # required backend for pafy on some systems

# --- Course & video lists (from your module) ---
# Keep your existing file: Courses.py exporting these lists
from Courses import (
    ds_course, web_course, android_course, ios_course, uiux_course,
    resume_videos, interview_videos
)

# =========================
# SPAcY MODEL (robust load)
# =========================
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(
            "spaCy model 'en_core_web_sm' is not installed.\n"
            "Run this once in your terminal:  \n"
            "    python -m spacy download en_core_web_sm"
        )
        st.stop()

nlp = load_spacy_model()

# =========================
# DATABASE CONNECTION
# =========================
# Make sure database Resume_Analyser exists in pgAdmin.
DB_CONFIG = {
    "host": "localhost",
    "database": "Resume_Analyser",
    "user": "postgres",
    "password": "204646",  # <-- change
    "port": "5432",
}

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def ensure_table():
    create_sql = """
    CREATE TABLE IF NOT EXISTS user_data (
        ID SERIAL PRIMARY KEY,
        Name VARCHAR(100) NOT NULL,
        Email_ID VARCHAR(100),
        resume_score VARCHAR(8) NOT NULL,
        Timestamp VARCHAR(50) NOT NULL,
        Page_no VARCHAR(5) NOT NULL,
        Predicted_Field VARCHAR(50),
        User_level VARCHAR(30),
        Actual_skills VARCHAR(600),
        Recommended_skills VARCHAR(600),
        Recommended_courses VARCHAR(1200)
    );
    """
    with get_conn() as connection:
        with connection.cursor() as cursor:
            cursor.execute(create_sql)
        connection.commit()

def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field,
                cand_level, skills, recommended_skills, courses):
    insert_sql = """
        INSERT INTO user_data
        (Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field,
         User_level, Actual_skills, Recommended_skills, Recommended_courses)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    values = (
        name, email, str(res_score), timestamp, str(no_of_pages), reco_field,
        cand_level, skills, recommended_skills, courses
    )
    with get_conn() as connection:
        with connection.cursor() as cursor:
            cursor.execute(insert_sql, values)
        connection.commit()

# =========================
# PDF I/O + VIEW
# =========================
def extract_text_and_pages(file_path: str):
    text = ""
    page_count = 0
    with fitz.open(file_path) as doc:
        page_count = doc.page_count
        for page in doc:
            text += page.get_text()
    return text, page_count

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    iframe = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000"></iframe>'
    st.markdown(iframe, unsafe_allow_html=True)

# =========================
# NLP EXTRACTORS
# =========================
EMAIL_RE = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
PHONE_RE = r"(\+?\d{1,4}[\s-]?)?(?:\(?\d{3,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}"

# Master skills vocabulary (expand as needed)
SKILLS_VOCAB = [
    # Data / ML / AI
    "Python", "R", "NumPy", "Pandas", "Scikit-learn", "TensorFlow", "Keras",
    "PyTorch", "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Data Analysis", "Data Visualization", "Matplotlib", "Seaborn",
    "Statistics", "EDA",

    # Web
    "JavaScript", "TypeScript", "React", "Angular", "Vue", "Next.js",
    "Node.js", "Express", "Django", "Flask", "FastAPI", "REST", "GraphQL",
    "HTML", "CSS", "Sass", "Tailwind",

    # Mobile
    "Android", "Kotlin", "Java", "Flutter", "Dart", "Swift", "iOS", "Xcode",

    # DevOps / Cloud
    "Git", "GitHub", "GitLab", "Docker", "Kubernetes", "AWS", "Azure", "GCP",
    "CI/CD", "Linux",

    # DB / Backend
    "SQL", "PostgreSQL", "MySQL", "MongoDB", "SQLite", "Redis",

    # UI/UX
    "Figma", "Adobe XD", "Sketch", "Wireframing", "Prototyping", "UX Research",
]

def build_skills_matcher(nlp_obj):
    matcher = PhraseMatcher(nlp_obj.vocab, attr="LOWER")
    patterns = [nlp_obj.make_doc(s) for s in SKILLS_VOCAB]
    matcher.add("SKILLS", patterns)
    return matcher

SKILL_MATCHER = build_skills_matcher(nlp)

def extract_email(text: str):
    m = re.search(EMAIL_RE, text)
    return m.group(0) if m else None

def extract_phone(text: str):
    m = re.search(PHONE_RE, text)
    if not m:
        return None
    # re.findall with capturing group returns only group; re.search gives the whole match at group(0)
    return m.group(0)

def extract_name(doc: spacy.tokens.Doc):
    # First PERSON entity as Name
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    # Fallback: try first two tokens capitalized
    tokens = [t.text for t in doc[:10] if t.is_title and t.is_alpha]
    return " ".join(tokens[:2]) if tokens else None

def extract_skills(doc: spacy.tokens.Doc):
    matches = SKILL_MATCHER(doc)
    found = sorted(set([doc[start:end].text for _, start, end in matches]))
    return found

# =========================
# CLASSIFICATION & RECOs
# =========================
DS_KEYWORDS = [s.lower() for s in [
    "tensorflow", "keras", "pytorch", "machine learning", "deep learning",
    "flask", "streamlit", "scikit-learn", "pandas", "numpy", "nlp"
]]
WEB_KEYWORDS = [s.lower() for s in [
    "react", "django", "node js", "node.js", "react js", "php", "laravel",
    "magento", "wordpress", "javascript", "angular", "typescript", "flask", "next.js"
]]
ANDROID_KEYWORDS = [s.lower() for s in [
    "android", "android development", "flutter", "kotlin", "xml", "kivy", "java"
]]
IOS_KEYWORDS = [s.lower() for s in [
    "ios", "ios development", "swift", "cocoa", "cocoa touch", "xcode"
]]
UIUX_KEYWORDS = [s.lower() for s in [
    "ux", "adobe xd", "figma", "zeplin", "balsamiq", "ui", "prototyping",
    "wireframes", "wireframing", "user research", "sketch"
]]

def predict_field(skills: list[str]) -> str:
    s = [x.lower() for x in skills]
    if any(k in s for k in DS_KEYWORDS):
        return "Data Science"
    if any(k in s for k in WEB_KEYWORDS):
        return "Web Development"
    if any(k in s for k in ANDROID_KEYWORDS):
        return "Android Development"
    if any(k in s for k in IOS_KEYWORDS):
        return "iOS Development"
    if any(k in s for k in UIUX_KEYWORDS):
        return "UI/UX"
    return ""

def recommended_skills_for(field: str) -> list[str]:
    mapping = {
        "Data Science": [
            "Data Visualization", "Predictive Analysis", "Statistical Modeling",
            "Data Mining", "Clustering & Classification", "ML Algorithms",
            "Scikit-learn", "TensorFlow", "PyTorch", "Flask", "Streamlit"
        ],
        "Web Development": [
            "React", "Django", "Node.js", "REST APIs", "GraphQL",
            "TypeScript", "Docker", "CI/CD"
        ],
        "Android Development": [
            "Kotlin", "Java", "Android Jetpack", "Room DB", "Coroutines",
            "Retrofit", "Unit Testing"
        ],
        "iOS Development": [
            "Swift", "UIKit", "SwiftUI", "CoreData", "Combine",
            "AutoLayout", "Unit Testing"
        ],
        "UI/UX": [
            "User Research", "Usability Testing", "Wireframing",
            "Prototyping", "Figma", "Design Systems"
        ],
    }
    return mapping.get(field, [])

def course_recommender(course_list):
    st.subheader("**Courses & Certificates 🎓 Recommendations**")
    c = 0
    rec_course = []
    no_of_reco = st.slider("Choose Number of Course Recommendations:", 1, 10, 4)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course

# =========================
# SCORE & UTILS
# =========================
def compute_resume_score(text: str) -> int:
    score = 0
    if re.search(r"\bObjective\b", text, flags=re.I):
        score += 20
    if re.search(r"\bDeclaration\b", text, flags=re.I):
        score += 20
    if re.search(r"\b(Hobbies|Interests)\b", text, flags=re.I):
        score += 20
    if re.search(r"\bAchievements?\b", text, flags=re.I):
        score += 20
    if re.search(r"\bProjects?\b", text, flags=re.I):
        score += 20
    return score

def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def fetch_yt_title(link: str) -> str:
    try:
        v = pafy.new(link)
        return v.title or "Video"
    except Exception:
        return "Video"

# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Smart Resume Analyzer", page_icon="./Logo/SRA_Logo.ico", layout="wide")

def run():
    ensure_table()

    st.title("Smart Resume Analyser (spaCy-powered)")
    st.sidebar.markdown("# Choose User")
    activities = ["Normal User", "Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)

    # Brand Image
    logo_path = "./Logo/SRA_Logo.jpg"
    if os.path.exists(logo_path):
        img = Image.open(logo_path).resize((250, 250))
        st.image(img)

    if choice == "Normal User":
        pdf_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
        if pdf_file is not None:
            # Save Uploaded PDF
            os.makedirs("./Uploaded_Resumes", exist_ok=True)
            save_path = os.path.join("./Uploaded_Resumes", pdf_file.name)
            with open(save_path, "wb") as f:
                f.write(pdf_file.getbuffer())

            # Preview
            show_pdf(save_path)

            # Extract text + pages
            resume_text, page_count = extract_text_and_pages(save_path)

            # NLP parse
            doc = nlp(resume_text)

            # Extract fields
            name = extract_name(doc)
            email = extract_email(resume_text)
            phone = extract_phone(resume_text)
            found_skills = extract_skills(doc)

            # Candidate level guess by page count
            cand_level = "Fresher" if page_count == 1 else ("Intermediate" if page_count == 2 else "Experienced")

            # UI: Basic info
            st.header("**Resume Analysis**")
            st.success(f"Hello {name or 'Candidate'}")
            st.subheader("**Your Basic Info**")
            st.text(f"Name: {name or 'Not Found'}")
            st.text(f"Email: {email or 'Not Found'}")
            st.text(f"Contact: {phone or 'Not Found'}")
            st.text(f"Resume pages: {page_count}")

            # Skills
            st.subheader("**Extracted Skills**")
            if found_skills:
                _ = st_tags(
                    label="### Skills we detected",
                    text="(you can add/remove if needed)",
                    value=found_skills,
                    key="skills_detected",
                )
            else:
                st.info("No skills detected from our vocabulary. Try adding more skills to your resume.")

            # Predict field + recommendations
            field = predict_field(found_skills)
            rec_skills = recommended_skills_for(field)
            rec_course = []

            if field:
                st.success(f"**Our analysis says you may be targeting: {field}**")
                if rec_skills:
                    st_tags(
                        label="### Recommended skills for you",
                        text="(based on your predicted field)",
                        value=rec_skills,
                        key="skills_reco",
                    )
                # Courses per field
                if field == "Data Science":
                    rec_course = course_recommender(ds_course)
                elif field == "Web Development":
                    rec_course = course_recommender(web_course)
                elif field == "Android Development":
                    rec_course = course_recommender(android_course)
                elif field == "iOS Development":
                    rec_course = course_recommender(ios_course)
                elif field == "UI/UX":
                    rec_course = course_recommender(uiux_course)
            else:
                st.warning("We couldn't confidently predict a field from your skills. Add more relevant skills to improve suggestions.")

            # Resume score
            st.subheader("**Resume Score 📝**")
            resume_score = compute_resume_score(resume_text)
            my_bar = st.progress(0)
            for i in range(resume_score):
                time.sleep(0.03)
                my_bar.progress(i + 1)
            st.success(f"**Your Resume Writing Score: {resume_score}**")
            st.caption("Note: Score is based on presence of helpful sections like Objective, Declaration, Hobbies/Interests, Achievements, and Projects.")

            # Save to DB
            ts = time.time()
            timestamp = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d_%H:%M:%S")
            insert_data(
                name or "",
                email or "",
                str(resume_score),
                timestamp,
                str(page_count),
                field,
                cand_level,
                ", ".join(found_skills),
                ", ".join(rec_skills),
                ", ".join(rec_course) if rec_course else "",
            )

            # Bonus videos
            st.header("**Bonus Video for Resume Writing Tips 💡**")
            resume_vid = random.choice(resume_videos)
            st.subheader("✅ " + fetch_yt_title(resume_vid))
            st.video(resume_vid)

            st.header("**Bonus Video for Interview 👨‍💼 Tips 💡**")
            interview_vid = random.choice(interview_videos)
            st.subheader("✅ " + fetch_yt_title(interview_vid))
            st.video(interview_vid)

            # Full text
            with st.expander("Show Extracted Text"):
                st.text_area("Raw Text", resume_text, height=300)

    else:
        st.success("Welcome to Admin Side")
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type="password")
        if st.button("Login"):
            if ad_user == "machine_learning_hub" and ad_password == "mlhub123":
                # Load data
                with get_conn() as connection:
                    df = pd.read_sql("SELECT * FROM user_data ORDER BY id DESC", connection)

                st.header("**Users' Data**")
                st.dataframe(df, use_container_width=True)
                st.markdown(get_table_download_link(df, "User_Data.csv", "⬇️ Download CSV"), unsafe_allow_html=True)

                # Pie charts
                if not df.empty:
                    st.subheader("📈 Predicted Field Distribution")
                    field_counts = df["Predicted_Field"].value_counts()
                    fig1 = px.pie(values=field_counts.values, names=field_counts.index, title="Predicted Fields")
                    st.plotly_chart(fig1, use_container_width=True)

                    st.subheader("📈 User Level Distribution")
                    ul_counts = df["User_level"].value_counts()
                    fig2 = px.pie(values=ul_counts.values, names=ul_counts.index, title="User Levels")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No records yet.")
            else:
                st.error("Wrong ID & Password Provided")

if __name__ == "__main__":
    run()
