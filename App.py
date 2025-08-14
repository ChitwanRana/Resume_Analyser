# app.py
import os
import re
import time
import base64
import random
import datetime
import io
import numpy as np

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
def compute_resume_score(text: str):
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
# JOB DESCRIPTION MATCHING
# =========================
def match_resume_to_job(resume_text, job_description, skills):
    """Match resume to job description and give recommendations"""
    # Create spaCy docs
    resume_doc = nlp(resume_text)
    job_doc = nlp(job_description)
    
    # Extract job skills
    job_skills = extract_skills(job_doc)
    
    # Calculate skill match percentage
    matching_skills = set(skills).intersection(set(job_skills))
    missing_skills = set(job_skills) - set(skills)
    
    if job_skills:
        match_percentage = (len(matching_skills) / len(job_skills)) * 100
    else:
        match_percentage = 0
    
    # Semantic similarity
    resume_vector = resume_doc.vector
    job_vector = job_doc.vector
    
    # Calculate cosine similarity if vectors are not zero
    if np.any(resume_vector) and np.any(job_vector):
        similarity = np.dot(resume_vector, job_vector) / (np.linalg.norm(resume_vector) * np.linalg.norm(job_vector))
    else:
        similarity = 0
    
    return {
        "match_percentage": match_percentage,
        "matching_skills": list(matching_skills),
        "missing_skills": list(missing_skills),
        "semantic_similarity": similarity * 100  # Convert to percentage
    }

# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Smart Resume Analyzer", page_icon="./Logo/SRA_Logo.ico", layout="wide")

# Custom CSS for better visual appeal
def load_custom_css():
    st.markdown("""
    <style>
        /* Overall theme */
        .main {
            background-color: #f8f9fa;
            color: #212529;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #0366d6;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Cards for important sections */
        .stcard {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        /* Success buttons */
        .stButton>button {
            background-color: #28a745;
            color: white;
        }
        
        /* Metrics */
        .metric-label {
            font-weight: bold;
            font-size: 1.2rem;
            color: #333;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #0366d6;
        }
        
        /* Skill tags */
        .skills-tag {
            display: inline-block;
            background-color: #e1ecf4;
            color: #0366d6;
            padding: 5px 10px;
            margin: 5px;
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-in-out;
        }
        
        /* Admin dashboard */
        .admin-section {
            background-color: #f1f8ff;
            border-left: 4px solid #0366d6;
            padding: 15px;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# Modern card component
def st_card(title, content, icon=None):
    card_html = f"""
    <div class="stcard fade-in">
        <h3>{icon + ' ' if icon else ''}{title}</h3>
        <div>{content}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# Skill tag display
def st_skill_tags(skills, is_recommended=False):
    if not skills:
        return ""
        
    tags_html = ""
    for skill in skills:
        bg_color = "#e1ecf4" if not is_recommended else "#d4edda"
        text_color = "#0366d6" if not is_recommended else "#155724"
        tags_html += f'<span class="skills-tag" style="background-color:{bg_color};color:{text_color}">{skill}</span>'
    
    return tags_html

def run():
    ensure_table()
    load_custom_css()  # Apply custom styling

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

            # Add more detailed resume analysis - Enhanced extraction
            def extract_education(doc):
                """Extract education details using pattern matching and entity recognition"""
                edu_orgs = []
                edu_degrees = ["bachelor", "master", "phd", "b.tech", "m.tech", "b.e.", "m.e.", "b.s.", "m.s."]
                
                # Find education entities
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        # Check if it's likely an educational institution
                        if any(edu_term in ent.text.lower() for edu_term in ["university", "college", "institute", "school"]):
                            edu_orgs.append(ent.text)
                
                # Look for degree mentions
                degree_matches = []
                for sent in doc.sents:
                    sent_lower = sent.text.lower()
                    if any(degree in sent_lower for degree in edu_degrees):
                        degree_matches.append(sent.text)
                
                return {
                    "institutions": edu_orgs[:3],  # Top 3 likely institutions
                    "degree_info": degree_matches[:2]  # Top 2 degree mentions
                }

            def extract_experience(doc):
                """Extract work experience details"""
                experience = []
                
                # Look for experience section
                exp_section = False
                for sent in doc.sents:
                    sent_lower = sent.text.lower()
                    if re.search(r"\b(work|professional|employment)\s+experience\b", sent_lower):
                        exp_section = True
                        continue
                        
                    if exp_section and len(sent.text.strip()) > 20:
                        if re.search(r"\b(education|skills|projects)\b", sent_lower):
                            exp_section = False
                            break
                        if re.search(r"\b(20\d{2}|19\d{2})\b", sent.text):  # Date indicators
                            experience.append(sent.text)
                
                return experience[:3]  # Return top 3 experience points

            # Enhanced scoring system
            def compute_resume_score(text: str, skills_count: int) -> dict:
                """More detailed resume scoring with feedback"""
                score_details = {
                    "sections": {
                        "objective": {"present": False, "points": 10, "feedback": "Add a career objective for clarity"},
                        "education": {"present": False, "points": 15, "feedback": "Include your education details"},
                        "experience": {"present": False, "points": 20, "feedback": "Add work experience details"},
                        "skills": {"present": False, "points": 15, "feedback": "List your technical and soft skills"}, 
                        "projects": {"present": False, "points": 15, "feedback": "Include relevant projects"},
                        "achievements": {"present": False, "points": 10, "feedback": "Highlight your achievements"},
                        "contact": {"present": False, "points": 10, "feedback": "Ensure contact information is complete"},
                        "formatting": {"present": False, "points": 5, "feedback": "Improve overall document formatting"}
                    },
                    "total": 0,
                    "max_points": 100,
                    "missing_sections": []
                }
                
                # Check sections
                if re.search(r"\b(objective|summary|profile)\b", text, flags=re.I):
                    score_details["sections"]["objective"]["present"] = True
                
                if re.search(r"\beducation\b", text, flags=re.I):
                    score_details["sections"]["education"]["present"] = True
                    
                if re.search(r"\b(experience|work|employment)\b", text, flags=re.I):
                    score_details["sections"]["experience"]["present"] = True
                    
                if re.search(r"\bskills\b", text, flags=re.I) or skills_count >= 5:
                    score_details["sections"]["skills"]["present"] = True
                    
                if re.search(r"\bprojects?\b", text, flags=re.I):
                    score_details["sections"]["projects"]["present"] = True
                    
                if re.search(r"\bachievements?\b", text, flags=re.I):
                    score_details["sections"]["achievements"]["present"] = True
                    
                if re.search(EMAIL_RE, text) and re.search(PHONE_RE, text):
                    score_details["sections"]["contact"]["present"] = True
                    
                if len(text.split('\n')) > 10:
                    score_details["sections"]["formatting"]["present"] = True
                
                # Calculate total score
                for section, details in score_details["sections"].items():
                    if details["present"]:
                        score_details["total"] += details["points"]
                    else:
                        score_details["missing_sections"].append(details["feedback"])
                
                return score_details

            # Add new extraction functions
            education_info = extract_education(doc)
            experience_info = extract_experience(doc)

            # Enhanced scoring
            score_details = compute_resume_score(resume_text, len(found_skills))
            resume_score = score_details["total"]

            # Display results
            # Education details
            st.subheader("📚 Education Details")
            if education_info["institutions"]:
                st.write("**Detected Institutions:**")
                for inst in education_info["institutions"]:
                    st.write(f"- {inst}")
            else:
                st.info("No education institutions clearly detected. Consider making this section more explicit.")

            if education_info["degree_info"]:
                st.write("**Degree Information:**")
                for degree in education_info["degree_info"]:
                    st.write(f"- {degree}")

            # Work experience
            st.subheader("💼 Work Experience")
            if experience_info:
                for exp in experience_info:
                    st.write(f"- {exp}")
            else:
                st.info("No clear work experience detected. If you have relevant experience, make it more explicit.")

            # Enhanced score display with feedback
            st.subheader("**Resume Score 📝**")
            col1, col2 = st.columns([2,1])
            with col1:
                st.progress(resume_score/100)
                st.success(f"**Your Resume Score: {resume_score}/100**")
            with col2:
                gauge_chart = {
                    "type": "indicator",
                    "mode": "gauge+number",
                    "value": resume_score,
                    "gauge": {
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#5DA5DA" if resume_score < 60 else "#60BD68"},
                        "steps": [
                            {"range": [0, 40], "color": "#F15854"},
                            {"range": [40, 70], "color": "#FAA43A"},
                            {"range": [70, 100], "color": "#60BD68"}
                        ],
                    },
                    "title": {"text": "Resume Score"}
                }
                st.plotly_chart({"data": [gauge_chart], "layout": {"height": 250}}, use_container_width=True)

            # Show improvement areas - Fix the visibility issue
            if score_details["missing_sections"]:
                st.subheader("🚀 Areas for Improvement")
                for i, feedback in enumerate(score_details["missing_sections"]):
                    # Using a more noticeable warning box with numbered bullets
                    st.warning(f"{i+1}. {feedback}")
                    
                # Add a spacer to ensure visibility
                st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.success("Great job! Your resume covers all the recommended sections.")

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

            # Job description matching
            with st.expander("📝 **Compare with Job Description**", expanded=False):
                st.write("Paste a job description to see how well your resume matches:")
                job_description = st.text_area("Job Description", height=200)
                
                if job_description and st.button("Analyze Match"):
                    with st.spinner("Analyzing match..."):
                        match_results = match_resume_to_job(resume_text, job_description, found_skills)
                        
                        # Display results
                        st.subheader("Match Analysis")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Skills Match", f"{match_results['match_percentage']:.1f}%")
                        with col2:
                            st.metric("Overall Content Similarity", f"{match_results['semantic_similarity']:.1f}%")
                        
                        # Display matching and missing skills
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("✅ **Matching Skills**")
                            if match_results['matching_skills']:
                                for skill in match_results['matching_skills']:
                                    st.success(skill)
                            else:
                                st.info("No matching skills found")
                        
                        with col2:
                            st.write("⚠️ **Missing Skills**")
                            if match_results['missing_skills']:
                                for skill in match_results['missing_skills']:
                                    st.error(skill)
                            else:
                                st.success("No critical skills missing!")

    else:
        st.success("Welcome to Admin Side")
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type="password")
        if st.button("Login"):
            if ad_user == "machine_learning_hub" and ad_password == "mlhub123":
                # Load data
                with get_conn() as connection:
                    # Modified SQL to exclude timestamp or get all and then drop
                    df = pd.read_sql("SELECT * FROM user_data ORDER BY id DESC", connection)
                    
                    # Remove timestamp column from display
                    if 'Timestamp' in df.columns:
                        df = df.drop('Timestamp', axis=1)

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
