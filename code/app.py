
import streamlit as st
import torch
import pandas as pd
import time
import joblib
from PIL import Image
from torchvision import transforms
import sys
import os
import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import hashlib
import secrets
import hmac
from pathlib import Path
from datetime import datetime

# ============================================================
# PROJECT SETUP
# ============================================================

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from model import DyslexiaCNN

st.set_page_config(
    page_title="NeuroScreen AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL UI THEME
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(31, 115, 255, 0.08), transparent 28%),
            radial-gradient(circle at 100% 10%, rgba(0, 190, 160, 0.08), transparent 25%),
            #f6f8fc;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #111c31 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #e8eef8 !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        padding: 8px 10px;
        border-radius: 10px;
    }

    .brand {
        padding: 8px 4px 22px 4px;
        border-bottom: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 18px;
    }

    .brand-title {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .brand-subtitle {
        color: #91a4c1 !important;
        font-size: 12px;
        margin-top: 4px;
    }

    .hero {
        background: linear-gradient(135deg, #0d1b34 0%, #123b68 55%, #087f7a 100%);
        padding: 34px 36px;
        border-radius: 22px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 16px 40px rgba(13, 27, 52, 0.16);
    }

    .hero h1 {
        color: white !important;
        font-size: 34px;
        margin: 0 0 8px 0;
        letter-spacing: -1px;
    }

    .hero p {
        color: #d6e4f5 !important;
        margin: 0;
        font-size: 15px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 800;
        color: #12213a;
        margin: 8px 0 16px 0;
    }

    .card {
        background: rgba(255,255,255,0.92);
        border: 1px solid #e4e9f2;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 8px 28px rgba(20, 35, 60, 0.06);
        margin-bottom: 16px;
    }

    .card-title {
        font-size: 16px;
        font-weight: 700;
        color: #16243b;
        margin-bottom: 6px;
    }

    .card-text {
        color: #66758d;
        font-size: 13px;
        line-height: 1.55;
    }

    .metric-card {
        background: white;
        border: 1px solid #e4e9f2;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 6px 22px rgba(20, 35, 60, 0.05);
    }

    .metric-label {
        color: #718096;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .6px;
    }

    .metric-value {
        color: #10213a;
        font-size: 27px;
        font-weight: 800;
        margin-top: 4px;
    }

    .status-pill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        background: #eaf7f4;
        color: #087c70;
    }

    .stButton > button {
        border-radius: 11px;
        border: 0;
        font-weight: 700;
        min-height: 42px;
        transition: all .2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(18, 50, 90, .15);
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #b8c5d8;
        border-radius: 16px;
        padding: 8px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e4e9f2;
        padding: 15px;
        border-radius: 15px;
    }

    .footer-note {
        color: #7b8799;
        font-size: 11px;
        text-align: center;
        margin-top: 28px;
    }

    h1, h2, h3 {
        color: #12213a;
    }

    .stProgress > div > div > div > div {
        border-radius: 999px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# GLOBAL HELPERS
# ============================================================

def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def card(title, text):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="card-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    
# ============================================================
# DATABASE + SECURE LOGIN
# ============================================================

DB_PATH = Path(__file__).resolve().parent / "neuroscreen.db"


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn




def initialize_database():
    with get_db_connection() as conn:

        # Admin users
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # Registered children
        conn.execute("""
            CREATE TABLE IF NOT EXISTS children (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                child_id TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                age INTEGER NOT NULL,
                grade TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # Permanent assessment results
        conn.execute("""
            CREATE TABLE IF NOT EXISTS assessment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                child_id TEXT NOT NULL,
                assessment_type TEXT NOT NULL,
                result TEXT,
                confidence REAL,
                details TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (child_id)
                    REFERENCES children (child_id)
            )
        """)


        # Update an existing assessment_results table safely
        existing_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(assessment_results)"
            ).fetchall()
        }

        required_columns = {
            "child_id": "TEXT",
            "assessment_type": "TEXT",
            "result": "TEXT",
            "confidence": "REAL",
            "details": "TEXT",
            "result_json": "TEXT",
            "created_at": "TEXT"
        }

        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                conn.execute(
                    f"ALTER TABLE assessment_results "
                    f"ADD COLUMN {column_name} {column_type}"
                )


        conn.commit()



def save_assessment_result(
    child_id,
    assessment_type,
    result,
    confidence=None,
    details=None
):
    import json
    from datetime import datetime

    # Package the prediction into JSON for the existing database schema
    result_data = {
        "result": result,
        "confidence": confidence,
        "details": details
    }

    result_json = json.dumps(result_data, ensure_ascii=False)

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO assessment_results (
                child_id,
                assessment_type,
                result,
                confidence,
                details,
                result_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                child_id,
                assessment_type,
                result,
                confidence,
                str(details) if details is not None else None,
                result_json,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )
        conn.commit()



def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        600_000
    )

    return salt.hex(), password_hash.hex()


def verify_password(password, salt_hex, expected_hash):
    salt = bytes.fromhex(salt_hex)
    _, actual_hash = hash_password(password, salt)

    return hmac.compare_digest(actual_hash, expected_hash)


initialize_database()

# ============================================================
# LOGIN GATE
# ============================================================

if not st.session_state.get("authenticated", False):

    st.markdown("""
        <div style="text-align:center; padding:35px 0 15px;">
            <h1>🧠 NeuroScreen AI</h1>
            <p>Secure Child Learning Screening Platform</p>
        </div>
    """, unsafe_allow_html=True)

    with get_db_connection() as conn:
        user_count = conn.execute(
            "SELECT COUNT(*) FROM users"
        ).fetchone()[0]

    if user_count == 0:

        st.subheader("Create the first administrator account")
        st.info(
            "Set up your administrator credentials. "
            "Keep the password private."
        )

        with st.form("create_admin_form"):
            username = st.text_input(
                "Administrator username",
                placeholder="Choose a username"
            ).strip().lower()

            password = st.text_input(
                "Create password",
                type="password"
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password"
            )

            create_admin = st.form_submit_button(
                "Create administrator",
                type="primary",
                use_container_width=True
            )

        if create_admin:
            if not username or len(username) < 4:
                st.error("Username must contain at least 4 characters.")

            elif len(password) < 10:
                st.error("Use a password with at least 10 characters.")

            elif password != confirm_password:
                st.error("The passwords do not match.")

            else:
                salt, password_hash = hash_password(password)

                try:
                    with get_db_connection() as conn:
                        conn.execute("""
                            INSERT INTO users
                            (username, password_salt, password_hash, created_at)
                            VALUES (?, ?, ?, ?)
                        """, (
                            username,
                            salt,
                            password_hash,
                            datetime.now().isoformat(timespec="seconds")
                        ))

                    st.success(
                        "Administrator created. You can now log in."
                    )
                    st.rerun()

                except sqlite3.IntegrityError:
                    st.error("That username already exists.")

    else:

        st.subheader("Administrator Login")

        with st.form("login_form"):
            username = st.text_input("Username").strip().lower()

            password = st.text_input(
                "Password",
                type="password"
            )

            login_clicked = st.form_submit_button(
                "Login",
                type="primary",
                use_container_width=True
            )

        if login_clicked:
            with get_db_connection() as conn:
                user = conn.execute("""
                    SELECT username, password_salt, password_hash
                    FROM users
                    WHERE username = ?
                """, (username,)).fetchone()

            if user and verify_password(
                password,
                user["password_salt"],
                user["password_hash"]
            ):
                st.session_state["authenticated"] = True
                st.session_state["logged_in_user"] = user["username"]
                st.rerun()

            else:
                st.error("Invalid username or password.")

    st.caption("Authorized staff only · Research prototype")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================


st.sidebar.markdown("""
<div class="brand">
    <div class="brand-title">🧠 NeuroScreen AI</div>
    <div class="brand-subtitle">Child Learning Screening Platform</div>
</div>
""", unsafe_allow_html=True)

# Sidebar styling
st.sidebar.markdown("""
<style>
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #64748B !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #111827 !important;
}

.selected-child-card {
    background-color: #1E3A5F;
    border: 2px solid #60A5FA;
    border-radius: 10px;
    padding: 14px;
    margin: 8px 0 12px 0;
    overflow-wrap: anywhere;
}

.selected-child-title {
    color: #BFDBFE;
    font-size: 12px;
    font-weight: 600;
}

.selected-child-name {
    color: #FFFFFF;
    font-size: 17px;
    font-weight: 700;
    margin-top: 6px;
}

.selected-child-detail {
    color: #FFFFFF;
    font-size: 13px;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Load registered children
with get_db_connection() as conn:
    registered_children = conn.execute("""
        SELECT child_id, full_name, age, grade
        FROM children
        ORDER BY full_name
    """).fetchall()

if registered_children:
    child_options = {
        f"{child['full_name']} ({child['child_id']})": dict(child)
        for child in registered_children
    }

    child_labels = list(child_options.keys())

    current_child_id = st.session_state.get("child_id")

    default_index = next(
        (
            i for i, label in enumerate(child_labels)
            if child_options[label]["child_id"] == current_child_id
        ),
        0
    )

    st.sidebar.markdown("### 👧 Selected Child")

    selected_label = st.sidebar.selectbox(
        "Choose a registered child",
        child_labels,
        index=default_index,
        key="global_child_selector"
    )

    # Get the selected child BEFORE displaying the card
    selected_child = child_options[selected_label]

    # Store the active child profile
    st.session_state["child_id"] = selected_child["child_id"]
    st.session_state["child_name"] = selected_child["full_name"]
    st.session_state["child_age"] = selected_child["age"]
    st.session_state["child_grade"] = selected_child["grade"]

    # High-contrast selected-child card
    st.sidebar.markdown(
        f"""
        <div class="selected-child-card">
            <div class="selected-child-title">
                CURRENTLY SELECTED CHILD
            </div>
            <div class="selected-child-name">
                {selected_child['full_name']}
            </div>
            <div class="selected-child-detail">
                ID: {selected_child['child_id']}
            </div>
            <div class="selected-child-detail">
                Age: {selected_child['age']} years
                &nbsp;|&nbsp; {selected_child['grade']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.sidebar.info("Register a child in Child Profile first.")

    for key in (
        "child_id",
        "child_name",
        "child_age",
        "child_grade"
    ):
        st.session_state.pop(key, None)

# Workspace navigation

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "🏠 Dashboard",
        "👤 Child Profile",
        "✍️ Handwriting Screening",
        "📖 Reading Assessment",
        "🔢 Dyscalculia Screening",
        "🔢 Mathematics Assessment",
        "👁️ Eye Tracking Assessment",
        "📊 Overall Risk Analysis",
        "📄 Final Report"
    ]
)


st.sidebar.markdown("---")
st.sidebar.caption("AI-assisted research prototype")
st.sidebar.caption("Not a medical diagnosis")

st.sidebar.caption(
    f"Logged in: {st.session_state['logged_in_user']}"
)

if st.sidebar.button("🔒 Logout", use_container_width=True):
    st.session_state["authenticated"] = False
    st.session_state.pop("logged_in_user", None)
    st.rerun()



# ============================================================
# DEVICE + MODELS
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

HANDWRITING_MODEL_PATH = "models/dyslexia_cnn.pth"
DYSCALCULIA_MODEL_PATH = "models/dyscalculia_app_model.pkl"

handwriting_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@st.cache_resource
def load_handwriting_model():
    model = DyslexiaCNN(num_classes=3)
    model.load_state_dict(
        torch.load(
            HANDWRITING_MODEL_PATH,
            map_location=device
        )
    )
    model = model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_dyscalculia_model():
    return joblib.load(DYSCALCULIA_MODEL_PATH)


@st.cache_resource
def load_reading_whisper_model():
    """Load the speech-to-text model once per Streamlit server process."""
    from faster_whisper import WhisperModel

    # CPU/int8 is a practical default for a CPU-only student project.
    # The first run downloads the model files and requires internet access.
    return WhisperModel("base", device="cpu", compute_type="int8")

# ============================================================
# DYSCALCULIA AI PREDICTION
# ============================================================

DYSCALCULIA_FEATURES = [
    "Mean_ACC_ANS",
    "Mean_RTs_ANS",
    "Accuracy_SymbolicComp",
    "RTs_SymbolicComp",
    "Enumeration_RTs",
    "Enumeration_Accuracy",
    "wm_K",
    "Comparison_RTs",
    "Comparison_Accuracy",
    "Visuospatial_RTs",
    "Visuospatial_Accuracy"
]


def run_dyscalculia_prediction():

    required_features = DYSCALCULIA_FEATURES

    missing_features = []

    for feature in required_features:

        if feature not in st.session_state:
            missing_features.append(feature)

    if missing_features:

        return None, None, missing_features

    # --------------------------------------------------------
    # Collect the 11 features from the completed child tasks
    # --------------------------------------------------------

    input_data = {

        "Mean_ACC_ANS":
            st.session_state["Mean_ACC_ANS"],

        "Mean_RTs_ANS":
            st.session_state["Mean_RTs_ANS"],

        "Accuracy_SymbolicComp":
            st.session_state["Accuracy_SymbolicComp"],

        "RTs_SymbolicComp":
            st.session_state["RTs_SymbolicComp"],

        "Enumeration_RTs":
            st.session_state["Enumeration_RTs"],

        "Enumeration_Accuracy":
            st.session_state["Enumeration_Accuracy"],

        "wm_K":
            st.session_state["wm_K"],

        "Comparison_RTs":
            st.session_state["Comparison_RTs"],

        "Comparison_Accuracy":
            st.session_state["Comparison_Accuracy"],

        "Visuospatial_RTs":
            st.session_state["Visuospatial_RTs"],

        "Visuospatial_Accuracy":
            st.session_state["Visuospatial_Accuracy"]
    }

    input_df = pd.DataFrame(
        [input_data],
        columns=required_features
    )

    # --------------------------------------------------------
    # Load new model
    # --------------------------------------------------------

    model = load_dyscalculia_model()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    classes = list(model.classes_)

    probability_dict = {
        cls: probability * 100
        for cls, probability in zip(
            classes,
            probabilities
        )
    }

    return prediction, probability_dict, None

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    hero(
        "AI Child Learning Screening",
        "A structured research platform for handwriting, mathematics and cognitive-performance screening."
    )

    st.markdown('<div class="section-title">Assessment overview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        card(
            "✍️ Handwriting",
            "CNN-based classification of handwriting samples into Normal, Corrected and Reversal patterns."
        )

    with c2:
        card(
            "🔢 Mathematical performance",
            "Grade-based mathematics assessment measuring accuracy and response time."
        )

    with c3:
        card(
            "🧠 Cognitive pattern screening",
            "Research model using mathematical, reaction-time, symbolic and visuospatial features."
        )

    st.markdown('<div class="section-title">Workflow</div>', unsafe_allow_html=True)

    steps = st.columns(4)

    workflow = [
        ("01", "Profile", "Create the child's assessment profile."),
        ("02", "Screen", "Run handwriting and mathematics assessments."),
        ("03", "Analyze", "Review model outputs and performance metrics."),
        ("04", "Report", "View the combined screening summary.")
    ]

    for col, (num, title, desc) in zip(steps, workflow):
        with col:
            card(f"{num} · {title}", desc)

    st.info(
        "This platform supports screening/research only. A model output should not be interpreted as a clinical diagnosis."
    )


# ============================================================
# CHILD PROFILE
# ============================================================


elif page == "👤 Child Profile":

    hero(
        "Child Registration",
        "Register a child and generate a unique NeuroScreen ID."
    )

    with st.container(border=True):

        st.markdown("### 🧒 Child details")

        with st.form("child_registration_form"):

            full_name = st.text_input(
                "Full Name",
                placeholder="Enter child's full name"
            ).strip()

            c1, c2 = st.columns(2)

            with c1:
                age = st.selectbox(
                    "Age",
                    list(range(6, 13)),
                    index=2
                )

            with c2:
                grade = st.selectbox(
                    "Grade",
                    [f"Grade {i}" for i in range(1, 8)]
                )

            st.caption(
                "Child ID will be generated automatically "
                "after registration."
            )

            save_profile = st.form_submit_button(
                "Register Child",
                type="primary",
                use_container_width=True
            )

        if save_profile:

            if not full_name:
                st.error("Please enter the child's full name.")

            elif len(full_name.split()) < 1:
                st.error("Please enter a valid name.")

            elif not full_name.replace(" ", "").replace("-", "").replace("'", "").isalpha():
                st.error(
                    "Name should contain letters, spaces, "
                    "hyphens or apostrophes only."
                )

            else:
                import re

                name_part = re.sub(
                    r"[^A-Za-z]",
                    "",
                    full_name
                ).upper()[:3]

                name_part = name_part.ljust(3, "X")

                year = datetime.now().year

                try:
                    with get_db_connection() as conn:

                        # Generate a unique sequential ID.
                        last_child = conn.execute(
                            """
                            SELECT child_id
                            FROM children
                            WHERE child_id LIKE ?
                            ORDER BY child_id DESC
                            LIMIT 1
                            """,
                            (f"NS-{name_part}-{year}-%",)
                        ).fetchone()

                        if last_child:
                            last_serial = int(
                                last_child["child_id"].split("-")[-1]
                            )
                            serial = last_serial + 1
                        else:
                            serial = 1

                        new_child_id = (
                            f"NS-{name_part}-{year}-{serial:04d}"
                        )

                        conn.execute(
                            """
                            INSERT INTO children
                            (
                                child_id,
                                full_name,
                                age,
                                grade,
                                created_by,
                                created_at
                            )
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                new_child_id,
                                full_name,
                                age,
                                grade,
                                st.session_state["logged_in_user"],
                                datetime.now().isoformat(
                                    timespec="seconds"
                                )
                            )
                        )

                    st.session_state["child_id"] = new_child_id
                    st.session_state["child_name"] = full_name
                    st.session_state["child_age"] = age
                    st.session_state["child_grade"] = grade

                    st.success("Child registered successfully!")
                    st.rerun()

                except sqlite3.IntegrityError:
                    st.error(
                        "This Child ID already exists. "
                        "Please register again."
                    )

    # Registered children list
    st.markdown("### 📋 Registered Children")

    with get_db_connection() as conn:
        children = conn.execute(
            """
            SELECT child_id, full_name, age, grade, created_at
            FROM children
            ORDER BY id DESC
            """
        ).fetchall()

    if children:
        children_df = pd.DataFrame(
            [dict(child) for child in children]
        )

        st.dataframe(
            children_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Registered records are stored in the local SQLite database."
        )
    else:
        st.info("No children registered yet.")

    # Current selected profile
    if "child_id" in st.session_state:

        st.markdown("### Current profile")

        a, b, c = st.columns(3)

        with a:
            metric_card(
                "Child ID",
                st.session_state["child_id"]
            )

        with b:
            metric_card(
                "Age",
                f"{st.session_state['child_age']} years"
            )

        with c:
            metric_card(
                "Grade",
                st.session_state["child_grade"]
            )

        st.write(
            f"**Child Name:** "
            f"{st.session_state.get('child_name', 'Not available')}"
        )

# ============================================================
# HANDWRITING SCREENING
# ============================================================


elif page == "✍️ Handwriting Screening":

    hero(
        "Handwriting Screening",
        "AI-assisted handwriting pattern analysis for children aged 6–12."
    )

    st.info(
        "Research screening tool only. This assessment cannot "
        "diagnose or rule out dyslexia or dysgraphia."
    )

    # --------------------------------------------------
    # SELECTED CHILD
    # --------------------------------------------------
    if "child_id" not in st.session_state:
        st.warning(
            "Please register or select a child in the sidebar first."
        )
        st.stop()

    child_id = st.session_state["child_id"]
    child_name = st.session_state.get("child_name", "Child")
    child_age = st.session_state.get("child_age", "Not available")
    child_grade = st.session_state.get("child_grade", "Not available")

    with st.container(border=True):
        st.markdown("### 👧 Child Details")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.write("**Name**")
            st.write(child_name)

        with c2:
            st.write("**Age**")
            st.write(f"{child_age} years")

        with c3:
            st.write("**Grade**")
            st.write(child_grade)

        st.caption(f"Child ID: {child_id}")

    st.markdown("---")

    # --------------------------------------------------
    # HANDWRITING IMAGE INPUT
    # --------------------------------------------------
    st.markdown("### ✍️ Upload Handwriting Sample")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help=(
            "Upload a clear image with the child's handwriting "
            "fully visible and minimal background clutter."
        ),
        key="handwriting_upload"
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            left, right = st.columns([1, 1.2])

            with left:
                st.image(
                    image,
                    caption="Uploaded handwriting sample",
                    use_container_width=True
                )

            with right:
                st.markdown("### 🧠 AI Analysis")

                if st.button(
                    "Analyze Handwriting",
                    type="primary",
                    use_container_width=True,
                    key="analyze_handwriting_button"
                ):

                    with st.spinner("Analyzing handwriting..."):

                        # Prepare the image using the existing transform
                        image_tensor = handwriting_transform(image)
                        image_tensor = image_tensor.unsqueeze(0).to(device)

                        # Load the trained CNN
                        model = load_handwriting_model()
                        model.eval()

                        with torch.no_grad():
                            output = model(image_tensor)

                            if output.ndim != 2 or output.shape[1] != 3:
                                raise ValueError(
                                    "Expected model output with "
                                    "three handwriting classes."
                                )

                            probabilities = torch.softmax(
                                output, dim=1
                            )[0]

                        # Preserve the trained model's class order
                        classes = [
                            "Corrected",
                            "Normal",
                            "Reversal"
                        ]

                        predicted_class = int(
                            torch.argmax(probabilities).item()
                        )

                        prediction = classes[predicted_class]

                        confidence = (
                            float(probabilities[predicted_class].item())
                            * 100
                        )

                        class_probabilities = {
                            classes[i]: (
                                float(probabilities[i].item()) * 100
                            )
                            for i in range(len(classes))
                        }

                        # Interpret the predicted dataset category
                        if prediction == "Normal":
                            handwriting_indicator = (
                                "The model predicted the Normal class. "
                                "This does not rule out dyslexia, "
                                "dysgraphia, or another learning difficulty."
                            )

                        elif prediction == "Corrected":
                            handwriting_indicator = (
                                "The model predicted the Corrected class. "
                                "This describes a dataset category and "
                                "does not establish a learning disability."
                            )

                        else:
                            handwriting_indicator = (
                                "The model predicted the Reversal class. "
                                "This pattern may warrant review alongside "
                                "age, writing development, reading, spelling, "
                                "and other assessment findings."
                            )

                        # Save the existing handwriting result
                        save_assessment_result(
                            child_id=child_id,
                            assessment_type="handwriting",
                            result=prediction,
                            confidence=confidence,
                            details=str({
                                "class_probabilities": class_probabilities,
                                "interpretation": handwriting_indicator
                            })
                        )

                        # Store results for the current child
                        st.session_state[
                            "handwriting_prediction"
                        ] = prediction

                        st.session_state[
                            "handwriting_confidence"
                        ] = confidence

                        st.session_state[
                            "handwriting_child_id"
                        ] = child_id

                        st.session_state[
                            "handwriting_probabilities"
                        ] = class_probabilities

                        st.session_state[
                            "handwriting_indicator"
                        ] = handwriting_indicator

                        st.session_state[
                            "handwriting_saved"
                        ] = True

                        st.success(
                            "Handwriting assessment saved successfully."
                        )

            # --------------------------------------------------
            # EXISTING HANDWRITING RESULTS
            # --------------------------------------------------
            if (
                "handwriting_prediction" in st.session_state
                and st.session_state.get("handwriting_child_id")
                == child_id
            ):

                prediction = st.session_state[
                    "handwriting_prediction"
                ]

                confidence = st.session_state[
                    "handwriting_confidence"
                ]

                class_probabilities = st.session_state[
                    "handwriting_probabilities"
                ]

                st.markdown("---")
                st.markdown("### 📊 Handwriting Results")

                r1, r2 = st.columns(2)

                with r1:
                    st.metric(
                        "Predicted handwriting class",
                        prediction
                    )

                with r2:
                    st.metric(
                        "Model confidence",
                        f"{confidence:.2f}%"
                    )

                st.caption(
                    "Model confidence reflects the CNN's classification "
                    "output. It is not the probability that the child "
                    "has dyslexia or dysgraphia."
                )

                st.markdown("#### Class probabilities")

                for class_name, probability in class_probabilities.items():
                    st.write(
                        f"**{class_name}:** {probability:.2f}%"
                    )

                    st.progress(
                        min(1.0, max(0.0, probability / 100))
                    )

                st.markdown("#### 🧠 Interpretation")

                st.write(
                    st.session_state["handwriting_indicator"]
                )

                st.warning(
                    "Dyslexia screening status: Not determined by this "
                    "classifier. A separately validated model trained "
                    "on appropriately labelled samples would be required "
                    "to estimate dyslexia-related screening risk."
                )

                st.caption(
                    f"Result associated with Child ID: {child_id}. "
                    "The handwriting assessment is saved in SQLite."
                )

            # --------------------------------------------------
            # EXPERIMENTAL DYSGRAPHIA-RELATED FEATURE ANALYSIS
            # --------------------------------------------------
            st.markdown("---")
            st.markdown("### ✏️ Dysgraphia-Related Writing Analysis")

            st.info(
                "Experimental research feature. The measurements below "
                "are image characteristics, not validated dysgraphia "
                "indicators. They cannot diagnose or rule out dysgraphia."
            )

            if st.button(
                "🔎 Analyze Writing Features",
                type="secondary",
                use_container_width=True,
                key="analyze_dysgraphia_features"
            ):

                try:
                    import numpy as np
                    from PIL import ImageOps

                    # Convert to grayscale and normalize contrast
                    gray_image = ImageOps.autocontrast(
                        image.convert("L")
                    ).resize((256, 256))

                    gray_array = np.asarray(
                        gray_image,
                        dtype=np.uint8
                    )

                    # Preliminary image-level measurements
                    ink_mask = gray_array < 128

                    ink_density = float(
                        np.mean(ink_mask)
                    )

                    row_ink = np.sum(
                        ink_mask,
                        axis=1
                    )

                    active_rows = row_ink > (256 * 0.015)

                    active_row_fraction = float(
                        np.mean(active_rows)
                    )

                    transitions = np.diff(
                        np.concatenate((
                            [False],
                            active_rows,
                            [False]
                        )).astype(np.int8)
                    )

                    band_count = int(
                        np.sum(transitions == 1)
                    )

                    st.markdown(
                        "#### 📊 Preliminary Image Measurements"
                    )

                    m1, m2, m3 = st.columns(3)

                    m1.metric(
                        "Dark-pixel coverage",
                        f"{ink_density * 100:.1f}%"
                    )

                    m2.metric(
                        "Rows containing ink",
                        f"{active_row_fraction * 100:.1f}%"
                    )

                    m3.metric(
                        "Horizontal ink bands",
                        str(band_count)
                    )

                    st.caption(
                        "These measurements describe the uploaded image. "
                        "They can be affected by lighting, paper, image "
                        "cropping, contrast, and the amount of writing."
                    )

                    st.markdown(
                        "#### 🧠 Dysgraphia Screening Status"
                    )

                    st.warning(
                        "A dysgraphia prediction is not available yet. "
                        "These experimental measurements have not been "
                        "validated to distinguish children with "
                        "dysgraphia from children without it. "
                        "No dysgraphia risk level is assigned."
                    )

                    st.write(
                        "Next step: obtain appropriately labelled "
                        "handwriting samples, train a separate model, "
                        "and evaluate it on held-out data before "
                        "displaying an AI dysgraphia prediction."
                    )

                except Exception as feature_error:
                    st.error(
                        "Could not analyze the handwriting image features."
                    )
                    st.exception(feature_error)

        except Exception as e:
            st.error(
                "Could not process or save this assessment. Check the "
                "image, model configuration, database connection, "
                "and selected child record."
            )
            st.exception(e)

    else:
        st.info(
            "Upload a handwriting image to begin the analysis."
        )



elif page == "📖 Reading Assessment":

    hero(
        "Reading Assessment",
        "Capture a child's spoken reading and compare the speech transcript with the passage."
    )

    st.info(
        "Research screening only. Speech recognition can make mistakes because of "
        "accent, pronunciation, background noise, or microphone quality. Reading "
        "measures cannot independently diagnose or rule out dyslexia."
    )

    child_id = st.session_state.get("child_id")
    if not child_id:
        st.warning("Please select or register a child in Child Profile first.")
        st.stop()

    st.success(f"Assessment linked to Child ID: {child_id}")

    passages = {
        "Grades 1–2": (
            "Sam has a little red ball. He takes the ball to the park. "
            "His friend Mia comes to play. They run and laugh in the sun. "
            "After playing, Sam puts the ball in his bag and walks home."
        ),
        "Grades 3–4": (
            "Every Saturday, Riya visits the garden near her home. "
            "She waters the plants and watches butterflies move between "
            "the flowers. One morning, she noticed a small bird building "
            "a nest in a tree. Riya stayed quiet so the bird would not be "
            "frightened. She returned the next week to see how the nest was growing."
        ),
        "Grades 5–7": (
            "A school science club decided to investigate how plants grow "
            "under different conditions. The students placed identical seeds "
            "in three pots and gave each pot a different amount of light. "
            "They kept the amount of water and the type of soil the same. "
            "Over several weeks, they measured the plants and recorded their "
            "observations. Their results helped them discuss why a fair experiment "
            "changes one factor at a time."
        ),
    }

    grade = st.session_state.get("child_grade", "Grade 3")
    try:
        grade_number = int(str(grade).split()[-1])
    except (ValueError, IndexError):
        grade_number = 3

    if grade_number <= 2:
        default_passage = "Grades 1–2"
    elif grade_number <= 4:
        default_passage = "Grades 3–4"
    else:
        default_passage = "Grades 5–7"

    passage_levels = list(passages.keys())
    passage_level = st.selectbox(
        "Choose reading level",
        passage_levels,
        index=passage_levels.index(default_passage),
        key="reading_passage_level"
    )
    passage = passages[passage_level]
    reference_words = passage.split()

    # Clear old audio/transcript if a different child is selected.
    if st.session_state.get("reading_audio_child_id") != child_id:
        for key in (
            "reading_audio_bytes", "reading_audio_duration",
            "reading_transcript", "reading_results",
            "reading_results_child_id", "reading_audio_passage_level"
        ):
            st.session_state.pop(key, None)
        st.session_state["reading_audio_child_id"] = child_id

    # Audio from a different passage must not be scored against the new passage.
    if (
        st.session_state.get("reading_audio_passage_level") is not None
        and st.session_state.get("reading_audio_passage_level") != passage_level
    ):
        for key in ("reading_audio_bytes", "reading_audio_duration", "reading_transcript", "reading_results"):
            st.session_state.pop(key, None)

    st.subheader("📖 Reading Passage")
    st.write(passage)
    st.caption(f"Reference passage: {len(reference_words)} words · {passage_level}")

    st.markdown("### 🎙️ Record the child's reading")
    st.write(
        "Allow microphone access when your browser asks. Press Start recording, "
        "ask the child to read the passage aloud, then press Stop recording. "
        "Use a quiet room and keep the microphone near the child."
    )

    try:
        from streamlit_mic_recorder import mic_recorder
    except ImportError:
        st.error(
            "The microphone component is not installed in this Python environment. "
            "Activate the project's .venv and run: "
            "python -m pip install streamlit-mic-recorder"
        )
        st.stop()

    audio_recording = mic_recorder(
        start_prompt="🎙️ Start recording",
        stop_prompt="⏹️ Stop recording",
        just_once=True,
        use_container_width=True,
        format="wav",
        key=f"reading_mic_recorder_{child_id}"
    )

    if audio_recording and audio_recording.get("bytes"):
        audio_bytes = bytes(audio_recording["bytes"])
        if audio_bytes:
            st.session_state["reading_audio_bytes"] = audio_bytes
            st.session_state["reading_audio_passage_level"] = passage_level
            st.session_state.pop("reading_transcript", None)
            st.session_state.pop("reading_results", None)

            # WAV duration is measured from the actual recording, not a manual timer.
            import io
            import wave
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                    frame_count = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration_seconds = frame_count / float(sample_rate) if sample_rate else None
                st.session_state["reading_audio_duration"] = duration_seconds
            except (wave.Error, EOFError, OSError, ZeroDivisionError):
                st.session_state["reading_audio_duration"] = None

    saved_audio = st.session_state.get("reading_audio_bytes")
    saved_audio_passage = st.session_state.get("reading_audio_passage_level")
    if saved_audio and saved_audio_passage == passage_level:
        st.success("Audio recording captured. You can listen before running transcription.")
        st.audio(saved_audio, format="audio/wav")
        audio_duration = st.session_state.get("reading_audio_duration")
        if audio_duration and audio_duration > 0:
            st.metric("Recording duration", f"{audio_duration:.1f} seconds")
        else:
            st.caption("The recording duration could not be read from the audio metadata.")

        if st.button(
            "🧠 Transcribe & Analyze Reading",
            type="primary",
            use_container_width=True,
            key="transcribe_reading_audio"
        ):
            import tempfile
            import os
            import string
            import re

            temp_audio_path = None
            try:
                with st.spinner(
                    "Loading the speech-recognition model and transcribing audio... "
                    "The first run may take longer while model files download."
                ):
                    model = load_reading_whisper_model()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
                        audio_file.write(saved_audio)
                        temp_audio_path = audio_file.name

                    segments, info = model.transcribe(
                        temp_audio_path,
                        language="en",
                        beam_size=5,
                        vad_filter=True
                    )
                    segment_list = list(segments)
                    transcript = " ".join(
                        segment.text.strip() for segment in segment_list if segment.text.strip()
                    ).strip()

                if not transcript:
                    st.error(
                        "No speech could be transcribed. Please try again in a quieter room "
                        "and make sure the child is close enough to the microphone."
                    )
                else:
                    def tokenize(text):
                        return re.findall(r"[a-z0-9']+", text.lower())

                    reference_tokens = tokenize(passage)
                    spoken_tokens = tokenize(transcript)

                    # Levenshtein alignment calculates substitutions, deletions and insertions.
                    n, m = len(reference_tokens), len(spoken_tokens)
                    dp = [[0] * (m + 1) for _ in range(n + 1)]
                    back = [[None] * (m + 1) for _ in range(n + 1)]
                    for i in range(1, n + 1):
                        dp[i][0] = i
                        back[i][0] = "deletion"
                    for j in range(1, m + 1):
                        dp[0][j] = j
                        back[0][j] = "insertion"
                    for i in range(1, n + 1):
                        for j in range(1, m + 1):
                            if reference_tokens[i - 1] == spoken_tokens[j - 1]:
                                dp[i][j] = dp[i - 1][j - 1]
                                back[i][j] = "match"
                            else:
                                choices = (
                                    (dp[i - 1][j - 1] + 1, "substitution"),
                                    (dp[i - 1][j] + 1, "deletion"),
                                    (dp[i][j - 1] + 1, "insertion")
                                )
                                dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])

                    substitutions = deletions = insertions = matches = 0
                    alignment = []
                    i, j = n, m
                    while i > 0 or j > 0:
                        operation = back[i][j]
                        if i > 0 and j > 0 and operation in ("match", "substitution"):
                            ref_word = reference_tokens[i - 1]
                            said_word = spoken_tokens[j - 1]
                            if operation == "match":
                                matches += 1
                                alignment.append(("Match", ref_word, said_word))
                            else:
                                substitutions += 1
                                alignment.append(("Substitution", ref_word, said_word))
                            i -= 1
                            j -= 1
                        elif i > 0 and operation == "deletion":
                            deletions += 1
                            alignment.append(("Skipped / not recognized", reference_tokens[i - 1], "—"))
                            i -= 1
                        elif j > 0 and operation == "insertion":
                            insertions += 1
                            alignment.append(("Extra / repeated word", "—", spoken_tokens[j - 1]))
                            j -= 1
                        else:
                            break
                    alignment.reverse()

                    total_errors = substitutions + deletions + insertions
                    wer = (total_errors / n * 100) if n else 0.0
                    reading_accuracy = (matches / n * 100) if n else 0.0
                    duration = st.session_state.get("reading_audio_duration")
                    if not duration or duration <= 0:
                        # Use speech segment end time if WAV metadata is unavailable.
                        duration = max((float(seg.end) for seg in segment_list), default=0.0)
                    wpm = (len(spoken_tokens) / (duration / 60)) if duration and duration > 0 else None

                    results = {
                        "passage_level": passage_level,
                        "reference_word_count": n,
                        "transcribed_word_count": m,
                        "matched_words": matches,
                        "substitutions": substitutions,
                        "deletions_or_skipped_words": deletions,
                        "insertions_or_repetitions": insertions,
                        "word_error_rate_percent": round(wer, 2),
                        "exact_word_match_percent": round(reading_accuracy, 2),
                        "recording_duration_seconds": round(duration, 2) if duration and duration > 0 else None,
                        "transcribed_words_per_minute": round(wpm, 2) if wpm is not None else None,
                        "transcript": transcript,
                        "method": "Browser microphone recording + Faster-Whisper speech-to-text + word-level edit-distance comparison",
                        "speech_language_detected": getattr(info, "language", "en"),
                        "interpretation": "Descriptive performance measures only; not a standardized score, validated risk score, or diagnosis."
                    }

                    st.session_state["reading_transcript"] = transcript
                    st.session_state["reading_results"] = results
                    st.session_state["reading_results_child_id"] = child_id

                    try:
                        save_assessment_result(
                            child_id=child_id,
                            assessment_type="reading",
                            result="Voice reading assessment recorded",
                            confidence=None,
                            details=__import__("json").dumps(results, ensure_ascii=False)
                        )
                        st.session_state["reading_results_saved"] = True
                        st.success("Transcription and reading metrics saved for this child.")
                    except Exception as save_error:
                        st.session_state["reading_results_saved"] = False
                        st.error(f"The analysis completed, but saving to the database failed: {save_error}")

                    st.rerun()

            except Exception as transcription_error:
                st.error(
                    "Could not transcribe this recording. Check that faster-whisper is "
                    "installed in the active project environment and that the computer "
                    "has internet access for the first model download."
                )
                st.exception(transcription_error)
            finally:
                if temp_audio_path and os.path.exists(temp_audio_path):
                    try:
                        os.remove(temp_audio_path)
                    except OSError:
                        pass

    else:
        st.caption("No recording saved yet. Record the passage to enable transcription.")

    transcript = st.session_state.get("reading_transcript")
    results = st.session_state.get("reading_results")
    results_child = st.session_state.get("reading_results_child_id")

    if transcript and results and results_child == child_id:
        st.divider()
        st.subheader("📝 Speech Recognition Transcript")
        st.text_area(
            "Words recognized from the recording",
            value=transcript,
            height=140,
            disabled=True,
            key="reading_transcript_display"
        )
        st.caption(
            "Review the transcript: speech recognition errors can be mistaken for reading errors. "
            "If it is clearly inaccurate, record again in a quiet room."
        )

        st.subheader("📊 Reading Performance Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Exact word matches", f"{results['exact_word_match_percent']:.1f}%")
        c2.metric("Word error rate", f"{results['word_error_rate_percent']:.1f}%")
        c3.metric(
            "Reading speed",
            f"{results['transcribed_words_per_minute']:.1f} words/min"
            if results.get("transcribed_words_per_minute") is not None else "Unavailable"
        )

        c4, c5, c6 = st.columns(3)
        c4.metric("Matched words", f"{results['matched_words']} / {results['reference_word_count']}")
        c5.metric("Substitutions", str(results["substitutions"]))
        c6.metric(
            "Skipped / extra words",
            f"{results['deletions_or_skipped_words']} / {results['insertions_or_repetitions']}"
        )

        st.markdown("#### Word-level comparison")
        alignment_rows = [
            {"Comparison": kind, "Passage word": ref_word, "Recognized word": said_word}
            for kind, ref_word, said_word in []
        ]
        # Recreate a compact comparison from the transcript for display.
        import re
        reference_tokens = re.findall(r"[a-z0-9']+", passage.lower())
        spoken_tokens = re.findall(r"[a-z0-9']+", transcript.lower())
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, reference_tokens, spoken_tokens, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace":
                st.write(f"**Substitution:** passage `{ ' '.join(reference_tokens[i1:i2]) }` → recognized `{ ' '.join(spoken_tokens[j1:j2]) }`")
            elif tag == "delete":
                st.write(f"**Possibly skipped/not recognized:** `{ ' '.join(reference_tokens[i1:i2]) }`")
            elif tag == "insert":
                st.write(f"**Extra/repeated recognized words:** `{ ' '.join(spoken_tokens[j1:j2]) }`")
        if all(tag == "equal" for tag, *_ in matcher.get_opcodes()):
            st.success("The recognized words match the reference passage exactly.")

        st.caption(
            "Word error rate uses substitutions + deletions + insertions divided by reference words. "
            "Exact word matches are a descriptive comparison, not a validated dyslexia score."
        )
        st.warning(
            "Do not assign a learning-disability diagnosis or risk level from this result alone. "
            "A trained professional should review reading, spelling, language, development, and context."
        )

elif page == "🔢 Dyscalculia Screening":

    hero(
        "Dyscalculia Child Assessment",
        "Interactive mathematical and cognitive tasks for screening."
    )

    st.info(
        "The child will complete several short tasks. "
        "The system records accuracy and reaction time automatically. "
        "Results are screening/research signals and are not a diagnosis."
    )

    # ========================================================
    # ANS NUMBER COMPARISON
    # ========================================================

    st.markdown("## 🔢 Task 1 — Number Comparison")

    st.write(
        "Choose the number that is larger."
    )

    # Initialize ANS session variables
    if "ans_trial" not in st.session_state:
        st.session_state["ans_trial"] = 0

    if "ans_correct" not in st.session_state:
        st.session_state["ans_correct"] = 0

    if "ans_response_times" not in st.session_state:
        st.session_state["ans_response_times"] = []

    if "ans_start_time" not in st.session_state:
        st.session_state["ans_start_time"] = None

    # --------------------------------------------------------
    # Start task
    # --------------------------------------------------------

    if st.session_state["ans_trial"] == 0:

        st.write(
            "You will see two numbers. "
            "Click the larger number as quickly and accurately as possible."
        )

        if st.button(
            "Start Number Comparison",
            type="primary",
            use_container_width=True,
            key="start_ans_task"
        ):

            st.session_state["ans_trial"] = 1
            st.session_state["ans_correct"] = 0
            st.session_state["ans_response_times"] = []
            st.session_state["ans_start_time"] = time.time()

            st.rerun()

    # --------------------------------------------------------
    # Generate question
    # --------------------------------------------------------

    elif st.session_state["ans_trial"] <= 5:

        # Generate numbers only once for the current trial
        if "ans_left" not in st.session_state:
            import random

            left = random.randint(1, 50)
            right = random.randint(1, 50)

            while left == right:
                right = random.randint(1, 50)

            st.session_state["ans_left"] = left
            st.session_state["ans_right"] = right

            st.session_state["ans_start_time"] = time.time()

        left = st.session_state["ans_left"]
        right = st.session_state["ans_right"]

        st.markdown(
            f"<h1 style='text-align:center;'>{left}  &nbsp;&nbsp;  {right}</h1>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                str(left),
                use_container_width=True,
                key=f"ans_left_{st.session_state['ans_trial']}"
            ):

                response_time = (
                    time.time()
                    - st.session_state["ans_start_time"]
                )

                st.session_state["ans_response_times"].append(
                    response_time
                )

                if left > right:
                    st.session_state["ans_correct"] += 1

                st.session_state["ans_trial"] += 1

                del st.session_state["ans_left"]
                del st.session_state["ans_right"]

                st.rerun()

        with col2:

            if st.button(
                str(right),
                use_container_width=True,
                key=f"ans_right_{st.session_state['ans_trial']}"
            ):

                response_time = (
                    time.time()
                    - st.session_state["ans_start_time"]
                )

                st.session_state["ans_response_times"].append(
                    response_time
                )

                if right > left:
                    st.session_state["ans_correct"] += 1

                st.session_state["ans_trial"] += 1

                del st.session_state["ans_left"]
                del st.session_state["ans_right"]

                st.rerun()

        st.progress(
            min(
                st.session_state["ans_trial"] / 6,
                1.0
            )
        )

        st.caption(
            f"Trial {st.session_state['ans_trial']} of 5"
        )

    # --------------------------------------------------------
    # Task completed
    # --------------------------------------------------------

    else:

        total_trials = 5

        accuracy = (
            st.session_state["ans_correct"]
            / total_trials
        )

        response_times = (
            st.session_state["ans_response_times"]
        )

        if response_times:
            average_rt = sum(response_times) / len(response_times)
        else:
            average_rt = 0.0

        # Save model features
        st.session_state["Mean_ACC_ANS"] = accuracy
        st.session_state["Mean_RTs_ANS"] = average_rt * 1000

        st.success("Number Comparison task completed!")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Correct answers",
                f"{st.session_state['ans_correct']}/{total_trials}"
            )

        with c2:
            st.metric(
                "Accuracy",
                f"{accuracy * 100:.1f}%"
            )

        with c3:
            st.metric(
                "Average reaction time",
                f"{average_rt:.2f} sec"
            )

        st.info(
            "ANS results have been saved for the dyscalculia screening."
        )

        if st.button(
            "Reset Number Comparison",
            key="reset_ans_task"
        ):

            for key in [
                "ans_trial",
                "ans_correct",
                "ans_response_times",
                "ans_start_time",
                "ans_left",
                "ans_right",
                "Mean_ACC_ANS",
                "Mean_RTs_ANS"
            ]:

                if key in st.session_state:
                    del st.session_state[key]

            st.rerun()

    # ========================================================
    # SYMBOLIC COMPARISON
    # ========================================================

    st.markdown("## 🔢 Task 2 — Symbolic Comparison")

    st.write(
        "Look at the two numbers and choose the correct mathematical symbol."
    )

    st.info(
        "Example: If 25 is smaller than 40, the correct answer is 25 < 40."
    )

    # Initialize variables
    if "symbolic_trial" not in st.session_state:
        st.session_state["symbolic_trial"] = 0

    if "symbolic_correct" not in st.session_state:
        st.session_state["symbolic_correct"] = 0

    if "symbolic_response_times" not in st.session_state:
        st.session_state["symbolic_response_times"] = []

    if "symbolic_start_time" not in st.session_state:
        st.session_state["symbolic_start_time"] = None

    # --------------------------------------------------------
    # Start task
    # --------------------------------------------------------

    if st.session_state["symbolic_trial"] == 0:

        st.write(
            "You will see two numbers. "
            "Select whether the first number is smaller than, "
            "greater than, or equal to the second number."
        )

        if st.button(
            "Start Symbolic Comparison",
            type="primary",
            use_container_width=True,
            key="start_symbolic_task"
        ):

            st.session_state["symbolic_trial"] = 1
            st.session_state["symbolic_correct"] = 0
            st.session_state["symbolic_response_times"] = []

            st.rerun()

    # --------------------------------------------------------
    # Generate question
    # --------------------------------------------------------

    elif st.session_state["symbolic_trial"] <= 5:

        if "symbolic_left" not in st.session_state:

            import random

            left = random.randint(1, 100)
            right = random.randint(1, 100)

            st.session_state["symbolic_left"] = left
            st.session_state["symbolic_right"] = right

            st.session_state["symbolic_start_time"] = time.time()

        left = st.session_state["symbolic_left"]
        right = st.session_state["symbolic_right"]

        # Display numbers
        st.markdown(
            f"""
            <div style="
                text-align:center;
                font-size:48px;
                font-weight:bold;
                padding:20px;
            ">
                {left} &nbsp;&nbsp; ? &nbsp;&nbsp; {right}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(
            "Choose the correct relationship:"
        )

        col1, col2, col3 = st.columns(3)

        # ----------------------------------------------------
        # Less than
        # ----------------------------------------------------

        with col1:

            if st.button(
                "<",
                use_container_width=True,
                key=f"symbolic_less_{st.session_state['symbolic_trial']}"
            ):

                response_time = (
                    time.time()
                    - st.session_state["symbolic_start_time"]
                )

                st.session_state["symbolic_response_times"].append(
                    response_time
                )

                if left < right:
                    st.session_state["symbolic_correct"] += 1

                st.session_state["symbolic_trial"] += 1

                del st.session_state["symbolic_left"]
                del st.session_state["symbolic_right"]

                st.rerun()

        # ----------------------------------------------------
        # Greater than
        # ----------------------------------------------------

        with col2:

            if st.button(
                ">",
                use_container_width=True,
                key=f"symbolic_greater_{st.session_state['symbolic_trial']}"
            ):

                response_time = (
                    time.time()
                    - st.session_state["symbolic_start_time"]
                )

                st.session_state["symbolic_response_times"].append(
                    response_time
                )

                if left > right:
                    st.session_state["symbolic_correct"] += 1

                st.session_state["symbolic_trial"] += 1

                del st.session_state["symbolic_left"]
                del st.session_state["symbolic_right"]

                st.rerun()

        # ----------------------------------------------------
        # Equal
        # ----------------------------------------------------

        with col3:

            if st.button(
                "=",
                use_container_width=True,
                key=f"symbolic_equal_{st.session_state['symbolic_trial']}"
            ):

                response_time = (
                    time.time()
                    - st.session_state["symbolic_start_time"]
                )

                st.session_state["symbolic_response_times"].append(
                    response_time
                )

                if left == right:
                    st.session_state["symbolic_correct"] += 1

                st.session_state["symbolic_trial"] += 1

                del st.session_state["symbolic_left"]
                del st.session_state["symbolic_right"]

                st.rerun()

        st.progress(
            min(
                st.session_state["symbolic_trial"] / 6,
                1.0
            )
        )

        st.caption(
            f"Trial {min(st.session_state['symbolic_trial'], 5)} of 5"
        )

    # --------------------------------------------------------
    # Task completed
    # --------------------------------------------------------

    else:

        total_trials = 5

        accuracy = (
            st.session_state["symbolic_correct"]
            / total_trials
        )

        response_times = (
            st.session_state["symbolic_response_times"]
        )

        if response_times:
            average_rt = (
                sum(response_times)
                / len(response_times)
            )
        else:
            average_rt = 0.0

        # Save model features
        st.session_state["Accuracy_SymbolicComp"] = accuracy

        st.session_state["RTs_SymbolicComp"] = (
            average_rt * 1000
        )

        st.success(
            "Symbolic Comparison task completed!"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Correct answers",
                f"{st.session_state['symbolic_correct']}/{total_trials}"
            )

        with c2:
            st.metric(
                "Accuracy",
                f"{accuracy * 100:.1f}%"
            )

        with c3:
            st.metric(
                "Average reaction time",
                f"{average_rt:.2f} sec"
            )

        st.info(
            "Symbolic Comparison results have been saved "
            "for the dyscalculia screening."
        )

        if st.button(
            "Reset Symbolic Comparison",
            key="reset_symbolic_task"
        ):

            for key in [
                "symbolic_trial",
                "symbolic_correct",
                "symbolic_response_times",
                "symbolic_start_time",
                "symbolic_left",
                "symbolic_right",
                "Accuracy_SymbolicComp",
                "RTs_SymbolicComp"
            ]:

                if key in st.session_state:
                    del st.session_state[key]

            st.rerun()

    # ========================================================
    # ENUMERATION TASK
    # ========================================================

    st.markdown("## 🔢 Task 3 — Enumeration")

    st.write(
        "Count the objects shown on the screen and select the correct number."
    )

    st.info(
        "Count carefully. Try to answer both accurately and as quickly as possible."
    )

    # Initialize variables
    if "enumeration_trial" not in st.session_state:
        st.session_state["enumeration_trial"] = 0

    if "enumeration_correct" not in st.session_state:
        st.session_state["enumeration_correct"] = 0

    if "enumeration_response_times" not in st.session_state:
        st.session_state["enumeration_response_times"] = []

    if "enumeration_start_time" not in st.session_state:
        st.session_state["enumeration_start_time"] = None

    # --------------------------------------------------------
    # Start task
    # --------------------------------------------------------

    if st.session_state["enumeration_trial"] == 0:

        st.write(
            "You will see a group of objects. "
            "Count them and select the correct number."
        )

        if st.button(
            "Start Enumeration",
            type="primary",
            use_container_width=True,
            key="start_enumeration_task"
        ):

            st.session_state["enumeration_trial"] = 1
            st.session_state["enumeration_correct"] = 0
            st.session_state["enumeration_response_times"] = []

            st.rerun()

    # --------------------------------------------------------
    # Generate question
    # --------------------------------------------------------

    elif st.session_state["enumeration_trial"] <= 5:

        if "enumeration_count" not in st.session_state:

            import random

            count = random.randint(3, 12)

            st.session_state["enumeration_count"] = count

            st.session_state["enumeration_start_time"] = time.time()

        count = st.session_state["enumeration_count"]

        # Create dots
        dots = " ".join(["●"] * count)

        st.markdown(
            f"""
            <div style="
                text-align:center;
                font-size:42px;
                font-weight:bold;
                padding:30px;
                letter-spacing:8px;
            ">
                {dots}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "<h3 style='text-align:center;'>How many objects are there?</h3>",
            unsafe_allow_html=True
        )

        # Create answer choices
        import random

        options = [count]

        while len(options) < 4:

            option = random.randint(
                max(1, count - 3),
                count + 3
            )

            if option not in options:
                options.append(option)

        random.shuffle(options)

        col1, col2, col3, col4 = st.columns(4)

        # ----------------------------------------------------
        # Answer 1
        # ----------------------------------------------------

        with col1:

            if st.button(
                str(options[0]),
                use_container_width=True,
                key=f"enumeration_answer_1_{st.session_state['enumeration_trial']}"
            ):

                response_time = (
                    time.time()
                    - st.session_state["enumeration_start_time"]
                )

                st.session_state["enumeration_response_times"].append(
                    response_time
                )

                if options[0] == count:
                    st.session_state["enumeration_correct"] += 1

                st.session_state["enumeration_trial"] += 1

                del st.session_state["enumeration_count"]

                st.rerun()

        # ----------------------------------------------------
        # Answer 2
        # ----------------------------------------------------

        with col2:

            if st.button(
                str(options[1]),
                use_container_width=True,
                key=f"enumeration_answer_2_{st.session_state['enumeration_trial']}"
            ):

                response_time = (
                    time.time()
                    - st.session_state["enumeration_start_time"]
                )

                st.session_state["enumeration_response_times"].append(
                    response_time
                )

                if options[1] == count:
                    st.session_state["enumeration_correct"] += 1

                st.session_state["enumeration_trial"] += 1

                del st.session_state["enumeration_count"]

                st.rerun()

        # ----------------------------------------------------
        # Answer 3
        # ----------------------------------------------------

        with col3:

            if st.button(
                str(options[2]),
                use_container_width=True,
                key=f"enumeration_answer_3_{st.session_state['enumeration_trial']}"
            ):

                response_time = (
                    time.time()
                    - st.session_state["enumeration_start_time"]
                )

                st.session_state["enumeration_response_times"].append(
                    response_time
                )

                if options[2] == count:
                    st.session_state["enumeration_correct"] += 1

                st.session_state["enumeration_trial"] += 1

                del st.session_state["enumeration_count"]

                st.rerun()

        # ----------------------------------------------------
        # Answer 4
        # ----------------------------------------------------

        with col4:

            if st.button(
                str(options[3]),
                use_container_width=True,
                key=f"enumeration_answer_4_{st.session_state['enumeration_trial']}"
            ):

                response_time = (
                    time.time()
                    - st.session_state["enumeration_start_time"]
                )

                st.session_state["enumeration_response_times"].append(
                    response_time
                )

                if options[3] == count:
                    st.session_state["enumeration_correct"] += 1

                st.session_state["enumeration_trial"] += 1

                del st.session_state["enumeration_count"]

                st.rerun()

        st.progress(
            min(
                st.session_state["enumeration_trial"] / 6,
                1.0
            )
        )

        st.caption(
            f"Trial {min(st.session_state['enumeration_trial'], 5)} of 5"
        )

    # --------------------------------------------------------
    # Task completed
    # --------------------------------------------------------

    else:

        total_trials = 5

        accuracy = (
            st.session_state["enumeration_correct"]
            / total_trials
        )

        response_times = (
            st.session_state["enumeration_response_times"]
        )

        if response_times:
            average_rt = (
                sum(response_times)
                / len(response_times)
            )
        else:
            average_rt = 0.0

        # Save model features
        st.session_state["Enumeration_Accuracy"] = accuracy

        st.session_state["Enumeration_RTs"] = (
            average_rt * 1000
        )

        st.success(
            "Enumeration task completed!"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Correct answers",
                f"{st.session_state['enumeration_correct']}/{total_trials}"
            )

        with c2:
            st.metric(
                "Accuracy",
                f"{accuracy * 100:.1f}%"
            )

        with c3:
            st.metric(
                "Average reaction time",
                f"{average_rt:.2f} sec"
            )

        st.info(
            "Enumeration results have been saved "
            "for the dyscalculia screening."
        )

        if st.button(
            "Reset Enumeration",
            key="reset_enumeration_task"
        ):

            for key in [
                "enumeration_trial",
                "enumeration_correct",
                "enumeration_response_times",
                "enumeration_start_time",
                "enumeration_count",
                "Enumeration_Accuracy",
                "Enumeration_RTs"
            ]:

                if key in st.session_state:
                    del st.session_state[key]

            st.rerun()

    # ========================================================
    # WORKING MEMORY TASK
    # ========================================================

    st.markdown("## 🧠 Task 4 — Working Memory")

    st.write(
        "Remember the sequence of numbers and enter them in the same order."
    )

    st.info(
        "You will briefly see a sequence. "
        "Remember it, then reproduce it in the same order."
    )

    # --------------------------------------------------------
    # Initialize variables
    # --------------------------------------------------------

    if "wm_trial" not in st.session_state:
        st.session_state["wm_trial"] = 0

    if "wm_correct" not in st.session_state:
        st.session_state["wm_correct"] = 0

    if "wm_sequence" not in st.session_state:
        st.session_state["wm_sequence"] = []

    if "wm_phase" not in st.session_state:
        st.session_state["wm_phase"] = "start"

    if "wm_start_time" not in st.session_state:
        st.session_state["wm_start_time"] = None

    if "wm_response_times" not in st.session_state:
        st.session_state["wm_response_times"] = []

    # ========================================================
    # START SCREEN
    # ========================================================

    if st.session_state["wm_phase"] == "start":

        st.write(
            "You will complete 5 memory trials."
        )

        if st.button(
            "Start Working Memory",
            type="primary",
            use_container_width=True,
            key="start_wm_task"
        ):

            import random

            st.session_state["wm_trial"] = 1
            st.session_state["wm_correct"] = 0
            st.session_state["wm_response_times"] = []

            sequence = [
                random.randint(1, 9)
                for _ in range(3)
            ]

            st.session_state["wm_sequence"] = sequence

            st.session_state["wm_phase"] = "show"

            st.rerun()

    # ========================================================
    # SHOW SEQUENCE
    # ========================================================

    elif st.session_state["wm_phase"] == "show":

        sequence = st.session_state["wm_sequence"]

        st.markdown(
            """
            <h3 style="text-align:center;">
                Remember this sequence:
            </h3>
            """,
            unsafe_allow_html=True
        )

        sequence_text = " → ".join(
            str(number)
            for number in sequence
        )

        st.markdown(
            f"""
            <div style="
                text-align:center;
                font-size:42px;
                font-weight:bold;
                padding:30px;
            ">
                {sequence_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            f"Trial {st.session_state['wm_trial']} of 5"
        )

        if st.button(
            "I Remember It",
            type="primary",
            use_container_width=True,
            key=f"wm_hide_{st.session_state['wm_trial']}"
        ):

            st.session_state["wm_phase"] = "answer"
            st.session_state["wm_start_time"] = time.time()

            st.rerun()

    # ========================================================
    # ANSWER SCREEN
    # ========================================================

    elif st.session_state["wm_phase"] == "answer":

        sequence = st.session_state["wm_sequence"]

        st.markdown(
            """
            <h3 style="text-align:center;">
                Now enter the sequence in the same order
            </h3>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            f"Trial {st.session_state['wm_trial']} of 5"
        )

        answer = st.text_input(
            "Enter the numbers without spaces",
            placeholder="Example: 739",
            key=f"wm_answer_{st.session_state['wm_trial']}"
        )

        if st.button(
            "Submit Answer",
            type="primary",
            use_container_width=True,
            key=f"wm_submit_{st.session_state['wm_trial']}"
        ):

            if not answer.strip():

                st.warning(
                    "Please enter your answer before submitting."
                )

            else:

                response_time = (
                    time.time()
                    - st.session_state["wm_start_time"]
                )

                st.session_state["wm_response_times"].append(
                    response_time
                )

                correct_answer = "".join(
                    str(number)
                    for number in sequence
                )

                if answer.strip() == correct_answer:

                    st.session_state["wm_correct"] += 1

                # ------------------------------------------------
                # Move to next trial
                # ------------------------------------------------

                if st.session_state["wm_trial"] < 5:

                    st.session_state["wm_trial"] += 1

                    import random

                    next_sequence = [
                        random.randint(1, 9)
                        for _ in range(3)
                    ]

                    st.session_state["wm_sequence"] = next_sequence

                    st.session_state["wm_phase"] = "show"

                else:

                    st.session_state["wm_phase"] = "completed"

                st.rerun()

    # ========================================================
    # COMPLETED
    # ========================================================

    elif st.session_state["wm_phase"] == "completed":

        total_trials = 5

        accuracy = (
            st.session_state["wm_correct"]
            / total_trials
        )

        response_times = (
            st.session_state["wm_response_times"]
        )

        if response_times:

            average_rt = (
                sum(response_times)
                / len(response_times)
            )

        else:

            average_rt = 0.0

        # Save working memory score
        st.session_state["wm_K"] = (
            st.session_state["wm_correct"]
        )

        st.success(
            "Working Memory task completed!"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Correct sequences",
                f"{st.session_state['wm_correct']}/{total_trials}"
            )

        with c2:

            st.metric(
                "Accuracy",
                f"{accuracy * 100:.1f}%"
            )

        with c3:

            st.metric(
                "Average response time",
                f"{average_rt:.2f} sec"
            )

        st.info(
            "Working Memory performance has been saved "
            "for the dyscalculia screening."
        )

        if st.button(
            "Reset Working Memory",
            key="reset_wm_task"
        ):

            for key in [
                "wm_trial",
                "wm_correct",
                "wm_sequence",
                "wm_phase",
                "wm_start_time",
                "wm_response_times",
                "wm_K"
            ]:

                if key in st.session_state:
                    del st.session_state[key]

            st.rerun()

        # ============================================================
    # TASK 5 — VISUOSPATIAL MEMORY
    # ============================================================

    st.markdown("---")
    st.subheader("🧩 Task 5: Visuospatial Memory")

    st.write(
        "Remember the highlighted positions and select the same "
        "positions after they disappear."
    )

    if "vs_phase" not in st.session_state:
        st.session_state["vs_phase"] = "start"

    if "vs_trial" not in st.session_state:
        st.session_state["vs_trial"] = 1

    if "vs_correct" not in st.session_state:
        st.session_state["vs_correct"] = 0

    if "vs_target" not in st.session_state:
        st.session_state["vs_target"] = []

    if "vs_selected" not in st.session_state:
        st.session_state["vs_selected"] = []

    if "vs_start_time" not in st.session_state:
        st.session_state["vs_start_time"] = 0.0

    if "vs_response_times" not in st.session_state:
        st.session_state["vs_response_times"] = []


    # ------------------------------------------------------------
    # START
    # ------------------------------------------------------------

    if st.session_state["vs_phase"] == "start":

        st.info("You will see 3 highlighted positions. Remember them.")

        if st.button("Start Task 5", key="vs_start"):

            import random

            st.session_state["vs_target"] = random.sample(
                range(16), 3
            )

            st.session_state["vs_selected"] = []
            st.session_state["vs_trial"] = 1
            st.session_state["vs_correct"] = 0
            st.session_state["vs_response_times"] = []

            st.session_state["vs_phase"] = "show"

            st.rerun()


    # ------------------------------------------------------------
    # SHOW POSITIONS
    # ------------------------------------------------------------

    elif st.session_state["vs_phase"] == "show":

        st.info(
            f"Trial {st.session_state['vs_trial']} of 5"
        )

        st.write("Remember these 3 positions:")

        cols = st.columns(4)

        for i in range(16):

            with cols[i % 4]:

                if i in st.session_state["vs_target"]:

                    st.button(
                        "🔵",
                        key=f"vs_show_{st.session_state['vs_trial']}_{i}",
                        disabled=True
                    )

                else:

                    st.button(
                        "⬜",
                        key=f"vs_show_{st.session_state['vs_trial']}_{i}",
                        disabled=True
                    )

        if st.button(
            "I Remember It",
            key=f"vs_remember_{st.session_state['vs_trial']}"
        ):

            st.session_state["vs_selected"] = []
            st.session_state["vs_start_time"] = time.time()
            st.session_state["vs_phase"] = "answer"

            st.rerun()


    # ------------------------------------------------------------
    # ANSWER
    # ------------------------------------------------------------

    elif st.session_state["vs_phase"] == "answer":

        st.info(
            f"Trial {st.session_state['vs_trial']} of 5"
        )

        st.write("Select the 3 positions you remember:")

        cols = st.columns(4)

        for i in range(16):

            with cols[i % 4]:

                if i in st.session_state["vs_selected"]:

                    if st.button(
                        "🔵",
                        key=f"vs_selected_{st.session_state['vs_trial']}_{i}"
                    ):

                        st.session_state["vs_selected"].remove(i)
                        st.rerun()

                else:

                    if st.button(
                        "⬜",
                        key=f"vs_answer_{st.session_state['vs_trial']}_{i}"
                    ):

                        if len(st.session_state["vs_selected"]) < 3:

                            st.session_state["vs_selected"].append(i)

                            st.rerun()


        st.write(
            f"Selected: {len(st.session_state['vs_selected'])} / 3"
        )


        if len(st.session_state["vs_selected"]) == 3:

            if st.button(
                "Submit Answer",
                key=f"vs_submit_{st.session_state['vs_trial']}"
            ):

                response_time = (
                    time.time()
                    - st.session_state["vs_start_time"]
                )

                st.session_state["vs_response_times"].append(
                    response_time
                )

                target = set(st.session_state["vs_target"])
                selected = set(st.session_state["vs_selected"])

                if target == selected:

                    st.session_state["vs_correct"] += 1
                    st.success("Correct! 🎉")

                else:

                    st.error("Incorrect.")


                # ------------------------------------------------
                # NEXT TRIAL
                # ------------------------------------------------

                if st.session_state["vs_trial"] < 5:

                    st.session_state["vs_trial"] += 1

                    import random

                    st.session_state["vs_target"] = random.sample(
                        range(16), 3
                    )

                    st.session_state["vs_selected"] = []
                    st.session_state["vs_start_time"] = 0.0
                    st.session_state["vs_phase"] = "show"

                    st.rerun()

                else:

                    accuracy = (
                        st.session_state["vs_correct"] / 5
                    ) * 100

                    response_times = st.session_state[
                        "vs_response_times"
                    ]

                    if response_times:

                        average_rt = (
                            sum(response_times)
                            / len(response_times)
                        )

                    else:

                        average_rt = 0.0

                    st.session_state[
                        "Visuospatial_Accuracy"
                    ] = accuracy

                    st.session_state[
                        "Visuospatial_RTs"
                    ] = average_rt * 1000

                    st.session_state["vs_phase"] = "completed"

                    st.rerun()


    # ------------------------------------------------------------
    # COMPLETED
    # ------------------------------------------------------------

    elif st.session_state["vs_phase"] == "completed":

        accuracy = st.session_state.get(
            "Visuospatial_Accuracy", 0
        )

        average_rt = st.session_state.get(
            "Visuospatial_RTs", 0
        )

        st.success("✅ Task 5 completed!")

        st.write(
            f"Accuracy: {accuracy:.1f}%"
        )

        st.write(
            f"Average Response Time: {average_rt:.0f} ms"
        )

        st.write(
            f"Correct Trials: "
            f"{st.session_state['vs_correct']} / 5"
        )

    # ============================================================
    # TASK 6 — QUANTITY COMPARISON
    # ============================================================

    st.markdown("---")
    st.subheader("🔢 Task 6 — Quantity Comparison")

    st.write(
        "Look at the two groups of objects and select the group "
        "that contains more objects."
    )

    # ------------------------------------------------------------
    # INITIALIZE SESSION STATE
    # ------------------------------------------------------------

    if "comp_phase" not in st.session_state:
        st.session_state["comp_phase"] = "start"

    if "comp_trial" not in st.session_state:
        st.session_state["comp_trial"] = 1

    if "comp_correct" not in st.session_state:
        st.session_state["comp_correct"] = 0

    if "comp_left_count" not in st.session_state:
        st.session_state["comp_left_count"] = 0

    if "comp_right_count" not in st.session_state:
        st.session_state["comp_right_count"] = 0

    if "comp_correct_answer" not in st.session_state:
        st.session_state["comp_correct_answer"] = ""

    if "comp_start_time" not in st.session_state:
        st.session_state["comp_start_time"] = 0.0

    if "comp_response_times" not in st.session_state:
        st.session_state["comp_response_times"] = []


    # ============================================================
    # START TASK
    # ============================================================

    if st.session_state["comp_phase"] == "start":

        st.info(
            "You will see two groups of objects. "
            "Choose the group that has more objects."
        )

        if st.button(
            "▶ Start Task 6",
            key="comp_start_button"
        ):

            import random

            left_count = random.randint(2, 8)
            right_count = random.randint(2, 8)

            # Make sure both groups are different
            while right_count == left_count:
                right_count = random.randint(2, 8)

            st.session_state["comp_left_count"] = left_count
            st.session_state["comp_right_count"] = right_count

            if left_count > right_count:
                st.session_state["comp_correct_answer"] = "LEFT"

            else:
                st.session_state["comp_correct_answer"] = "RIGHT"

            st.session_state["comp_trial"] = 1
            st.session_state["comp_correct"] = 0
            st.session_state["comp_response_times"] = []

            st.session_state["comp_start_time"] = time.time()

            st.session_state["comp_phase"] = "question"

            st.rerun()


    # ============================================================
    # QUESTION
    # ============================================================

    elif st.session_state["comp_phase"] == "question":

        st.info(
            f"Trial {st.session_state['comp_trial']} of 5"
        )

        left_count = st.session_state["comp_left_count"]
        right_count = st.session_state["comp_right_count"]

        # --------------------------------------------------------
        # DISPLAY GROUPS
        # --------------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### LEFT GROUP")

            left_objects = " ".join(
                ["●"] * left_count
            )

            st.markdown(
                f"""
                <div style="
                    padding:30px;
                    border:2px solid #cccccc;
                    border-radius:12px;
                    text-align:center;
                    font-size:32px;
                    min-height:100px;
                ">
                    {left_objects}
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown("### RIGHT GROUP")

            right_objects = " ".join(
                ["●"] * right_count
            )

            st.markdown(
                f"""
                <div style="
                    padding:30px;
                    border:2px solid #cccccc;
                    border-radius:12px;
                    text-align:center;
                    font-size:32px;
                    min-height:100px;
                ">
                    {right_objects}
                </div>
                """,
                unsafe_allow_html=True
            )


        st.write("")

        # --------------------------------------------------------
        # ANSWER BUTTONS
        # --------------------------------------------------------

        answer_col1, answer_col2 = st.columns(2)

        with answer_col1:

            if st.button(
                "⬅ LEFT GROUP",
                key=f"comp_left_{st.session_state['comp_trial']}"
            ):

                response_time = (
                    time.time()
                    - st.session_state["comp_start_time"]
                )

                st.session_state[
                    "comp_response_times"
                ].append(response_time)

                if (
                    st.session_state["comp_correct_answer"]
                    == "LEFT"
                ):

                    st.session_state["comp_correct"] += 1

                    st.success("Correct! 🎉")

                else:

                    st.error("Incorrect.")

                # --------------------------------------------
                # NEXT TRIAL
                # --------------------------------------------

                if st.session_state["comp_trial"] < 5:

                    st.session_state["comp_trial"] += 1

                    import random

                    left_count = random.randint(2, 8)
                    right_count = random.randint(2, 8)

                    while right_count == left_count:
                        right_count = random.randint(2, 8)

                    st.session_state[
                        "comp_left_count"
                    ] = left_count

                    st.session_state[
                        "comp_right_count"
                    ] = right_count

                    if left_count > right_count:

                        st.session_state[
                            "comp_correct_answer"
                        ] = "LEFT"

                    else:

                        st.session_state[
                            "comp_correct_answer"
                        ] = "RIGHT"

                    st.session_state[
                        "comp_start_time"
                    ] = time.time()

                    st.rerun()

                else:

                    # ----------------------------------------
                    # FINAL RESULT
                    # ----------------------------------------

                    accuracy = (
                        st.session_state["comp_correct"]
                        / 5
                    ) * 100

                    response_times = st.session_state[
                        "comp_response_times"
                    ]

                    if len(response_times) > 0:

                        average_rt = (
                            sum(response_times)
                            / len(response_times)
                        )

                    else:

                        average_rt = 0.0

                    st.session_state[
                        "Comparison_Accuracy"
                    ] = accuracy

                    st.session_state[
                        "Comparison_RTs"
                    ] = average_rt * 1000

                    st.session_state[
                        "comp_phase"
                    ] = "completed"

                    st.rerun()


        with answer_col2:

            if st.button(
                "RIGHT GROUP ➡",
                key=f"comp_right_{st.session_state['comp_trial']}"
            ):

                response_time = (
                    time.time()
                    - st.session_state["comp_start_time"]
                )

                st.session_state[
                    "comp_response_times"
                ].append(response_time)

                if (
                    st.session_state["comp_correct_answer"]
                    == "RIGHT"
                ):

                    st.session_state["comp_correct"] += 1

                    st.success("Correct! 🎉")

                else:

                    st.error("Incorrect.")

                # --------------------------------------------
                # NEXT TRIAL
                # --------------------------------------------

                if st.session_state["comp_trial"] < 5:

                    st.session_state["comp_trial"] += 1

                    import random

                    left_count = random.randint(2, 8)
                    right_count = random.randint(2, 8)

                    while right_count == left_count:
                        right_count = random.randint(2, 8)

                    st.session_state[
                        "comp_left_count"
                    ] = left_count

                    st.session_state[
                        "comp_right_count"
                    ] = right_count

                    if left_count > right_count:

                        st.session_state[
                            "comp_correct_answer"
                        ] = "LEFT"

                    else:

                        st.session_state[
                            "comp_correct_answer"
                        ] = "RIGHT"

                    st.session_state[
                        "comp_start_time"
                    ] = time.time()

                    st.rerun()

                else:

                    # ----------------------------------------
                    # FINAL RESULT
                    # ----------------------------------------

                    accuracy = (
                        st.session_state["comp_correct"]
                        / 5
                    ) * 100

                    response_times = st.session_state[
                        "comp_response_times"
                    ]

                    if len(response_times) > 0:

                        average_rt = (
                            sum(response_times)
                            / len(response_times)
                        )

                    else:

                        average_rt = 0.0

                    st.session_state[
                        "Comparison_Accuracy"
                    ] = accuracy

                    st.session_state[
                        "Comparison_RTs"
                    ] = average_rt * 1000

                    st.session_state[
                        "comp_phase"
                    ] = "completed"

                    st.rerun()


    # ============================================================
    # COMPLETED
    # ============================================================

    elif st.session_state["comp_phase"] == "completed":

        accuracy = st.session_state.get(
            "Comparison_Accuracy",
            0
        )

        average_rt_ms = st.session_state.get(
            "Comparison_RTs",
            0
        )

        st.success("✅ Task 6 completed!")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Accuracy",
                f"{accuracy:.1f}%"
            )

        with col2:

            st.metric(
                "Average Response Time",
                f"{average_rt_ms:.0f} ms"
            )

        st.write(
            f"Correct trials: "
            f"{st.session_state['comp_correct']} / 5"
        )

 
    # ========================================================
    # RUN DYSCALCULIA AI ANALYSIS
    # ========================================================

    st.markdown("---")
    st.subheader("🧠 Final Dyscalculia AI Analysis")

    missing_features = [
        feature
        for feature in DYSCALCULIA_FEATURES
        if feature not in st.session_state
    ]

    if missing_features:
        st.info(
            "Complete all six tasks before running the AI analysis."
        )
    else:
        if st.button(
            "🧠 Run Dyscalculia AI Analysis",
            type="primary",
            use_container_width=True,
            key="run_dyscalculia_ai"
        ):
            try:
                prediction, probability_dict, missing = (
                    run_dyscalculia_prediction()
                )

                if missing:
                    st.error(
                        "Missing assessment features: "
                        + ", ".join(missing)
                    )

                elif prediction is None or not probability_dict:
                    st.error("The model did not return a valid prediction.")

                else:
                    # Identify the model's dyscalculia class.
                    dyscalculia_class = next(
                        (
                            cls for cls in probability_dict
                            if "dyscalculia" in str(cls).lower()
                        ),
                        None
                    )

                    # For a binary model encoded as 0/1,
                    # class 1 is treated as the positive class.
                    if dyscalculia_class is None:
                        dyscalculia_class = next(
                            (
                                cls for cls in probability_dict
                                if str(cls).strip() == "1"
                            ),
                            None
                        )

                    if dyscalculia_class is None:
                        st.warning(
                            "Prediction generated, but the model's "
                            "class labels need to be verified before "
                            "displaying a dyscalculia probability."
                        )
                        st.write(f"Predicted model class: {prediction}")
                        st.write(
                            "Class probabilities:",
                            probability_dict
                        )

                    else:
                        dyscalculia_probability = float(
                            probability_dict[dyscalculia_class]
                        )

                        predicted_class_probability = float(
                            probability_dict[prediction]
                        )

                        st.session_state[
                            "dyscalculia_prediction"
                        ] = str(prediction)

                        st.session_state[
                            "dyscalculia_probability"
                        ] = dyscalculia_probability

                        st.session_state[
                            "dyscalculia_confidence"
                        ] = predicted_class_probability

                        st.session_state[
                            "dyscalculia_child_id"
                        ] = st.session_state.get("child_id")

                        st.success("Dyscalculia AI analysis completed!")

                        st.metric(
                            "Dyscalculia-class model probability",
                            f"{dyscalculia_probability:.1f}%"
                        )

                        st.caption(
                            "Experimental model output only; "
                            "not a clinical diagnosis."
                        )

            except Exception as e:
                st.error("Dyscalculia AI analysis failed.")
                st.exception(e)



# ============================================================
# MATHEMATICS ASSESSMENT
# ============================================================

elif page == "🔢 Mathematics Assessment":

    hero(
        "Mathematics Assessment",
        "A short grade-based assessment measuring accuracy and response speed."
    )

    if "child_id" not in st.session_state:

        st.warning(
            "Please complete the Child Profile before starting."
        )

    else:

        child_grade = st.session_state["child_grade"]


        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "Child ID",
                st.session_state["child_id"]
            )

        with c2:
            metric_card(
                "Age",
                f"{st.session_state['child_age']} years"
            )

        with c3:
            metric_card(
                "Grade",
                child_grade
            )

        grade_questions = {

            "Grade 1": [
                ("What is 3 + 2?", ["4", "5", "6", "7"], "5"),
                ("What is 7 - 3?", ["3", "4", "5", "6"], "4"),
                ("Which number is greater?", ["4", "6", "2", "3"], "6"),
                ("What comes after 8?", ["7", "9", "10", "6"], "9"),
                ("How many fingers are on one hand?", ["4", "5", "6", "10"], "5"),
            ],

            "Grade 2": [
                ("What is 15 + 7?", ["20", "21", "22", "23"], "22"),
                ("What is 24 - 9?", ["13", "14", "15", "16"], "15"),
                ("What is 3 × 4?", ["7", "10", "12", "14"], "12"),
                ("Which number is greater?", ["27", "72", "17", "27"], "72"),
                ("What is 20 ÷ 5?", ["2", "4", "5", "10"], "4"),
            ],

            "Grade 3": [
                ("What is 36 + 27?", ["53", "63", "73", "83"], "63"),
                ("What is 72 - 38?", ["24", "34", "44", "54"], "34"),
                ("What is 7 × 6?", ["36", "42", "48", "56"], "42"),
                ("What is 48 ÷ 6?", ["6", "7", "8", "9"], "8"),
                ("Which fraction is larger?", ["1/4", "1/2", "1/3", "1/5"], "1/2"),
            ],

            "Grade 4": [
                ("What is 125 + 238?", ["353", "363", "373", "383"], "363"),
                ("What is 500 - 275?", ["215", "225", "235", "245"], "225"),
                ("What is 12 × 8?", ["86", "96", "106", "116"], "96"),
                ("What is 144 ÷ 12?", ["10", "11", "12", "13"], "12"),
                ("What is 0.5 as a fraction?", ["1/2", "1/3", "2/3", "3/4"], "1/2"),
            ],

            "Grade 5": [
                ("What is 3/4 + 1/4?", ["1/2", "1", "3/4", "2"], "1"),
                ("What is 25% of 80?", ["10", "15", "20", "25"], "20"),
                ("What is 3.5 + 2.7?", ["5.2", "6.2", "6.7", "7.2"], "6.2"),
                ("What is 144 ÷ 12?", ["10", "11", "12", "14"], "12"),
                ("Which is greater?", ["0.45", "0.54", "0.40", "0.05"], "0.54"),
            ],

            "Grade 6": [
                ("What is -5 + 8?", ["2", "3", "4", "5"], "3"),
                ("What is 2/3 + 1/6?", ["1/2", "2/3", "5/6", "1"], "5/6"),
                ("What is 15% of 200?", ["20", "25", "30", "35"], "30"),
                ("What is 3 × 4.5?", ["12.5", "13.5", "14.5", "15.5"], "13.5"),
                ("Simplify 12/18.", ["2/3", "3/4", "4/5", "5/6"], "2/3"),
            ],

            "Grade 7": [
                ("Solve: 3x = 18", ["4", "5", "6", "7"], "6"),
                ("What is 2/3 + 5/6?", ["7/9", "3/2", "4/3", "5/3"], "3/2"),
                ("What is 20% of 350?", ["60", "70", "80", "90"], "70"),
                ("Simplify: 4x + 3x", ["6x", "7x", "8x", "9x"], "7x"),
                ("What is (-4) × (-6)?", ["-24", "-10", "10", "24"], "24"),
            ]
        }

        questions = grade_questions[child_grade]

        if "math_question_number" not in st.session_state:
            st.session_state["math_question_number"] = 0

        if "math_correct" not in st.session_state:
            st.session_state["math_correct"] = 0

        if "math_response_times" not in st.session_state:
            st.session_state["math_response_times"] = []

        question_number = st.session_state["math_question_number"]

        if (
            "math_timer_question" not in st.session_state
            or st.session_state["math_timer_question"] != question_number
        ):
            st.session_state["math_question_start"] = time.perf_counter()
            st.session_state["math_timer_question"] = question_number

        if question_number >= len(questions):

            st.success("Assessment completed.")

            correct = st.session_state["math_correct"]
            total = len(questions)
            accuracy = (correct / total) * 100

            response_times = st.session_state["math_response_times"]

            if response_times:
                average_response_time = (
                    sum(response_times) / len(response_times)
                )
            else:
                average_response_time = 0

            st.session_state["math_accuracy"] = accuracy
            st.session_state["math_avg_response_time"] = average_response_time
            st.session_state["math_correct_answers"] = correct
            st.session_state["math_total_questions"] = total

            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card("Accuracy", f"{accuracy:.1f}%")

            with c2:
                metric_card("Correct", f"{correct}/{total}")

            with c3:
                metric_card(
                    "Avg. response time",
                    f"{average_response_time:.2f}s"
                )

            st.progress(
                min(accuracy / 100, 1.0)
            )

            if accuracy >= 80:
                st.success("Strong performance on this assessment.")
            elif accuracy >= 60:
                st.warning("Some difficulty was observed on this assessment.")
            else:
                st.warning("Substantial difficulty was observed on this assessment.")

            if st.button(
                "Restart assessment",
                use_container_width=True
            ):

                st.session_state["math_question_number"] = 0
                st.session_state["math_correct"] = 0
                st.session_state["math_response_times"] = []

                st.session_state.pop(
                    "math_timer_question",
                    None
                )

                st.session_state.pop(
                    "math_question_start",
                    None
                )

                st.rerun()

        else:

            question, options, correct_answer = questions[question_number]

            st.progress(
                question_number / len(questions)
            )

            st.caption(
                f"Question {question_number + 1} of {len(questions)}"
            )

            st.markdown(
                f"### {question}"
            )

            answer = st.radio(
                "Choose an answer",
                options,
                key=f"math_answer_{question_number}"
            )

            if st.button(
                "Submit & continue",
                type="primary",
                use_container_width=True
            ):

                response_time = (
                    time.perf_counter()
                    - st.session_state["math_question_start"]
                )

                st.session_state["math_response_times"].append(
                    response_time
                )

                if answer == correct_answer:
                    st.session_state["math_correct"] += 1

                st.session_state["math_question_number"] += 1

                st.rerun()


# ============================================================
# EYE TRACKING
# ============================================================



elif page == "👁️ Eye Tracking Assessment":

    hero(
        "Eye Tracking Assessment",
        "Experimental webcam-based gaze calibration and visual tracking."
    )

    st.info(
        "Research prototype only. Webcam gaze estimates are approximate "
        "and must not be used alone to diagnose a learning disability."
    )

    st.write(
        "Keep your face centered in good lighting. During calibration, "
        "look directly at each target. During the tracking task, follow "
        "the target with your eyes while keeping your head relatively still."
    )

    st.warning(
        "This experimental task requires iris landmarks. Results may vary "
        "with lighting, camera quality, glasses, and head movement."
    )

    if st.button("Start Calibration & Tracking Task", type="primary"):

        model_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "face_landmarker.task"
        )

        if not os.path.exists(model_path):
            st.error(
                f"Face landmark model file not found: {model_path}"
            )

        else:
            camera = cv2.VideoCapture(0)
            face_landmarker = None

            if not camera.isOpened():
                camera.release()
                st.error(
                    "Could not open the webcam. Check camera permissions."
                )

            else:
                preview = st.empty()
                target_panel = st.empty()
                status = st.empty()

                frames_seen = 0
                frames_with_face = 0
                calibration_features = []
                calibration_targets = []
                tracking_errors = []
                tracking_samples = []

                last_timestamp_ms = -1
                last_preview_time = 0
                last_target_update = 0
                task_error = None

                # Target positions are percentages within the target panel.
                calibration_points = [
                    (15.0, 15.0),
                    (85.0, 15.0),
                    (50.0, 50.0),
                    (15.0, 85.0),
                    (85.0, 85.0),
                ]

                moving_points = [
                    (15.0, 50.0),
                    (25.0, 25.0),
                    (50.0, 15.0),
                    (75.0, 25.0),
                    (85.0, 50.0),
                    (75.0, 75.0),
                    (50.0, 85.0),
                    (25.0, 75.0),
                    (15.0, 50.0),
                ]

                def show_target(x, y, heading, detail):
                    target_panel.markdown(
                        f"""
                        <div style="
                            position:relative;
                            height:360px;
                            width:100%;
                            border:2px solid #888;
                            border-radius:12px;
                            background:#111827;
                            overflow:hidden;
                        ">
                          <div style="
                              position:absolute;
                              left:50%;
                              top:12px;
                              transform:translateX(-50%);
                              color:white;
                              font-size:16px;
                              font-weight:600;
                              white-space:nowrap;
                          ">{heading}</div>
                          <div style="
                              position:absolute;
                              left:{x}%;
                              top:{y}%;
                              transform:translate(-50%,-50%);
                              width:24px;
                              height:24px;
                              border-radius:50%;
                              background:#ff453a;
                              border:4px solid white;
                              box-shadow:0 0 12px #ff453a;
                          "></div>
                          <div style="
                              position:absolute;
                              left:50%;
                              bottom:12px;
                              transform:translateX(-50%);
                              color:white;
                              font-size:13px;
                              white-space:nowrap;
                          ">{detail}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                def get_iris_features(landmarks):
                    # MediaPipe Face Landmarker iris points:
                    # 468 = one iris center, 473 = other iris center.
                    if len(landmarks) < 478:
                        return None

                    features = []

                    eye_definitions = [
                        {
                            "iris": 468,
                            "corners": (362, 263),
                            "bounds": (362, 385, 387, 263, 373, 380),
                        },
                        {
                            "iris": 473,
                            "corners": (33, 133),
                            "bounds": (33, 160, 158, 133, 153, 144),
                        },
                    ]

                    for eye in eye_definitions:
                        iris = landmarks[eye["iris"]]
                        corner_a = landmarks[eye["corners"][0]]
                        corner_b = landmarks[eye["corners"][1]]

                        eye_points = [
                            landmarks[i] for i in eye["bounds"]
                        ]

                        min_x = min(p.x for p in eye_points)
                        max_x = max(p.x for p in eye_points)
                        min_y = min(p.y for p in eye_points)
                        max_y = max(p.y for p in eye_points)

                        eye_width = abs(corner_b.x - corner_a.x)
                        eye_height = max_y - min_y

                        if eye_width < 0.005 or eye_height < 0.003:
                            return None

                        # Normalize iris location within each eye.
                        x_value = (
                            (iris.x - min(corner_a.x, corner_b.x))
                            / eye_width
                        )
                        y_value = (iris.y - min_y) / eye_height

                        features.extend([x_value, y_value])

                    result = np.asarray(features, dtype=float)

                    if not np.all(np.isfinite(result)):
                        return None

                    return result

                try:
                    options = mp.tasks.vision.FaceLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(
                            model_asset_path=model_path
                        ),
                        running_mode=mp.tasks.vision.RunningMode.VIDEO,
                        num_faces=1
                    )

                    face_landmarker = (
                        mp.tasks.vision.FaceLandmarker
                        .create_from_options(options)
                    )

                    # -----------------------------------------
                    # STAGE 1: CALIBRATION
                    # -----------------------------------------
                    st.subheader("Stage 1: Gaze Calibration")
                    st.caption(
                        "Look at each red target. Each calibration point "
                        "is sampled for approximately 2 seconds."
                    )

                    for point_number, (target_x, target_y) in enumerate(
                        calibration_points, start=1
                    ):
                        show_target(
                            target_x,
                            target_y,
                            f"Calibration {point_number}/5",
                            "Look at the red target"
                        )

                        point_start = time.time()
                        collected_here = 0

                        while time.time() - point_start < 2.0:
                            success, frame = camera.read()

                            if not success:
                                raise RuntimeError(
                                    "The webcam stopped returning frames."
                                )

                            frames_seen += 1
                            frame = cv2.flip(frame, 1)

                            rgb_frame = cv2.cvtColor(
                                frame, cv2.COLOR_BGR2RGB
                            )

                            mp_image = mp.Image(
                                image_format=mp.ImageFormat.SRGB,
                                data=rgb_frame
                            )

                            timestamp_ms = int(time.time() * 1000)
                            timestamp_ms = max(
                                timestamp_ms, last_timestamp_ms + 1
                            )
                            last_timestamp_ms = timestamp_ms

                            result = face_landmarker.detect_for_video(
                                mp_image, timestamp_ms
                            )

                            if result.face_landmarks:
                                frames_with_face += 1
                                landmarks = result.face_landmarks[0]
                                features = get_iris_features(landmarks)

                                if features is not None:
                                    calibration_features.append(features)
                                    calibration_targets.append(
                                        [target_x, target_y]
                                    )
                                    collected_here += 1

                                # Draw eye outlines and iris centers.
                                eye_indices = [
                                    362, 385, 387, 263, 373, 380,
                                    33, 160, 158, 133, 153, 144
                                ]

                                for idx in eye_indices:
                                    lm = landmarks[idx]
                                    cv2.circle(
                                        frame,
                                        (
                                            int(lm.x * frame.shape[1]),
                                            int(lm.y * frame.shape[0])
                                        ),
                                        3,
                                        (255, 100, 0),
                                        -1
                                    )

                                for idx in (468, 473):
                                    lm = landmarks[idx]
                                    cv2.circle(
                                        frame,
                                        (
                                            int(lm.x * frame.shape[1]),
                                            int(lm.y * frame.shape[0])
                                        ),
                                        6,
                                        (0, 255, 255),
                                        2
                                    )

                            now = time.time()

                            if now - last_preview_time >= 0.12:
                                preview.image(
                                    cv2.cvtColor(
                                        frame, cv2.COLOR_BGR2RGB
                                    ),
                                    channels="RGB",
                                    use_container_width=True
                                )
                                last_preview_time = now

                            status.caption(
                                f"Calibrating point {point_number}/5 · "
                                f"Samples collected: {collected_here}"
                            )

                    # Fit a simple calibration mapping from the four
                    # normalized iris features to screen target positions.
                    if len(calibration_features) < 10:
                        raise RuntimeError(
                            "Too few valid iris samples were collected. "
                            "Check lighting, camera position, and whether "
                            "the model provides iris landmarks."
                        )

                    X = np.asarray(calibration_features, dtype=float)
                    Y = np.asarray(calibration_targets, dtype=float)

                    X_design = np.column_stack([
                        X, np.ones(len(X))
                    ])

                    coefficients, _, _, _ = np.linalg.lstsq(
                        X_design, Y, rcond=None
                    )

                    st.success(
                        f"Calibration samples collected: {len(X)}"
                    )

                    # -----------------------------------------
                    # STAGE 2: MOVING-TARGET TASK
                    # -----------------------------------------
                    st.subheader("Stage 2: Moving-Target Tracking")
                    st.caption(
                        "Follow the red target with your eyes. "
                        "The task lasts approximately 10 seconds."
                    )

                    task_start = time.time()
                    task_duration = 10.0

                    while time.time() - task_start < task_duration:
                        success, frame = camera.read()

                        if not success:
                            raise RuntimeError(
                                "The webcam stopped returning frames."
                            )

                        frames_seen += 1
                        frame = cv2.flip(frame, 1)

                        elapsed = time.time() - task_start

                        # Interpolate between successive target positions.
                        segment_length = 1.25
                        segment = int(elapsed / segment_length)
                        fraction = (
                            (elapsed % segment_length) / segment_length
                        )

                        start_point = moving_points[
                            min(segment, len(moving_points) - 2)
                        ]
                        end_point = moving_points[
                            min(segment + 1, len(moving_points) - 1)
                        ]

                        target_x = (
                            start_point[0]
                            + fraction * (end_point[0] - start_point[0])
                        )
                        target_y = (
                            start_point[1]
                            + fraction * (end_point[1] - start_point[1])
                        )

                        now = time.time()

                        if now - last_target_update >= 0.12:
                            show_target(
                                target_x,
                                target_y,
                                "Follow the moving target",
                                f"Task time: {elapsed:.1f}/{task_duration:.0f}s"
                            )
                            last_target_update = now

                        rgb_frame = cv2.cvtColor(
                            frame, cv2.COLOR_BGR2RGB
                        )

                        mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=rgb_frame
                        )

                        timestamp_ms = int(time.time() * 1000)
                        timestamp_ms = max(
                            timestamp_ms, last_timestamp_ms + 1
                        )
                        last_timestamp_ms = timestamp_ms

                        result = face_landmarker.detect_for_video(
                            mp_image, timestamp_ms
                        )

                        if result.face_landmarks:
                            frames_with_face += 1
                            landmarks = result.face_landmarks[0]
                            features = get_iris_features(landmarks)

                            if features is not None:
                                gaze_estimate = np.append(
                                    features, 1.0
                                ) @ coefficients

                                estimated_x = float(
                                    np.clip(gaze_estimate[0], 0, 100)
                                )
                                estimated_y = float(
                                    np.clip(gaze_estimate[1], 0, 100)
                                )

                                error = float(np.sqrt(
                                    (estimated_x - target_x) ** 2
                                    + (estimated_y - target_y) ** 2
                                ))

                                tracking_errors.append(error)
                                tracking_samples.append({
                                    "Estimated gaze X (%)": round(
                                        estimated_x, 1
                                    ),
                                    "Estimated gaze Y (%)": round(
                                        estimated_y, 1
                                    ),
                                    "Target X (%)": round(target_x, 1),
                                    "Target Y (%)": round(target_y, 1),
                                    "Estimated error (percentage points)": round(
                                        error, 1
                                    )
                                })

                            # Draw eye and iris points on the webcam image.
                            for idx in (
                                362, 385, 387, 263, 373, 380,
                                33, 160, 158, 133, 153, 144
                            ):
                                lm = landmarks[idx]
                                cv2.circle(
                                    frame,
                                    (
                                        int(lm.x * frame.shape[1]),
                                        int(lm.y * frame.shape[0])
                                    ),
                                    3,
                                    (255, 100, 0),
                                    -1
                                )

                            if len(landmarks) >= 478:
                                for idx in (468, 473):
                                    lm = landmarks[idx]
                                    cv2.circle(
                                        frame,
                                        (
                                            int(lm.x * frame.shape[1]),
                                            int(lm.y * frame.shape[0])
                                        ),
                                        6,
                                        (0, 255, 255),
                                        2
                                    )

                        if now - last_preview_time >= 0.12:
                            preview.image(
                                cv2.cvtColor(
                                    frame, cv2.COLOR_BGR2RGB
                                ),
                                channels="RGB",
                                use_container_width=True
                            )
                            last_preview_time = now

                        status.caption(
                            f"Tracking task: {elapsed:.1f}/"
                            f"{task_duration:.0f} seconds · "
                            f"Valid gaze samples: {len(tracking_errors)}"
                        )

                except Exception as error:
                    task_error = (
                        f"{type(error).__name__}: {error}"
                    )

                finally:
                    camera.release()

                    if face_landmarker is not None:
                        face_landmarker.close()

                if task_error:
                    st.error(f"Eye tracking could not finish: {task_error}")

                elif tracking_errors:
                    detection_rate = (
                        frames_with_face / frames_seen * 100
                        if frames_seen else 0
                    )

                    mean_error = float(np.mean(tracking_errors))
                    median_error = float(np.median(tracking_errors))

                    # Store measurements for later report integration.
                    summary = {
                        "Calibration samples": len(
                            calibration_features
                        ),
                        "Valid gaze samples": len(tracking_errors),
                        "Frames captured": frames_seen,
                        "Frames with face detected": frames_with_face,
                        "Face detection rate (%)": round(
                            detection_rate, 1
                        ),
                        "Mean estimated target error (percentage points)": round(
                            mean_error, 2
                        ),
                        "Median estimated target error (percentage points)": round(
                            median_error, 2
                        ),
                        "Interpretation": (
                            "Experimental gaze-tracking measurements; "
                            "not a validated disability screening score."
                        )
                    }

                    st.session_state["eye_tracking_result"] = (
                        "Experimental gaze-tracking task completed"
                    )
                    st.session_state["eye_tracking_summary"] = summary
                    st.session_state["eye_tracking_samples"] = (
                        tracking_samples
                    )

                    st.success("Calibration and tracking task completed.")

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        metric_card(
                            "Face detection rate",
                            f"{detection_rate:.1f}%"
                        )

                    with c2:
                        metric_card(
                            "Valid gaze samples",
                            str(len(tracking_errors))
                        )

                    with c3:
                        metric_card(
                            "Mean estimated error",
                            f"{mean_error:.1f} pp"
                        )

                    st.caption(
                        "The error is measured in percentage points of the "
                        "target panel, not degrees of visual angle. A lower "
                        "value means the estimated gaze was closer to the "
                        "moving target in this experimental task. It is not "
                        "a validated clinical score."
                    )

                    with st.expander("View assessment measurements"):
                        st.json(summary)

                    with st.expander("View sample gaze measurements"):
                        st.dataframe(
                            pd.DataFrame(tracking_samples),
                            use_container_width=True
                        )

                elif not task_error:
                    st.warning(
                        "The task ended without valid gaze samples. "
                        "Check lighting, face position, and iris landmark "
                        "availability before trying again."
                    )



# ============================================================
# OVERALL ANALYSIS
# ============================================================



elif page == "📊 Overall Risk Analysis":

    import json

    hero(
        "Overall Screening Analysis",
        "A consolidated view of the results collected across the assessment modules."
    )

    child_id = st.session_state.get("child_id")
    saved_results = {}

    # Load the selected child's latest saved result for each assessment.
    if child_id:
        try:
            with get_db_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT assessment_type, result, confidence,
                           details, result_json, created_at
                    FROM assessment_results
                    WHERE child_id = ?
                    ORDER BY created_at DESC
                    """,
                    (child_id,)
                ).fetchall()

            for row in rows:
                assessment_type = str(
                    row["assessment_type"] or ""
                ).strip().lower().replace(" ", "_")

                if not assessment_type or assessment_type in saved_results:
                    continue

                record = {
                    "result": row["result"],
                    "confidence": row["confidence"],
                    "details": row["details"],
                    "created_at": row["created_at"]
                }

                # Read the structured result when available.
                if row["result_json"]:
                    try:
                        structured = json.loads(row["result_json"])
                        if isinstance(structured, dict):
                            for key in ("result", "confidence", "details"):
                                if structured.get(key) is not None:
                                    record[key] = structured[key]
                    except (json.JSONDecodeError, TypeError):
                        pass

                saved_results[assessment_type] = record

        except Exception as e:
            st.error("Could not load saved assessment results.")
            st.exception(e)

    # Find the latest saved result using supported assessment-name variants.
    def find_saved_result(*names):
        for name in names:
            key = name.lower().replace(" ", "_")
            if key in saved_results:
                return saved_results[key]
        return None

    if child_id:
        st.markdown("### Child")

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card("Child ID", child_id)

        with c2:
            metric_card(
                "Age",
                f"{st.session_state.get('child_age', 'Not available')} years"
            )

        with c3:
            metric_card(
                "Grade",
                str(st.session_state.get("child_grade", "Not available"))
            )

        st.caption("Results shown for the currently selected child.")

    else:
        st.warning("Please select a child before viewing screening results.")

    st.markdown("### Screening results")

    c1, c2 = st.columns(2)

    # HANDWRITING
    with c1:
        st.markdown("#### ✍️ Handwriting")

        saved = find_saved_result("handwriting")

        if saved:
            prediction = saved.get("result")
            confidence = saved.get("confidence")

            if prediction:
                metric_card("Predicted pattern", str(prediction))

            if confidence is not None:
                metric_card("Model confidence", f"{float(confidence):.1f}%")

            details = saved.get("details")
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except (json.JSONDecodeError, TypeError):
                    pass

            if isinstance(details, dict) and details.get("interpretation"):
                st.info(details["interpretation"])
            elif st.session_state.get("handwriting_indicator"):
                st.info(st.session_state["handwriting_indicator"])

            st.caption(f"Saved on: {saved.get('created_at') or 'Date unavailable'}")

        elif st.session_state.get("handwriting_child_id") == child_id:
            metric_card(
                "Predicted pattern",
                st.session_state.get("handwriting_prediction", "Unavailable")
            )
            confidence = st.session_state.get("handwriting_confidence")
            if confidence is not None:
                st.caption(f"Model confidence: {confidence:.1f}%")
        else:
            st.info("No saved handwriting assessment found for this child.")

    # DYSCALCULIA
    with c2:
        st.markdown("#### 🧠 Dyscalculia")

        saved = find_saved_result("dyscalculia")

        if saved:
            if saved.get("result") is not None:
                metric_card("Saved result", str(saved["result"]))

            if saved.get("confidence") is not None:
                st.caption(
                    f"Stored model confidence: "
                    f"{float(saved['confidence']):.1f}%"
                )

            st.caption(f"Saved on: {saved.get('created_at') or 'Date unavailable'}")

        elif "dyscalculia_probability" in st.session_state:
            metric_card(
                "Model probability",
                f"{st.session_state['dyscalculia_probability']:.1f}%"
            )

            confidence = st.session_state.get("dyscalculia_confidence")
            if confidence is not None:
                st.caption(f"Model confidence: {confidence:.1f}%")

            st.caption("Showing the current session result; database saving still needs verification.")
        else:
            st.info("No saved dyscalculia assessment found for this child.")

    # MATHEMATICS
    st.markdown("### 🔢 Mathematics")

    saved = find_saved_result("mathematics", "math", "math_assessment")

    if saved:
        if saved.get("result") is not None:
            metric_card("Saved result", str(saved["result"]))

        if saved.get("details"):
            st.write(saved["details"])

        st.caption(f"Saved on: {saved.get('created_at') or 'Date unavailable'}")

    elif "math_accuracy" in st.session_state:
        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card("Accuracy", f"{st.session_state['math_accuracy']:.1f}%")

        with c2:
            metric_card(
                "Correct answers",
                f"{st.session_state.get('math_correct_answers', 0)}/"
                f"{st.session_state.get('math_total_questions', 0)}"
            )

        with c3:
            metric_card(
                "Average response",
                f"{st.session_state.get('math_avg_response_time', 0):.2f}s"
            )

        st.caption("Showing the current session result; database saving still needs verification.")
    else:
        st.info("No saved mathematics assessment found for this child.")

    # EYE TRACKING
    st.markdown("### 👁️ Eye Tracking")

    saved = find_saved_result("eye_tracking", "eye_tracking_assessment")

    if saved:
        if saved.get("result") is not None:
            metric_card("Saved result", str(saved["result"]))

        if saved.get("details"):
            st.write(saved["details"])

        st.caption(f"Saved on: {saved.get('created_at') or 'Date unavailable'}")

    elif "eye_tracking_result" in st.session_state:
        metric_card(
            "Assessment status",
            st.session_state["eye_tracking_result"]
        )

        if st.session_state.get("eye_tracking_summary"):
            st.write(st.session_state["eye_tracking_summary"])

        st.caption("Showing the current session result; database saving still needs verification.")
    else:
        st.info("No saved eye-tracking assessment found for this child.")

    # READING ASSESSMENT
    st.markdown("### 📖 Reading Assessment")
    saved = find_saved_result("reading")
    reading_details = None
    if saved and saved.get("details"):
        try:
            reading_details = json.loads(saved["details"])
        except (json.JSONDecodeError, TypeError):
            reading_details = None

    if reading_details:
        r1, r2, r3 = st.columns(3)
        r1.metric(
            "Exact word matches",
            f"{float(reading_details.get('exact_word_match_percent', 0)):.1f}%"
        )
        r2.metric(
            "Word error rate",
            f"{float(reading_details.get('word_error_rate_percent', 0)):.1f}%"
        )
        wpm_value = reading_details.get("transcribed_words_per_minute")
        r3.metric(
            "Reading speed",
            f"{float(wpm_value):.1f} words/min" if wpm_value is not None else "Unavailable"
        )
        st.caption(f"Saved on: {saved.get('created_at') or 'Date unavailable'}")
        st.caption(
            "Speech-recognition-based descriptive measures only; transcription errors can affect these metrics."
        )
    elif st.session_state.get("reading_results_child_id") == child_id and st.session_state.get("reading_results"):
        reading_details = st.session_state["reading_results"]
        r1, r2, r3 = st.columns(3)
        r1.metric("Exact word matches", f"{reading_details['exact_word_match_percent']:.1f}%")
        r2.metric("Word error rate", f"{reading_details['word_error_rate_percent']:.1f}%")
        wpm_value = reading_details.get("transcribed_words_per_minute")
        r3.metric("Reading speed", f"{float(wpm_value):.1f} words/min" if wpm_value is not None else "Unavailable")
        st.caption("Showing the current session result.")
    else:
        st.info("No saved voice reading assessment found for this child.")

    st.divider()

    st.warning(
        "This is an AI-assisted research screening summary, not a medical "
        "diagnosis. Handwriting class confidence is not a probability of "
        "dyslexia. Individual assessment outputs must be interpreted cautiously."
    )



# ============================================================
# FINAL REPORT
# ============================================================

elif page == "📄 Final Report":

    hero(
        "Final Screening Report",
        "Review the information and results collected during this session."
    )

    if "child_id" not in st.session_state:

        st.info(
            "Complete the Child Profile and assessments to populate the report."
        )

    else:

        st.markdown("### Child information")

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "Child ID",
                st.session_state["child_id"]
            )

        with c2:
            metric_card(
                "Age",
                f"{st.session_state['child_age']} years"
            )

        with c3:
            metric_card(
                "Grade",
                st.session_state["child_grade"]
            )

        st.markdown("### Module results")

        if "handwriting_prediction" in st.session_state:
            st.write(
                f"**Handwriting:** {st.session_state['handwriting_prediction']} "
                f"({st.session_state['handwriting_confidence']:.1f}% confidence)"
            )
        else:
            st.write("**Handwriting:** Not completed")

        if "dyscalculia_probability" in st.session_state:
            st.write(
                f"**Dyscalculia model probability:** "
                f"{st.session_state['dyscalculia_probability']:.1f}%"
            )
        else:
            st.write("**Dyscalculia:** Not completed")

        if "math_accuracy" in st.session_state:
            st.write(
                f"**Mathematics:** "
                f"{st.session_state['math_accuracy']:.1f}% accuracy, "
                f"{st.session_state['math_avg_response_time']:.2f}s average response"
            )
        else:
            st.write("**Mathematics:** Not completed")

        reading_results = st.session_state.get("reading_results")
        if (
            reading_results
            and st.session_state.get("reading_results_child_id") == st.session_state.get("child_id")
        ):
            st.write(
                f"**Voice reading:** {reading_results['exact_word_match_percent']:.1f}% exact word matches, "
                f"{reading_results['word_error_rate_percent']:.1f}% word error rate, "
                f"{reading_results.get('transcribed_words_per_minute'):.1f} words/min"
                if reading_results.get("transcribed_words_per_minute") is not None
                else f"**Voice reading:** {reading_results['exact_word_match_percent']:.1f}% exact word matches, "
                     f"{reading_results['word_error_rate_percent']:.1f}% word error rate"
            )
        else:
            st.write("**Voice reading:** Not completed in this session")

        st.divider()

        st.info(
            "This report summarizes AI-assisted screening outputs and performance measures. "
            "It is not a clinical diagnosis."
        )

st.markdown(
    '<div class="footer-note">NeuroScreen AI · Final-year research prototype · Screening support only</div>',
    unsafe_allow_html=True
)
