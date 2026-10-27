
import streamlit as st
import torch
import pandas as pd
import time
import json
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
import ast
import secrets
import hmac
from pathlib import Path
from datetime import datetime

# ============================================================
# PROJECT SETUP
# ============================================================

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from model import DyslexiaCNN
from assessment_thresholds import (
    MODEL_CONFIDENCE_MIN_PERCENT, TASK_ACCURACY_TARGET_PERCENT,
    READING_WORD_MATCH_TARGET_PERCENT, eye_tracking_quality, meets_minimum,
)

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
            var(--background-color, #f6f8fc);
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

    /* Streamlit light/dark theme compatibility: prefer theme variables. */
    .stApp { color: var(--text-color, inherit) !important; }
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stWidgetLabel"],
    .stApp label, .stApp p, .stApp li,
    .stApp [data-testid="stCaptionContainer"] {
        color: var(--text-color, inherit);
    }
    .stApp .card, .stApp .metric-card,
    .stApp div[data-testid="stMetric"],
    .stApp div[data-testid="stFileUploader"] {
        background: var(--secondary-background-color, #ffffff);
        color: var(--text-color, #172033);
        border-color: var(--border-color, #d9e0ea);
    }
    .stApp .card-title, .stApp .metric-value,
    .stApp .section-title, .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp h5, .stApp h6 {
        color: var(--text-color, #172033);
    }
    .stApp .card-text, .stApp .metric-label, .stApp .footer-note {
        color: var(--text-color, #526174);
        opacity: .88;
    }
    .stApp input, .stApp textarea, .stApp select,
    .stApp [data-baseweb="select"] > div {
        color: var(--text-color, inherit);
    }
    html[data-theme="dark"] .stApp,
    body[data-theme="dark"] .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(31,115,255,.10), transparent 28%),
            radial-gradient(circle at 100% 10%, rgba(0,190,160,.08), transparent 25%),
            var(--background-color, #0e1117) !important;
        color: var(--text-color, #fafafa) !important;
    }
    html[data-theme="dark"] .stApp .card,
    html[data-theme="dark"] .stApp .metric-card,
    html[data-theme="dark"] .stApp div[data-testid="stMetric"],
    html[data-theme="dark"] .stApp div[data-testid="stFileUploader"] {
        background: var(--secondary-background-color, #1b2029) !important;
        color: var(--text-color, #fafafa) !important;
        border-color: var(--border-color, #3a414d) !important;
    }
    html[data-theme="dark"] .stApp .section-title,
    html[data-theme="dark"] .stApp .card-title,
    html[data-theme="dark"] .stApp .metric-value,
    html[data-theme="dark"] .stApp h1,
    html[data-theme="dark"] .stApp h2,
    html[data-theme="dark"] .stApp h3,
    html[data-theme="dark"] .stApp h4,
    html[data-theme="dark"] .stApp h5,
    html[data-theme="dark"] .stApp h6,
    html[data-theme="dark"] .stApp p,
    html[data-theme="dark"] .stApp li,
    html[data-theme="dark"] .stApp label,
    html[data-theme="dark"] .stApp [data-testid="stMarkdownContainer"],
    html[data-theme="dark"] .stApp [data-testid="stCaptionContainer"] {
        color: var(--text-color, #fafafa) !important;
    }
    html[data-theme="dark"] .stApp .card-text,
    html[data-theme="dark"] .stApp .metric-label,
    html[data-theme="dark"] .stApp .footer-note {
        color: var(--text-color, #d1d5db) !important;
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

HANDWRITING_MODEL_PATH = Path(__file__).resolve().parent / "models" / "dyslexia_cnn.pth"
DYSCALCULIA_MODEL_PATH = Path(__file__).resolve().parent / "models" / "dyscalculia_app_model.pkl"

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

    # The small model is generally more accurate than base, at the cost of
    # slower transcription and a larger initial download. CPU/int8 keeps it
    # practical for a CPU-only student project.
    # The first run downloads the model files and requires internet access.
    return WhisperModel("small", device="cpu", compute_type="int8")

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
# HYBRID SCREENING ENGINE (PROTOTYPE / TRANSPARENT RULES)
# ============================================================

def parse_assessment_details(value):
    """Parse details saved as JSON or as an older Python dict string."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError, NameError):
        pass
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError, TypeError):
        return {}


def get_latest_assessment_record(child_id, assessment_type):
    """Return the newest saved assessment for a child, or None."""
    if not child_id:
        return None
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                """
                SELECT assessment_type, result, confidence, details, created_at
                FROM assessment_results
                WHERE child_id = ? AND LOWER(assessment_type) = LOWER(?)
                ORDER BY id DESC
                LIMIT 1
                """,
                (child_id, assessment_type)
            ).fetchone()
        if not row:
            return None
        return {
            "assessment_type": row["assessment_type"],
            "result": row["result"],
            "confidence": row["confidence"],
            "details": row["details"],
            "created_at": row["created_at"]
        }
    except Exception:
        return None


def build_dyslexia_evidence_summary(reading_details=None, handwriting_pattern=None):
    """Create a descriptive, non-diagnostic summary of reading + handwriting observations."""
    reading_details = reading_details if isinstance(reading_details, dict) else {}
    summary = {
        "summary_type": "descriptive_cross_domain_review",
        "reading_available": bool(reading_details),
        "handwriting_pattern": handwriting_pattern,
        "reading_accuracy_percent": reading_details.get(
            "exact_word_match_percent",
            reading_details.get("reading_accuracy_percent_approx")
        ),
        "word_error_rate_percent": reading_details.get("word_error_rate_percent"),
        "reading_speed_words_per_minute": reading_details.get("transcribed_words_per_minute"),
        "substitutions": reading_details.get("substitutions"),
        "skipped_words": reading_details.get("deletions_or_skipped_words"),
        "extra_words": reading_details.get("insertions_or_repetitions"),
        "status": "Insufficient data",
        "review_note": "Complete a voice reading assessment to create a cross-domain review summary.",
        "limitations": [
            "This is a descriptive evidence summary, not a dyslexia prediction or risk probability.",
            "Speech-to-text errors can be mistaken for reading errors; listen to the audio and verify the transcript.",
            "The handwriting CNN classes describe its training dataset and are not dyslexia labels.",
            "A single passage cannot diagnose or rule out dyslexia; age- and grade-appropriate validated assessment is needed."
        ]
    }
    if not reading_details:
        return summary

    summary["status"] = "Reading and handwriting observations available; interpretation requires review"
    mismatch_values = [
        reading_details.get("substitutions"),
        reading_details.get("deletions_or_skipped_words"),
        reading_details.get("insertions_or_repetitions")
    ]
    mismatch_count = sum(int(v or 0) for v in mismatch_values)
    summary["transcript_mismatch_count"] = mismatch_count
    if mismatch_count > 0:
        summary["review_note"] = (
            f"The speech transcript contains {mismatch_count} word-level mismatch(es). "
            "Review the audio and correct the transcript before interpreting this as a reading difficulty."
        )
    else:
        summary["review_note"] = (
            "No word-level mismatch was recorded in this passage. This single sample does not rule out dyslexia."
        )
    if handwriting_pattern:
        summary["handwriting_context"] = (
            f"The handwriting CNN predicted '{handwriting_pattern}'. Keep this as a separate descriptive observation; "
            "it is not counted as evidence that the child has dyslexia."
        )
    else:
        summary["handwriting_context"] = "No handwriting CNN result is currently available for this child."
    return summary


def build_hybrid_screening_summary(reading_accuracy=None, math_accuracy=None,
                                   handwriting_pattern=None,
                                   dyscalculia_model_label=None):
    """Build transparent prototype indicators; never present them as diagnosis."""
    domains = {}
    signals = []
    rules = []

    # These are initial engineering thresholds for demonstrating the pipeline,
    # NOT validated age/grade norms or clinically validated cut-offs.
    if reading_accuracy is not None:
        value = float(reading_accuracy)
        flag = value < 80.0
        domains["Reading performance"] = {
            "value": value,
            "unit": "% approximate transcript match",
            "indicator": flag,
            "note": "Below the provisional 80% prototype threshold" if flag
                    else "At or above the provisional 80% prototype threshold"
        }
        rules.append("Reading indicator: approximate transcript match < 80% (provisional prototype rule).")
        if flag:
            signals.append("Reading performance was below the provisional prototype threshold.")

    if math_accuracy is not None:
        value = float(math_accuracy)
        flag = value < 60.0
        domains["Mathematics performance"] = {
            "value": value,
            "unit": "% correct",
            "indicator": flag,
            "note": "Below the provisional 60% prototype threshold" if flag
                    else "At or above the provisional 60% prototype threshold"
        }
        rules.append("Mathematics indicator: quiz accuracy < 60% (provisional prototype rule).")
        if flag:
            signals.append("Mathematics performance was below the provisional prototype threshold.")

    # The handwriting CNN's classes describe its training-dataset categories;
    # they are displayed as observations and are not converted into a disability score.
    handwriting_note = None
    if handwriting_pattern:
        handwriting_note = (
            f"Existing handwriting CNN pattern: {handwriting_pattern}. "
            "This class is descriptive only and is not counted as dyslexia/dysgraphia evidence."
        )

    model_note = None
    if dyscalculia_model_label:
        model_note = (
            f"Existing dyscalculia model output: {dyscalculia_model_label}. "
            "Shown separately; it is not added to the transparent score until its labels and "
            "performance are verified on independent child-level data."
        )

    available_count = len(domains)
    signal_count = len(signals)
    prototype_score = (signal_count / available_count * 100.0) if available_count else None

    if available_count < 2:
        category = "Insufficient data"
        explanation = "Complete both reading and mathematics assessments to calculate the initial two-domain prototype summary."
    elif signal_count == 0:
        category = "No prototype flags on available tasks"
        explanation = "The current prototype rules did not flag the available reading or mathematics measures. This does not rule out a learning difficulty."
    elif signal_count == 1:
        category = "One indicator to review"
        explanation = "One available task crossed a provisional prototype threshold. Review the task result and context; this is not a diagnosis."
    else:
        category = "Multiple indicators to review"
        explanation = "Both available task measures crossed provisional prototype thresholds. Consider review by a qualified education/health professional."

    return {
        "category": category,
        "explanation": explanation,
        "available_domains": domains,
        "indicator_count": signal_count,
        "domain_count": available_count,
        "prototype_indicator_score_percent": prototype_score,
        "signals": signals,
        "rules": rules,
        "handwriting_note": handwriting_note,
        "dyscalculia_model_note": model_note,
        "dysgraphia_status": "Not yet scored: validated handwriting-quality features and suitable labeled data are still required."
    }

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

    import json

    from PIL import ImageOps

    # Cache the OCR model so it does not reload for every Streamlit rerun.
    @st.cache_resource
    def load_handwriting_ocr_model():
        from transformers import (
            TrOCRProcessor,
            VisionEncoderDecoderModel
        )

        model_name = "microsoft/trocr-base-handwritten"

        processor = TrOCRProcessor.from_pretrained(model_name)
        ocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)

        ocr_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        ocr_model.to(ocr_device)
        ocr_model.eval()

        return processor, ocr_model, ocr_device

    def recognize_handwriting_text(pil_image):
        """Recognize handwritten text using the TrOCR model."""

        processor, ocr_model, ocr_device = load_handwriting_ocr_model()

        # Improve contrast while retaining the original color image
        # for the existing handwriting CNN.
        prepared_image = ImageOps.autocontrast(
            pil_image.convert("L")
        ).convert("RGB")

        pixel_values = processor(
            images=prepared_image,
            return_tensors="pt"
        ).pixel_values.to(ocr_device)

        with torch.no_grad():
            generated_ids = ocr_model.generate(
                pixel_values,
                max_new_tokens=128
            )

        recognized_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        return recognized_text

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

                        image_tensor = handwriting_transform(image)
                        image_tensor = image_tensor.unsqueeze(0).to(device)

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

                        if confidence < MODEL_CONFIDENCE_MIN_PERCENT:
                            st.warning(
                                f"Low model confidence ({confidence:.1f}%). "
                                "Treat this scan as uncertain and review/repeat it; "
                                "this is not a diagnostic threshold."
                            )
                        else:
                            st.info(
                                f"Demo quality gate met: model confidence ≥ "
                                f"{MODEL_CONFIDENCE_MIN_PERCENT:.0f}%. This is not a diagnosis."
                            )

                        class_probabilities = {
                            classes[i]: (
                                float(probabilities[i].item()) * 100
                            )
                            for i in range(len(classes))
                        }

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

                        save_assessment_result(
                            child_id=child_id,
                            assessment_type="handwriting",
                            result=prediction,
                            confidence=confidence,
                            details=json.dumps(
                                {
                                    "class_probabilities": class_probabilities,
                                    "interpretation": handwriting_indicator
                                },
                                ensure_ascii=False
                            )
                        )

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
                            "Handwriting classification saved successfully."
                        )

            # --------------------------------------------------
            # HANDWRITING TEXT RECOGNITION (OCR)
            # --------------------------------------------------
            st.markdown("---")
            st.markdown("### 📝 Handwriting Text Recognition")

            st.write(
                "This AI feature attempts to read the actual words in "
                "the uploaded handwriting image. Recognition may be "
                "inaccurate for unclear, joined, or unusual handwriting."
            )

            if st.button(
                "🔎 Recognize Handwriting Text",
                type="primary",
                use_container_width=True,
                key="recognize_handwriting_ocr_button"
            ):

                try:
                    with st.spinner(
                        "Loading the handwriting OCR model and "
                        "recognizing the text..."
                    ):
                        recognized_text = recognize_handwriting_text(image)

                    st.session_state["handwriting_ocr_text"] = (
                        recognized_text
                    )
                    st.session_state["handwriting_ocr_child_id"] = child_id

                    if recognized_text:
                        st.success("Text recognition completed.")
                    else:
                        st.warning(
                            "The model did not recognize any text. "
                            "Try a clearer image with larger, darker writing."
                        )

                except ImportError:
                    st.error(
                        "OCR dependencies are missing. Activate your "
                        "project's virtual environment and run:"
                    )
                    st.code(
                        "python -m pip install transformers sentencepiece"
                    )

                except Exception as ocr_error:
                    st.error(
                        "Handwriting text recognition failed. Check your "
                        "internet connection and OCR model installation."
                    )
                    st.exception(ocr_error)

            # Display recognized text only for the selected child.
            if (
                st.session_state.get("handwriting_ocr_child_id")
                == child_id
            ):
                st.markdown("#### 📄 Recognized Text")

                original_ocr_text = st.session_state.get(
                    "handwriting_ocr_text", ""
                )

                corrected_ocr_text = st.text_area(
                    "Review and correct the recognized text if needed",
                    value=original_ocr_text,
                    height=150,
                    key="handwriting_ocr_edit"
                )

                if st.button(
                    "💾 Save Recognized Text",
                    use_container_width=True,
                    key="save_handwriting_ocr_button"
                ):
                    try:
                        ocr_details = {
                            "recognized_text": corrected_ocr_text,
                            "ocr_model": "microsoft/trocr-base-handwritten",
                            "manually_reviewed": (
                                corrected_ocr_text != original_ocr_text
                            ),
                            "note": (
                                "OCR output may contain errors and should "
                                "be checked against the uploaded image."
                            )
                        }

                        save_assessment_result(
                            child_id=child_id,
                            assessment_type="handwriting_ocr",
                            result=(
                                "Text recognized"
                                if corrected_ocr_text.strip()
                                else "No text recorded"
                            ),
                            confidence=None,
                            details=json.dumps(
                                ocr_details,
                                ensure_ascii=False
                            )
                        )

                        st.session_state[
                            "handwriting_ocr_text"
                        ] = corrected_ocr_text

                        st.success(
                            "Recognized text saved for this child."
                        )

                    except Exception as ocr_save_error:
                        st.error("Could not save the recognized text.")
                        st.exception(ocr_save_error)

                if corrected_ocr_text.strip():
                    recognized_words = corrected_ocr_text.split()

                    st.markdown("#### 📊 Text Summary")

                    t1, t2 = st.columns(2)

                    t1.metric(
                        "Recognized words",
                        len(recognized_words)
                    )

                    t2.metric(
                        "Recognized characters",
                        len(corrected_ocr_text)
                    )

                    st.caption(
                        "These counts describe the recognized text only. "
                        "They do not measure reading ability, intelligence, "
                        "or the presence of a learning disability."
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
                    "The handwriting classification is saved in SQLite."
                )

            # --------------------------------------------------
            # DYSGRAPHIA-RELATED WRITING IMAGE FEATURES
            # --------------------------------------------------
            st.markdown("---")
            st.markdown("### ✏️ Dysgraphia-Related Writing Analysis")

            st.info(
                "This experimental feature measures visual characteristics "
                "of the uploaded image. It is not a validated dysgraphia "
                "classifier and will not assign a diagnosis or risk level."
            )

            st.write(
                "For more meaningful measurements, use a clear, well-lit "
                "image containing one writing sample, with the page filling "
                "most of the frame."
            )

            if st.button(
                "✏️ Analyze for Dysgraphia-Related Features",
                type="secondary",
                use_container_width=True,
                key="analyze_dysgraphia_features"
            ):
                try:
                    gray_image = ImageOps.autocontrast(
                        image.convert("L")
                    )
                    gray_array = np.asarray(
                        gray_image, dtype=np.uint8
                    )

                    _, binary = cv2.threshold(
                        gray_array,
                        0,
                        255,
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )

                    kernel = np.ones((2, 2), np.uint8)

                    cleaned = cv2.morphologyEx(
                        binary,
                        cv2.MORPH_OPEN,
                        kernel
                    )

                    image_height, image_width = cleaned.shape
                    ink_mask = cleaned > 0
                    ink_density = float(np.mean(ink_mask))

                    ys, xs = np.where(ink_mask)

                    if len(xs) == 0 or len(ys) == 0:
                        st.warning(
                            "No clear handwriting or ink was detected. "
                            "Try a closer, brighter image with darker writing."
                        )

                    else:
                        x_min, x_max = int(xs.min()), int(xs.max())
                        y_min, y_max = int(ys.min()), int(ys.max())

                        writing_width = x_max - x_min + 1
                        writing_height = y_max - y_min + 1

                        writing_area_fraction = float(
                            (writing_width * writing_height)
                            / (image_width * image_height)
                        )

                        row_ink = np.sum(ink_mask, axis=1)

                        row_threshold = max(
                            2,
                            int(image_width * 0.003)
                        )

                        active_rows = row_ink >= row_threshold

                        transitions = np.diff(
                            np.concatenate(
                                (
                                    [False],
                                    active_rows,
                                    [False]
                                )
                            ).astype(np.int8)
                        )

                        line_band_count = int(
                            np.sum(transitions == 1)
                        )

                        component_count, labels, stats, centroids = (
                            cv2.connectedComponentsWithStats(
                                cleaned,
                                connectivity=8
                            )
                        )

                        min_component_area = max(
                            5,
                            int(
                                image_width
                                * image_height
                                * 0.00001
                            )
                        )

                        component_areas = [
                            int(stats[i, cv2.CC_STAT_AREA])
                            for i in range(1, component_count)
                            if int(stats[i, cv2.CC_STAT_AREA])
                            >= min_component_area
                        ]

                        visible_components = len(component_areas)

                        median_component_area = (
                            float(np.median(component_areas))
                            if component_areas
                            else 0.0
                        )

                        band_starts = np.where(
                            np.diff(
                                np.concatenate(
                                    (
                                        [0],
                                        active_rows.astype(int)
                                    )
                                )
                            ) == 1
                        )[0]

                        band_ends = np.where(
                            np.diff(
                                np.concatenate(
                                    (
                                        active_rows.astype(int),
                                        [0]
                                    )
                                )
                            ) == -1
                        )[0]

                        band_centers = [
                            (int(a) + int(b)) / 2
                            for a, b in zip(
                                band_starts,
                                band_ends
                            )
                        ]

                        line_gaps = [
                            band_centers[i + 1] - band_centers[i]
                            for i in range(len(band_centers) - 1)
                            if band_centers[i + 1] > band_centers[i]
                        ]

                        median_row_gap_px = (
                            float(np.median(line_gaps))
                            if line_gaps
                            else None
                        )

                        feature_details = {
                            "analysis_type": "experimental_image_features",
                            "ink_density_percent": round(
                                ink_density * 100, 3
                            ),
                            "writing_region_width_px": writing_width,
                            "writing_region_height_px": writing_height,
                            "writing_region_area_percent": round(
                                writing_area_fraction * 100, 3
                            ),
                            "horizontal_ink_bands": line_band_count,
                            "visible_connected_components": visible_components,
                            "median_component_area_px": round(
                                median_component_area, 2
                            ),
                            "median_row_gap_px": (
                                round(median_row_gap_px, 2)
                                if median_row_gap_px is not None
                                else None
                            ),
                            "interpretation": (
                                "Image features were measured for research "
                                "review. They are not validated markers of "
                                "dysgraphia and must not be interpreted as "
                                "a diagnosis or risk score."
                            )
                        }

                        st.markdown("#### 📊 Image Feature Measurements")

                        f1, f2, f3 = st.columns(3)

                        f1.metric(
                            "Ink coverage",
                            f"{ink_density * 100:.2f}%"
                        )

                        f2.metric(
                            "Horizontal ink bands",
                            str(line_band_count)
                        )

                        f3.metric(
                            "Visible components",
                            str(visible_components)
                        )

                        f4, f5, f6 = st.columns(3)

                        f4.metric(
                            "Writing region width",
                            f"{writing_width}px"
                        )

                        f5.metric(
                            "Writing region height",
                            f"{writing_height}px"
                        )

                        f6.metric(
                            "Median row gap",
                            (
                                f"{median_row_gap_px:.1f}px"
                                if median_row_gap_px is not None
                                else "Not available"
                            )
                        )

                        st.caption(
                            "These measurements depend on image resolution, "
                            "lighting, cropping, paper/background contrast, "
                            "and the amount of writing. They are not compared "
                            "against validated age- or grade-specific ranges."
                        )

                        st.warning(
                            "Dysgraphia screening status: Not determined. "
                            "No dysgraphia prediction or risk level is assigned "
                            "because this project does not yet have a separately "
                            "trained and independently evaluated dysgraphia model."
                        )

                        save_assessment_result(
                            child_id=child_id,
                            assessment_type="dysgraphia",
                            result=(
                                "Writing image features recorded; "
                                "dysgraphia status not determined"
                            ),
                            confidence=None,
                            details=json.dumps(
                                feature_details,
                                ensure_ascii=False
                            )
                        )

                        st.session_state[
                            "dysgraphia_features"
                        ] = feature_details

                        st.session_state[
                            "dysgraphia_features_child_id"
                        ] = child_id

                        st.success(
                            "Dysgraphia-related image feature report saved."
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

    # --------------------------------------------------
    # COMBINED READING + HANDWRITING EVIDENCE
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("### 📚 Combined Reading + Handwriting Evidence")

    st.caption(
        "This connects reading and handwriting assessment outputs for "
        "review. It does not calculate a validated dyslexia risk score."
    )

    reading_details_for_review = None

    if (
        st.session_state.get("reading_results_child_id") == child_id
        and isinstance(
            st.session_state.get("reading_results"),
            dict
        )
    ):
        reading_details_for_review = st.session_state.get(
            "reading_results"
        )

    else:
        saved_reading_for_review = get_latest_assessment_record(
            child_id,
            "reading"
        )

        if saved_reading_for_review:
            reading_details_for_review = parse_assessment_details(
                saved_reading_for_review.get("details")
            )

    handwriting_pattern_for_review = None

    if st.session_state.get("handwriting_child_id") == child_id:
        handwriting_pattern_for_review = st.session_state.get(
            "handwriting_prediction"
        )

    else:
        saved_handwriting_for_review = get_latest_assessment_record(
            child_id,
            "handwriting"
        )

        if saved_handwriting_for_review:
            handwriting_pattern_for_review = (
                saved_handwriting_for_review.get("result")
            )

    dyslexia_evidence_summary = build_dyslexia_evidence_summary(
        reading_details=reading_details_for_review,
        handwriting_pattern=handwriting_pattern_for_review
    )

    if not reading_details_for_review:
        st.warning(
            "Reading evidence is not available for this child yet. "
            "Complete the Reading Assessment, then return here to "
            "review the combined findings."
        )

    else:
        e1, e2, e3 = st.columns(3)

        reading_match_value = dyslexia_evidence_summary.get(
            "reading_accuracy_percent"
        )

        e1.metric(
            "Exact word matches",
            (
                f"{float(reading_match_value):.1f}%"
                if reading_match_value is not None
                else "Unavailable"
            )
        )

        error_rate_value = dyslexia_evidence_summary.get(
            "word_error_rate_percent"
        )

        e2.metric(
            "Word error rate",
            (
                f"{float(error_rate_value):.1f}%"
                if error_rate_value is not None
                else "Unavailable"
            )
        )

        speed_value = dyslexia_evidence_summary.get(
            "reading_speed_words_per_minute"
        )

        e3.metric(
            "Estimated reading speed",
            (
                f"{float(speed_value):.1f} words/min"
                if speed_value is not None
                else "Unavailable"
            )
        )

        st.info(
            dyslexia_evidence_summary["review_note"]
        )

        st.write(
            dyslexia_evidence_summary.get(
                "handwriting_context",
                "Handwriting pattern not available."
            )
        )

        # Show saved OCR text as additional context, if available.
        saved_ocr_text = None

        if (
            st.session_state.get("handwriting_ocr_child_id")
            == child_id
        ):
            saved_ocr_text = st.session_state.get(
                "handwriting_ocr_text"
            )

        else:
            saved_ocr_record = get_latest_assessment_record(
                child_id,
                "handwriting_ocr"
            )

            if saved_ocr_record:
                saved_ocr_details = parse_assessment_details(
                    saved_ocr_record.get("details")
                )

                if isinstance(saved_ocr_details, dict):
                    saved_ocr_text = saved_ocr_details.get(
                        "recognized_text"
                    )

        if saved_ocr_text:
            st.markdown("#### 📝 Recognized Handwriting Text")
            st.text_area(
                "Saved OCR text",
                value=saved_ocr_text,
                height=100,
                disabled=True,
                key="combined_handwriting_ocr_display"
            )

        st.warning(
            "Dyslexia-related screening conclusion: not determined by "
            "these outputs alone. Verify the transcript and use "
            "age- and grade-appropriate reading, spelling, and language "
            "assessment."
        )

        if st.button(
            "💾 Save Combined Dyslexia Evidence Summary",
            key="save_dyslexia_evidence_summary",
            use_container_width=True
        ):
            try:
                save_assessment_result(
                    child_id=child_id,
                    assessment_type="dyslexia_evidence",
                    result=dyslexia_evidence_summary["status"],
                    confidence=None,
                    details=json.dumps(
                        dyslexia_evidence_summary,
                        ensure_ascii=False
                    )
                )

                st.session_state[
                    "dyslexia_evidence_summary"
                ] = dyslexia_evidence_summary

                st.session_state[
                    "dyslexia_evidence_child_id"
                ] = child_id

                st.success(
                    "Combined reading and handwriting evidence summary saved."
                )

            except Exception as summary_save_error:
                st.error(
                    "Could not save the combined evidence summary."
                )
                st.exception(summary_save_error)

# ============================================================
#  READING ASSESSMENT
# ============================================================

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
                        temperature=0.0,
                        condition_on_previous_text=False,
                        # Keep VAD off for short child-reading samples so quiet
                        # words and long pauses are not accidentally discarded.
                        vad_filter=False
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
        if meets_minimum(results["exact_word_match_percent"], READING_WORD_MATCH_TARGET_PERCENT):
            st.info(f"Reading recording review gate met (word-match target at least {READING_WORD_MATCH_TARGET_PERCENT:.0f}%). Speech recognition can make errors; this is not a diagnosis.")
        else:
            st.warning(f"Word-match result is below the demo review target of {READING_WORD_MATCH_TARGET_PERCENT:.0f}%. Check the transcript/audio and repeat if needed.")

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
            if meets_minimum(accuracy * 100, TASK_ACCURACY_TARGET_PERCENT):
                st.caption(f"Demo task target met (at least {TASK_ACCURACY_TARGET_PERCENT:.0f}% accuracy); not a diagnostic cut-off.")
            else:
                st.caption(f"Below demo task target ({TASK_ACCURACY_TARGET_PERCENT:.0f}%); interpret in context and repeat if needed.")

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
            if meets_minimum(accuracy * 100, TASK_ACCURACY_TARGET_PERCENT):
                st.caption(f"Demo task target met (at least {TASK_ACCURACY_TARGET_PERCENT:.0f}% accuracy); not a diagnostic cut-off.")
            else:
                st.caption(f"Below demo task target ({TASK_ACCURACY_TARGET_PERCENT:.0f}%); interpret in context and repeat if needed.")

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
            if meets_minimum(accuracy * 100, TASK_ACCURACY_TARGET_PERCENT):
                st.caption(f"Demo task target met (at least {TASK_ACCURACY_TARGET_PERCENT:.0f}% accuracy); not a diagnostic cut-off.")
            else:
                st.caption(f"Below demo task target ({TASK_ACCURACY_TARGET_PERCENT:.0f}%); interpret in context and repeat if needed.")

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
            if meets_minimum(accuracy * 100, TASK_ACCURACY_TARGET_PERCENT):
                st.caption(f"Demo task target met (at least {TASK_ACCURACY_TARGET_PERCENT:.0f}% accuracy); not a diagnostic cut-off.")
            else:
                st.caption(f"Below demo task target ({TASK_ACCURACY_TARGET_PERCENT:.0f}%); interpret in context and repeat if needed.")

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
                        
                        if float(predicted_class_probability) < MODEL_CONFIDENCE_MIN_PERCENT:
                            st.warning(
                                f"Low model confidence ({float(predicted_class_probability):.1f}%). "
                                "Flag for human review; do not treat this as a diagnosis."
                            )
                        else:
                            st.info(
                                f"Demo confidence gate met (at least {MODEL_CONFIDENCE_MIN_PERCENT:.0f}%). "
                                "This remains an experimental model output."
                            )

                        if float(st.session_state["dyscalculia_confidence"]) < MODEL_CONFIDENCE_MIN_PERCENT:
                            st.warning(
                                f"Low prediction confidence ({float(st.session_state['dyscalculia_confidence']):.1f}%). "
                                "Flag for human review; do not use as a diagnosis."
                            )
                        else:
                            st.info(
                                f"Demo confidence gate met (≥ {MODEL_CONFIDENCE_MIN_PERCENT:.0f}%). "
                                "The model output remains experimental and is not a diagnosis."
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

    import json

    hero(
        "Mathematics Assessment",
        "A short grade-based assessment measuring accuracy and response speed."
    )

    if "child_id" not in st.session_state:

        st.warning(
            "Please complete the Child Profile before starting."
        )

    else:

        child_id = st.session_state["child_id"]
        child_grade = st.session_state["child_grade"]

        # --------------------------------------------------
        # CHILD INFORMATION
        # --------------------------------------------------
        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card("Child ID", str(child_id))

        with c2:
            metric_card(
                "Age",
                f"{st.session_state.get('child_age', 'N/A')} years"
            )

        with c3:
            metric_card("Grade", str(child_grade))

        # --------------------------------------------------
        # GRADE-BASED QUESTIONS
        # --------------------------------------------------
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

        questions = grade_questions.get(child_grade)

        if not questions:
            st.error(
                f"No mathematics questions are configured for {child_grade}."
            )
            st.stop()

        # --------------------------------------------------
        # RESET ASSESSMENT STATE IF THE SELECTED CHILD CHANGES
        # --------------------------------------------------
        if st.session_state.get("math_active_child_id") != child_id:

            st.session_state["math_active_child_id"] = child_id
            st.session_state["math_question_number"] = 0
            st.session_state["math_correct"] = 0
            st.session_state["math_response_times"] = []
            st.session_state["math_assessment_saved_key"] = None
            st.session_state.pop("math_timer_question", None)
            st.session_state.pop("math_question_start", None)

        # Initialize state
        if "math_question_number" not in st.session_state:
            st.session_state["math_question_number"] = 0

        if "math_correct" not in st.session_state:
            st.session_state["math_correct"] = 0

        if "math_response_times" not in st.session_state:
            st.session_state["math_response_times"] = []

        # Unique identifier for this assessment attempt.
        # It prevents the same completed attempt being saved repeatedly.
        if "math_assessment_attempt_key" not in st.session_state:
            st.session_state["math_assessment_attempt_key"] = (
                f"{child_id}_{time.time_ns()}"
            )

        question_number = st.session_state["math_question_number"]

        # --------------------------------------------------
        # START TIMER FOR CURRENT QUESTION
        # --------------------------------------------------
        if (
            "math_timer_question" not in st.session_state
            or st.session_state["math_timer_question"] != question_number
        ):
            st.session_state["math_question_start"] = time.perf_counter()
            st.session_state["math_timer_question"] = question_number

        # --------------------------------------------------
        # ASSESSMENT COMPLETED
        # --------------------------------------------------
        if question_number >= len(questions):

            st.success("Assessment completed.")

            correct = st.session_state["math_correct"]
            total = len(questions)
            accuracy = (correct / total) * 100

            response_times = st.session_state["math_response_times"]

            average_response_time = (
                sum(response_times) / len(response_times)
                if response_times else 0.0
            )

            # Preserve results in session state for the current session.
            st.session_state["math_accuracy"] = accuracy
            st.session_state["math_avg_response_time"] = average_response_time
            st.session_state["math_correct_answers"] = correct
            st.session_state["math_total_questions"] = total
            st.session_state["math_child_id"] = child_id

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

            st.progress(min(accuracy / 100, 1.0))

            if accuracy >= 80:
                st.success("Strong performance on this assessment.")
            elif accuracy >= 60:
                st.warning(
                    "Some difficulty was observed on this assessment."
                )
            else:
                st.warning(
                    "Substantial difficulty was observed on this assessment."
                )

            # --------------------------------------------------
            # SAVE RESULTS TO SQLITE DATABASE
            # --------------------------------------------------
            attempt_key = st.session_state["math_assessment_attempt_key"]
            saved_key = st.session_state.get("math_assessment_saved_key")

            if saved_key != attempt_key:

                result_text = (
                    f"Mathematics assessment completed: "
                    f"{correct}/{total} correct ({accuracy:.1f}% accuracy)"
                )

                assessment_details = {
                    "child_id": str(child_id),
                    "grade": str(child_grade),
                    "accuracy": float(accuracy),
                    "math_accuracy": float(accuracy),
                    "correct_answers": int(correct),
                    "math_correct_answers": int(correct),
                    "total_questions": int(total),
                    "math_total_questions": int(total),
                    "average_response_time": float(average_response_time),
                    "avg_response_time": float(average_response_time),
                    "math_avg_response_time": float(average_response_time),
                    "response_times": [
                        float(value) for value in response_times
                    ],
                    "completed": True,
                    "assessment_attempt_key": str(attempt_key)
                }

                try:
                    save_assessment_result(
                        child_id=child_id,
                        assessment_type="mathematics",
                        result=result_text,
                        confidence=None,
                        details=json.dumps(assessment_details)
                    )

                    # Mark saved only after the save function returns successfully.
                    st.session_state["math_assessment_saved_key"] = attempt_key

                    st.success(
                        "Mathematics results saved successfully. "
                        "They are now available for Overall Risk Analysis "
                        "and the Final Report."
                    )

                except Exception as exc:
                    st.error(
                        "The assessment is complete, but saving to the database "
                        "failed. Please retry the save before leaving this page."
                    )
                    st.caption(f"Technical detail: {exc}")

                    if st.button(
                        "Retry saving mathematics results",
                        use_container_width=True,
                        key="retry_save_math_results"
                    ):
                        st.rerun()

            else:
                st.info(
                    "This assessment attempt has already been saved."
                )

            # --------------------------------------------------
            # RESTART ASSESSMENT
            # --------------------------------------------------
            if st.button(
                "Restart assessment",
                use_container_width=True,
                key="restart_math_assessment"
            ):

                st.session_state["math_question_number"] = 0
                st.session_state["math_correct"] = 0
                st.session_state["math_response_times"] = []

                # A new attempt will be saved as a new assessment record.
                st.session_state["math_assessment_attempt_key"] = (
                    f"{child_id}_{time.time_ns()}"
                )

                st.session_state["math_assessment_saved_key"] = None

                st.session_state.pop("math_timer_question", None)
                st.session_state.pop("math_question_start", None)

                st.rerun()

        # --------------------------------------------------
        # DISPLAY CURRENT QUESTION
        # --------------------------------------------------
        else:

            question, options, correct_answer = questions[question_number]

            st.progress(question_number / len(questions))

            st.caption(
                f"Question {question_number + 1} of {len(questions)}"
            )

            st.markdown(f"### {question}")

            answer = st.radio(
                "Choose an answer",
                options,
                key=f"math_answer_{child_id}_{question_number}"
            )

            if st.button(
                "Submit & continue",
                type="primary",
                use_container_width=True,
                key=f"submit_math_{child_id}_{question_number}"
            ):

                response_time = (
                    time.perf_counter()
                    - st.session_state["math_question_start"]
                )

                st.session_state["math_response_times"].append(
                    float(response_time)
                )

                if answer == correct_answer:
                    st.session_state["math_correct"] += 1

                st.session_state["math_question_number"] += 1

                st.rerun()

# ============================================================
# EYE TRACKING
# ============================================================





elif page == "👁️ Eye Tracking Assessment":

    import json
    import os
    import time

    import cv2
    import mediapipe as mp
    import numpy as np
    import pandas as pd
    import streamlit as st

    hero(
        "Eye Tracking Assessment",
        "Experimental webcam-based gaze calibration and visual tracking."
    )

    st.info(
        "Research prototype only. Webcam gaze estimates are approximate "
        "and must not be used alone to diagnose a learning disability."
    )

    child_id = st.session_state.get("child_id")

    if child_id:
        st.success(f"Selected child: {child_id}")
    else:
        st.warning(
            "Select a child in Child Profile before running this assessment."
        )

    st.write(
        "Keep your face centered in good lighting. Look directly at each "
        "calibration target, then follow the moving target with your eyes "
        "while keeping your head relatively still."
    )

    st.warning(
        "This task measures experimental gaze-following performance. "
        "It does not measure eye movements during reading."
    )

    # =========================================================
    # START ASSESSMENT
    # =========================================================
    if st.button(
        "Start Calibration & Tracking Task",
        type="primary",
        key="start_eye_tracking_task"
    ):

        if not child_id:
            st.error(
                "Please select a child in Child Profile first."
            )

        else:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
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
                        "Could not open the webcam. Check camera "
                        "permissions and close other applications "
                        "using the camera."
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

                    # A mutable list avoids the invalid nonlocal statement.
                    last_timestamp_ms = [-1]

                    last_preview_time = 0.0
                    last_target_update = 0.0
                    task_error = None

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

                    # -------------------------------------------------
                    # DISPLAY TARGET
                    # -------------------------------------------------
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

                    # -------------------------------------------------
                    # EXTRACT IRIS FEATURES
                    # -------------------------------------------------
                    def get_iris_features(landmarks):
                        if len(landmarks) < 478:
                            return None

                        features = []

                        eye_definitions = [
                            {
                                "iris": 468,
                                "corners": (362, 263),
                                "bounds": (
                                    362, 385, 387, 263, 373, 380
                                ),
                            },
                            {
                                "iris": 473,
                                "corners": (33, 133),
                                "bounds": (
                                    33, 160, 158, 133, 153, 144
                                ),
                            },
                        ]

                        for eye in eye_definitions:
                            iris = landmarks[eye["iris"]]

                            corner_a = landmarks[eye["corners"][0]]
                            corner_b = landmarks[eye["corners"][1]]

                            eye_points = [
                                landmarks[index]
                                for index in eye["bounds"]
                            ]

                            min_y = min(point.y for point in eye_points)
                            max_y = max(point.y for point in eye_points)

                            eye_width = abs(corner_b.x - corner_a.x)
                            eye_height = max_y - min_y

                            if eye_width < 0.005 or eye_height < 0.003:
                                return None

                            normalized_x = (
                                iris.x
                                - min(corner_a.x, corner_b.x)
                            ) / eye_width

                            normalized_y = (
                                iris.y - min_y
                            ) / eye_height

                            features.extend([
                                normalized_x,
                                normalized_y
                            ])

                        features = np.asarray(features, dtype=float)

                        if not np.all(np.isfinite(features)):
                            return None

                        return features

                    # -------------------------------------------------
                    # DRAW EYE LANDMARKS
                    # -------------------------------------------------
                    def draw_eye_landmarks(frame, landmarks):
                        eye_indices = [
                            362, 385, 387, 263, 373, 380,
                            33, 160, 158, 133, 153, 144
                        ]

                        height, width = frame.shape[:2]

                        for index in eye_indices:
                            point = landmarks[index]

                            cv2.circle(
                                frame,
                                (
                                    int(point.x * width),
                                    int(point.y * height)
                                ),
                                3,
                                (255, 100, 0),
                                -1
                            )

                        if len(landmarks) >= 478:
                            for index in (468, 473):
                                point = landmarks[index]

                                cv2.circle(
                                    frame,
                                    (
                                        int(point.x * width),
                                        int(point.y * height)
                                    ),
                                    6,
                                    (0, 255, 255),
                                    2
                                )

                    # -------------------------------------------------
                    # PROCESS CAMERA FRAMES
                    # -------------------------------------------------
                    def process_frame(frame):
                        frame = cv2.flip(frame, 1)

                        rgb_frame = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2RGB
                        )

                        mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=rgb_frame
                        )

                        timestamp_ms = int(time.time() * 1000)

                        timestamp_ms = max(
                            timestamp_ms,
                            last_timestamp_ms[0] + 1
                        )

                        last_timestamp_ms[0] = timestamp_ms

                        result = face_landmarker.detect_for_video(
                            mp_image,
                            timestamp_ms
                        )

                        landmarks = None
                        features = None

                        if result.face_landmarks:
                            landmarks = result.face_landmarks[0]

                            features = get_iris_features(landmarks)

                            draw_eye_landmarks(frame, landmarks)

                        return frame, landmarks, features

                    # =================================================
                    # RUN CALIBRATION AND TRACKING
                    # =================================================
                    try:
                        options = mp.tasks.vision.FaceLandmarkerOptions(
                            base_options=mp.tasks.BaseOptions(
                                model_asset_path=model_path
                            ),
                            running_mode=(
                                mp.tasks.vision.RunningMode.VIDEO
                            ),
                            num_faces=1
                        )

                        face_landmarker = (
                            mp.tasks.vision.FaceLandmarker
                            .create_from_options(options)
                        )

                        # =============================================
                        # STAGE 1: CALIBRATION
                        # =============================================
                        st.subheader("Stage 1: Gaze Calibration")

                        st.caption(
                            "Look at each red target. Each point is "
                            "sampled for approximately two seconds."
                        )

                        for point_number, (
                            target_x,
                            target_y
                        ) in enumerate(calibration_points, start=1):

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
                                        "The webcam stopped returning "
                                        "frames during calibration."
                                    )

                                frames_seen += 1

                                frame, landmarks, features = (
                                    process_frame(frame)
                                )

                                if landmarks is not None:
                                    frames_with_face += 1

                                if features is not None:
                                    calibration_features.append(features)

                                    calibration_targets.append([
                                        target_x,
                                        target_y
                                    ])

                                    collected_here += 1

                                now = time.time()

                                if now - last_preview_time >= 0.12:
                                    preview.image(
                                        cv2.cvtColor(
                                            frame,
                                            cv2.COLOR_BGR2RGB
                                        ),
                                        channels="RGB",
                                        use_container_width=True
                                    )

                                    last_preview_time = now

                                status.caption(
                                    f"Calibration point "
                                    f"{point_number}/5 · "
                                    f"Valid samples: {collected_here}"
                                )

                        # =============================================
                        # FIT CALIBRATION MAPPING
                        # =============================================
                        if len(calibration_features) < 10:
                            raise RuntimeError(
                                "Too few valid iris samples were "
                                "collected. Improve lighting, keep "
                                "your face centered, and try again."
                            )

                        X = np.asarray(
                            calibration_features,
                            dtype=float
                        )

                        Y = np.asarray(
                            calibration_targets,
                            dtype=float
                        )

                        X_design = np.column_stack([
                            X,
                            np.ones(len(X))
                        ])

                        coefficients, _, _, _ = np.linalg.lstsq(
                            X_design,
                            Y,
                            rcond=None
                        )

                        st.success(
                            f"Calibration samples collected: {len(X)}"
                        )

                        # =============================================
                        # STAGE 2: MOVING-TARGET TRACKING
                        # =============================================
                        st.subheader("Stage 2: Moving-Target Tracking")

                        st.caption(
                            "Follow the red target with your eyes. "
                            "This task lasts approximately ten seconds."
                        )

                        task_start = time.time()
                        task_duration = 10.0
                        segment_length = 1.25

                        while time.time() - task_start < task_duration:
                            success, frame = camera.read()

                            if not success:
                                raise RuntimeError(
                                    "The webcam stopped returning "
                                    "frames during tracking."
                                )

                            frames_seen += 1

                            elapsed = time.time() - task_start

                            segment = int(elapsed / segment_length)

                            fraction = (
                                (elapsed % segment_length)
                                / segment_length
                            )

                            start_point = moving_points[
                                min(segment, len(moving_points) - 2)
                            ]

                            end_point = moving_points[
                                min(segment + 1, len(moving_points) - 1)
                            ]

                            target_x = (
                                start_point[0]
                                + fraction * (
                                    end_point[0] - start_point[0]
                                )
                            )

                            target_y = (
                                start_point[1]
                                + fraction * (
                                    end_point[1] - start_point[1]
                                )
                            )

                            now = time.time()

                            if now - last_target_update >= 0.12:
                                show_target(
                                    target_x,
                                    target_y,
                                    "Follow the moving target",
                                    f"Task time: {elapsed:.1f}/"
                                    f"{task_duration:.0f}s"
                                )

                                last_target_update = now

                            frame, landmarks, features = process_frame(
                                frame
                            )

                            if landmarks is not None:
                                frames_with_face += 1

                            if features is not None:
                                gaze_estimate = (
                                    np.append(features, 1.0)
                                    @ coefficients
                                )

                                estimated_x = float(
                                    np.clip(
                                        gaze_estimate[0],
                                        0,
                                        100
                                    )
                                )

                                estimated_y = float(
                                    np.clip(
                                        gaze_estimate[1],
                                        0,
                                        100
                                    )
                                )

                                error = float(
                                    np.sqrt(
                                        (estimated_x - target_x) ** 2
                                        + (estimated_y - target_y) ** 2
                                    )
                                )

                                tracking_errors.append(error)

                                tracking_samples.append({
                                    "Elapsed time (seconds)": round(
                                        elapsed, 2
                                    ),
                                    "Estimated gaze X (%)": round(
                                        estimated_x, 1
                                    ),
                                    "Estimated gaze Y (%)": round(
                                        estimated_y, 1
                                    ),
                                    "Target X (%)": round(
                                        target_x, 1
                                    ),
                                    "Target Y (%)": round(
                                        target_y, 1
                                    ),
                                    "Estimated error (percentage points)": (
                                        round(error, 1)
                                    )
                                })

                            if now - last_preview_time >= 0.12:
                                preview.image(
                                    cv2.cvtColor(
                                        frame,
                                        cv2.COLOR_BGR2RGB
                                    ),
                                    channels="RGB",
                                    use_container_width=True
                                )

                                last_preview_time = now

                            status.caption(
                                f"Tracking task: {elapsed:.1f}/"
                                f"{task_duration:.0f} seconds · "
                                f"Valid gaze samples: "
                                f"{len(tracking_errors)}"
                            )

                    except Exception as error:
                        task_error = (
                            f"{type(error).__name__}: {error}"
                        )

                    finally:
                        camera.release()

                        if face_landmarker is not None:
                            face_landmarker.close()

                    # =================================================
                    # DISPLAY AND SAVE RESULTS
                    # =================================================
                    if task_error:
                        st.error(
                            f"Eye tracking could not finish: {task_error}"
                        )

                    elif tracking_errors:
                        detection_rate = (
                            frames_with_face / frames_seen * 100
                            if frames_seen
                            else 0.0
                        )

                        mean_error = float(np.mean(tracking_errors))
                        eye_quality_ok, eye_quality_issues = eye_tracking_quality(
                            detection_rate, len(tracking_errors), mean_error
                        )
                        median_error = float(np.median(tracking_errors))

                        summary = {
                            "Child ID": child_id,
                            "Assessment": (
                                "Experimental moving-target gaze tracking"
                            ),
                            "Calibration samples": len(
                                calibration_features
                            ),
                            "Valid gaze samples": len(tracking_errors),
                            "Frames captured": frames_seen,
                            "Frames with face detected": frames_with_face,
                            "Face detection rate (%)": round(
                                detection_rate, 1
                            ),
                            "Mean estimated target error "
                            "(percentage points)": round(mean_error, 2),
                            "Median estimated target error "
                            "(percentage points)": round(median_error, 2),
                            "Reading task performed": False,
                            "Recording quality gate passed": eye_quality_ok,
                            "Recording quality issues": eye_quality_issues,
                            "Interpretation": (
                                "Experimental gaze-following measurements "
                                "only. Not a validated dyslexia screening "
                                "score."
                            )
                        }

                        st.session_state["eye_tracking_result"] = (
                            "Experimental gaze-tracking task completed"
                        )

                        st.session_state["eye_tracking_summary"] = summary
                        st.session_state["eye_tracking_samples"] = (
                            tracking_samples
                        )
                        st.session_state["eye_tracking_child_id"] = child_id

                        try:
                            save_assessment_result(
                                child_id=child_id,
                                assessment_type="eye_tracking",
                                result=(
                                    "Experimental gaze-tracking "
                                    "task completed"
                                ),
                                confidence=None,
                                details=json.dumps(summary)
                            )

                            if eye_quality_ok:
                                st.success("Eye-tracking recording quality gate met. This is not a disability score.")
                            else:
                                st.warning("Eye-tracking recording quality gate not met: " + " ".join(eye_quality_issues))

                            st.success(
                                f"Eye-tracking results saved for "
                                f"child {child_id}."
                            )

                        except Exception as save_error:
                            st.error(
                                "The task completed, but saving to the "
                                "database failed: "
                                f"{type(save_error).__name__}: "
                                f"{save_error}"
                            )

                            st.warning(
                                "The results remain in this session, "
                                "but may not appear in reports that "
                                "read only from the database."
                            )

                        st.subheader("Assessment Results")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            metric_card(
                                "Face detection rate",
                                f"{detection_rate:.1f}%"
                            )

                        with col2:
                            metric_card(
                                "Valid gaze samples",
                                str(len(tracking_errors))
                            )

                        with col3:
                            metric_card(
                                "Mean estimated error",
                                f"{mean_error:.1f} pp"
                            )

                        st.caption(
                            "Error is measured in percentage points of "
                            "the target panel, not degrees of visual "
                            "angle. Lower error means the estimated gaze "
                            "was closer to the moving target in this "
                            "experimental task. It is not a clinical score."
                        )

                        with st.expander("View assessment measurements"):
                            st.json(summary)

                        with st.expander("View sample gaze measurements"):
                            st.dataframe(
                                pd.DataFrame(tracking_samples),
                                use_container_width=True
                            )

                    else:
                        st.warning(
                            "The task ended without valid gaze samples. "
                            "Check lighting, face position, and iris "
                            "landmark availability before trying again."
                        )

    # =========================================================
    # SHOW PREVIOUS SESSION RESULT AFTER STREAMLIT RERUN
    # =========================================================
    previous_summary = st.session_state.get(
        "eye_tracking_summary"
    )

    previous_child_id = st.session_state.get(
        "eye_tracking_child_id"
    )

    if (
        previous_summary
        and child_id
        and previous_child_id == child_id
    ):
        st.divider()
        st.subheader("Latest Session Result")

        st.caption(
            "This result belongs to the currently selected child."
        )

        st.json(previous_summary)



# ============================================================
# OVERALL ANALYSIS
# ============================================================




elif page == "📊 Overall Risk Analysis":

    import json
    import pandas as pd
    import streamlit as st

    hero(
        "Overall Screening Analysis",
        "Combined handwriting, reading, and eye-tracking evidence "
        "for preliminary dyslexia screening."
    )

    child_id = st.session_state.get("child_id")

    # =========================================================
    # LOAD THE SELECTED CHILD'S LATEST SAVED ASSESSMENTS
    # =========================================================
    saved_results = {}

    if not child_id:
        st.warning(
            "Please select a child in Child Profile before viewing "
            "the overall screening analysis."
        )
    else:
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

                if not assessment_type:
                    continue

                # Keep only the newest record for each type.
                if assessment_type in saved_results:
                    continue

                record = {
                    "result": row["result"],
                    "confidence": row["confidence"],
                    "details": row["details"],
                    "created_at": row["created_at"]
                }

                raw_json = row["result_json"]

                if raw_json:
                    try:
                        parsed = json.loads(raw_json)

                        if isinstance(parsed, dict):
                            for field in (
                                "result",
                                "confidence",
                                "details"
                            ):
                                if parsed.get(field) is not None:
                                    record[field] = parsed[field]

                    except (json.JSONDecodeError, TypeError):
                        pass

                saved_results[assessment_type] = record

        except Exception as error:
            st.error("Could not load saved assessment results.")
            st.exception(error)

    # =========================================================
    # HELPER FUNCTIONS
    # =========================================================
    def find_saved_result(*names):
        for name in names:
            key = str(name).strip().lower().replace(" ", "_")

            if key in saved_results:
                return saved_results[key]

        return None

    def get_details_dict(record):
        if not record:
            return {}

        details = record.get("details")

        if isinstance(details, dict):
            return details

        if isinstance(details, str):
            try:
                parsed = json.loads(details)

                if isinstance(parsed, dict):
                    return parsed

            except (json.JSONDecodeError, TypeError):
                pass

        return {}

    def safe_float(value):
        try:
            number = float(value)

            if pd.isna(number):
                return None

            return number

        except (TypeError, ValueError):
            return None

    def first_numeric(details, *keys):
        for key in keys:
            value = safe_float(details.get(key))

            if value is not None:
                return value

        return None

    def current_child_value(value_key, child_key):
        if not child_id:
            return None

        if st.session_state.get(child_key) != child_id:
            return None

        return st.session_state.get(value_key)

    def get_record_and_details(
        record_names,
        session_key=None,
        child_key=None
    ):
        record = find_saved_result(*record_names)

        if record:
            return record, get_details_dict(record)

        if session_key and child_key:
            session_value = current_child_value(
                session_key,
                child_key
            )

            if isinstance(session_value, dict):
                return None, session_value

        return None, {}

    def display_saved_date(record):
        if record and record.get("created_at"):
            st.caption(f"Saved on: {record['created_at']}")

    # =========================================================
    # CHILD PROFILE
    # =========================================================
    if child_id:
        st.subheader("Child Profile")

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card("Child ID", str(child_id))

        with c2:
            metric_card(
                "Age",
                f"{st.session_state.get('child_age', 'N/A')} years"
            )

        with c3:
            metric_card(
                "Grade",
                str(st.session_state.get('child_grade', 'N/A'))
            )

    # =========================================================
    # RETRIEVE HANDWRITING EVIDENCE
    # =========================================================
    handwriting_record = find_saved_result("handwriting")
    handwriting_details = get_details_dict(handwriting_record)

    handwriting_prediction = (
        handwriting_record.get("result")
        if handwriting_record
        else current_child_value(
            "handwriting_prediction",
            "handwriting_child_id"
        )
    )

    handwriting_confidence = (
        handwriting_record.get("confidence")
        if handwriting_record
        else current_child_value(
            "handwriting_confidence",
            "handwriting_child_id"
        )
    )

    handwriting_class = str(
        handwriting_prediction or ""
    ).strip().lower()

    # Only the model's dataset class is used here.
    # It is not interpreted as a clinical diagnosis.
    handwriting_is_normal = (
        handwriting_class == "normal"
        or handwriting_class.endswith(" normal")
    )

    handwriting_is_non_normal = (
        handwriting_prediction is not None
        and bool(handwriting_class)
        and not handwriting_is_normal
    )

    # =========================================================
    # RETRIEVE READING EVIDENCE
    # =========================================================
    reading_record, reading_details = get_record_and_details(
        ("reading", "reading_assessment"),
        "reading_results",
        "reading_results_child_id"
    )

    exact_match = first_numeric(
        reading_details,
        "exact_word_match_percent",
        "exact_word_match",
        "word_match_accuracy",
        "accuracy_percent"
    )

    word_error_rate = first_numeric(
        reading_details,
        "word_error_rate_percent",
        "word_error_rate"
    )

    reading_speed = first_numeric(
        reading_details,
        "transcribed_words_per_minute",
        "reading_speed_wpm",
        "words_per_minute"
    )

    # =========================================================
    # RETRIEVE EYE-TRACKING EVIDENCE
    # =========================================================
    eye_record, eye_details = get_record_and_details(
        ("eye_tracking", "eye_tracking_assessment"),
        "eye_tracking_summary",
        "eye_tracking_child_id"
    )

    # Some versions may save a nested summary dictionary.
    if isinstance(eye_details.get("summary"), dict):
        eye_details = eye_details["summary"]

    mean_error = first_numeric(
        eye_details,
        "Mean estimated target error (percentage points)",
        "mean_target_error",
        "mean_error"
    )

    median_error = first_numeric(
        eye_details,
        "Median estimated target error (percentage points)",
        "median_target_error",
        "median_error"
    )

    valid_samples = eye_details.get("Valid gaze samples")

    face_detection = first_numeric(
        eye_details,
        "Face detection rate (%)",
        "face_detection_rate"
    )

    # =========================================================
    # RETRIEVE OTHER ASSESSMENTS FOR DISPLAY
    # =========================================================
    ocr_record = find_saved_result("handwriting_ocr")
    ocr_details = get_details_dict(ocr_record)

    recognized_text = ocr_details.get("recognized_text", "")

    if not recognized_text:
        recognized_text = (
            current_child_value(
                "handwriting_ocr_text",
                "handwriting_ocr_child_id"
            ) or ""
        )

    dysgraphia_record = find_saved_result("dysgraphia")
    dysgraphia_details = get_details_dict(dysgraphia_record)

    if not dysgraphia_details:
        session_features = current_child_value(
            "dysgraphia_features",
            "dysgraphia_features_child_id"
        )

        if isinstance(session_features, dict):
            dysgraphia_details = session_features

    dyscalculia_record = find_saved_result("dyscalculia")

    dyscalculia_prediction = (
        dyscalculia_record.get("result")
        if dyscalculia_record
        else current_child_value(
            "dyscalculia_prediction",
            "dyscalculia_child_id"
        )
    )

    dyscalculia_confidence = (
        dyscalculia_record.get("confidence")
        if dyscalculia_record
        else current_child_value(
            "dyscalculia_confidence",
            "dyscalculia_child_id"
        )
    )

    math_record = find_saved_result(
        "mathematics",
        "math",
        "math_assessment"
    )

    math_details = get_details_dict(math_record)

    if not math_details:
        math_accuracy_session = current_child_value(
            "math_accuracy",
            "math_child_id"
        )

        if math_accuracy_session is not None:
            math_details = {
                "accuracy_percent": math_accuracy_session,
                "correct_answers": st.session_state.get(
                    "math_correct_answers"
                ),
                "total_questions": st.session_state.get(
                    "math_total_questions"
                ),
                "average_response_time": st.session_state.get(
                    "math_avg_response_time"
                )
            }

    math_accuracy = first_numeric(
        math_details,
        "accuracy_percent",
        "math_accuracy"
    )

    correct_answers = math_details.get(
        "correct_answers",
        math_details.get("math_correct_answers")
    )

    total_questions = math_details.get(
        "total_questions",
        math_details.get("math_total_questions")
    )

    response_time = first_numeric(
        math_details,
        "average_response_time",
        "math_avg_response_time"
    )

    # =========================================================
    # ASSESSMENT AVAILABILITY
    # =========================================================
    st.divider()
    st.subheader("Assessment Completion Overview")

    availability = {
        "Handwriting classification": (
            handwriting_prediction is not None
        ),
        "Handwriting OCR": bool(str(recognized_text).strip()),
        "Reading assessment": exact_match is not None,
        "Eye-tracking assessment": bool(eye_details),
        "Dyscalculia assessment": (
            dyscalculia_prediction is not None
        ),
        "Mathematics assessment": math_accuracy is not None,
        "Dysgraphia image features": bool(dysgraphia_details)
    }

    done_count = sum(availability.values())
    total_count = len(availability)

    st.progress(
        done_count / total_count if total_count else 0
    )

    st.write(
        f"**{done_count} of {total_count} assessment items available**"
    )

    st.dataframe(
        pd.DataFrame([
            {
                "Assessment": name,
                "Status": "Available" if available else "Not available"
            }
            for name, available in availability.items()
        ]),
        use_container_width=True,
        hide_index=True
    )

    # =========================================================
    # INDIVIDUAL EVIDENCE PANELS
    # =========================================================
    st.divider()
    st.subheader("✍️ Handwriting Evidence")

    if handwriting_prediction is not None:
        h1, h2 = st.columns(2)

        with h1:
            metric_card(
                "Predicted dataset class",
                str(handwriting_prediction)
            )

        with h2:
            confidence_value = safe_float(handwriting_confidence)

            metric_card(
                "Model confidence",
                f"{confidence_value:.1f}%"
                if confidence_value is not None
                else "Unavailable"
            )

        interpretation = handwriting_details.get("interpretation")

        if interpretation:
            st.info(str(interpretation))

        display_saved_date(handwriting_record)

        st.caption(
            "The model class is treated only as an experimental "
            "handwriting observation. It is not a dyslexia probability."
        )
    else:
        st.info("Complete the handwriting screening to add this evidence.")

    st.divider()
    st.subheader("📝 Recognized Handwriting Text")

    if str(recognized_text).strip():
        st.text_area(
            "OCR recognized text",
            value=str(recognized_text),
            height=100,
            disabled=True,
            key="overall_recognized_handwriting_text"
        )

        st.caption(
            "Verify OCR output against the original handwriting image."
        )

        display_saved_date(ocr_record)
    else:
        st.info("No saved handwriting OCR text is available.")

    st.divider()
    st.subheader("📖 Reading Evidence")

    if exact_match is not None or word_error_rate is not None:
        r1, r2, r3 = st.columns(3)

        with r1:
            metric_card(
                "Exact word matches",
                f"{exact_match:.1f}%"
                if exact_match is not None
                else "Unavailable"
            )

        with r2:
            metric_card(
                "Word error rate",
                f"{word_error_rate:.1f}%"
                if word_error_rate is not None
                else "Unavailable"
            )

        with r3:
            metric_card(
                "Transcribed words/min",
                f"{reading_speed:.1f}"
                if reading_speed is not None
                else "Unavailable"
            )

        display_saved_date(reading_record)

        st.caption(
            "These are speech-recognition-based measurements. "
            "They can be affected by pronunciation, language, recording "
            "quality, and errors in the reference transcript."
        )
    else:
        st.info(
            "Reading evidence is unavailable. Complete the Reading "
            "Assessment and save its results."
        )

    st.divider()
    st.subheader("👁️ Eye-Tracking Evidence")

    if eye_details:
        e1, e2, e3 = st.columns(3)

        with e1:
            metric_card(
                "Mean target error",
                f"{mean_error:.1f} pp"
                if mean_error is not None
                else "Unavailable"
            )

        with e2:
            metric_card(
                "Median target error",
                f"{median_error:.1f} pp"
                if median_error is not None
                else "Unavailable"
            )

        with e3:
            metric_card(
                "Valid gaze samples",
                str(valid_samples)
                if valid_samples is not None
                else "Unavailable"
            )

        if face_detection is not None:
            st.caption(
                f"Face detection rate: {face_detection:.1f}%"
            )

        display_saved_date(eye_record)

        with st.expander("View complete eye-tracking details"):
            st.json(eye_details)

        st.warning(
            "Your current task tracks a moving target. It does not measure "
            "reading eye movements, and its results are not used to decide "
            "the combined dyslexia evidence outcome."
        )
    else:
        st.info("No saved eye-tracking measurements are available.")

    # =========================================================
    # COMBINED DYSLEXIA EVIDENCE ANALYSIS
    # =========================================================
    st.divider()
    st.subheader("🧠 Combined Dyslexia Evidence Analysis")

    st.write(
        "This prototype brings together reading performance, the "
        "handwriting model's dataset class, and eye-tracking task "
        "availability. The eye-tracking measurements are shown as "
        "experimental context and do not increase or decrease the "
        "dyslexia evidence result."
    )

    # Prototype threshold only; not a clinically validated cutoff.
    READING_REVIEW_THRESHOLD = 80.0

    reading_available = exact_match is not None
    handwriting_available = handwriting_prediction is not None
    eye_tracking_available = bool(eye_details)

    reading_indicator = (
        reading_available
        and exact_match < READING_REVIEW_THRESHOLD
    )

    # Only the literal "Normal" class is treated as a normal
    # dataset classification. Other classes are supporting observations,
    # not proof of a learning difficulty.
    handwriting_supporting_indicator = handwriting_is_non_normal

    if not reading_available or not handwriting_available:
        combined_result = "INSUFFICIENT EVIDENCE"
        combined_reason = (
            "Both a saved reading result with exact word-match accuracy "
            "and a saved handwriting classification are needed for this "
            "prototype's combined review."
        )

    elif reading_indicator and handwriting_supporting_indicator:
        combined_result = "YES — INDICATORS DETECTED"
        combined_reason = (
            "The reading result is below the prototype review threshold, "
            "and the handwriting model returned a dataset class other "
            "than Normal. These observations warrant further review."
        )

    elif (
        not reading_indicator
        and handwriting_is_normal
    ):
        combined_result = "NO — INDICATORS NOT DETECTED"
        combined_reason = (
            "The available reading result is at or above the prototype "
            "review threshold, and the handwriting model returned the "
            "Normal dataset class. This does not rule out dyslexia."
        )

    else:
        combined_result = "INCONCLUSIVE — REVIEW NEEDED"
        combined_reason = (
            "The reading and handwriting observations do not agree "
            "sufficiently under the prototype's simple decision rules. "
            "Review the original tasks and consider a qualified assessment."
        )

    # Show the combined result prominently.
    if combined_result.startswith("YES"):
        st.error(f"### {combined_result}")
    elif combined_result.startswith("NO"):
        st.success(f"### {combined_result}")
    else:
        st.warning(f"### {combined_result}")

    st.write(combined_reason)

    st.markdown("#### Evidence used in this result")

    evidence_rows = [
        {
            "Evidence source": "Reading",
            "Observed result": (
                f"{exact_match:.1f}% exact word matches"
                if exact_match is not None
                else "Unavailable"
            ),
            "Prototype interpretation": (
                "Below review threshold"
                if reading_indicator
                else (
                    "At/above review threshold"
                    if reading_available
                    else "Cannot assess"
                )
            )
        },
        {
            "Evidence source": "Handwriting",
            "Observed result": (
                str(handwriting_prediction)
                if handwriting_prediction is not None
                else "Unavailable"
            ),
            "Prototype interpretation": (
                "Non-Normal dataset class; supporting observation only"
                if handwriting_supporting_indicator
                else (
                    "Normal dataset class"
                    if handwriting_is_normal
                    else "Cannot assess"
                )
            )
        },
        {
            "Evidence source": "Eye tracking",
            "Observed result": (
                f"Mean target error: {mean_error:.1f} pp"
                if mean_error is not None
                else (
                    "Task data available"
                    if eye_tracking_available
                    else "Unavailable"
                )
            ),
            "Prototype interpretation": (
                "Context only; not used to decide dyslexia result"
            )
        }
    ]

    st.dataframe(
        pd.DataFrame(evidence_rows),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Prototype reading review threshold: exact word-match accuracy "
        f"below {READING_REVIEW_THRESHOLD:.0f}%. This threshold is an "
        "engineering demonstration setting, not a validated clinical "
        "cutoff. The handwriting classes also require independent "
        "validation before they can support meaningful screening claims."
    )

    if not eye_tracking_available:
        st.info(
            "Eye-tracking results are missing. The current prototype can "
            "still display a reading/handwriting evidence outcome, but "
            "the eye-tracking portion of the combined report is incomplete."
        )

    st.warning(
        "This is a research screening prototype, not a medical diagnosis. "
        "A YES result does not confirm dyslexia, and a NO result does not "
        "exclude it. Persistent concerns should be assessed by a qualified "
        "educational or clinical professional."
    )

    # =========================================================
    # OTHER SCREENING RESULTS — KEPT SEPARATE FROM DYSLEXIA
    # =========================================================
    st.divider()
    st.subheader("Other Assessment Results")

    with st.expander("🔢 Mathematics Assessment"):
        if math_accuracy is not None:
            m1, m2, m3 = st.columns(3)

            with m1:
                metric_card("Accuracy", f"{math_accuracy:.1f}%")

            with m2:
                metric_card(
                    "Correct answers",
                    (
                        f"{correct_answers}/{total_questions}"
                        if correct_answers is not None
                        and total_questions is not None
                        else "Unavailable"
                    )
                )

            with m3:
                metric_card(
                    "Average response time",
                    (
                        f"{response_time:.2f}s"
                        if response_time is not None
                        else "Unavailable"
                    )
                )

            display_saved_date(math_record)
        else:
            st.info("No mathematics result is available.")

    with st.expander("🧮 Dyscalculia Screening"):
        if dyscalculia_prediction is not None:
            st.write(f"**Recorded model output:** {dyscalculia_prediction}")

            dyscalculia_confidence_value = safe_float(
                dyscalculia_confidence
            )

            if dyscalculia_confidence_value is not None:
                st.metric(
                    "Stored model confidence",
                    f"{dyscalculia_confidence_value:.1f}%"
                )

            display_saved_date(dyscalculia_record)
        else:
            st.info("No dyscalculia model result is available.")

        st.caption(
            "This experimental model output is separate from the "
            "combined dyslexia evidence result."
        )

    with st.expander("✏️ Dysgraphia-Related Image Features"):
        if dysgraphia_details:
            feature_labels = [
                ("ink_density_percent", "Ink coverage (%)"),
                (
                    "writing_region_area_percent",
                    "Writing region area (%)"
                ),
                ("horizontal_ink_bands", "Horizontal ink bands"),
                (
                    "visible_connected_components",
                    "Visible components"
                ),
                ("median_row_gap_px", "Median row gap (pixels)")
            ]

            feature_rows = [
                {
                    "Measurement": label,
                    "Observed value": dysgraphia_details[key]
                }
                for key, label in feature_labels
                if dysgraphia_details.get(key) is not None
            ]

            if feature_rows:
                st.dataframe(
                    pd.DataFrame(feature_rows),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.json(dysgraphia_details)

        else:
            st.info("No writing-image feature measurements are available.")

        st.caption(
            "These are descriptive image measurements, not a validated "
            "dysgraphia classifier."
        )


    # =========================================================
    # COMBINED DYSGRAPHIA EVIDENCE ANALYSIS
    # =========================================================
    st.divider()
    st.subheader("✏️ Combined Dysgraphia Evidence Analysis")

    st.write(
        "This section reviews handwriting-image measurements, "
        "OCR-recognized text, and observations made by the reviewer. "
        "The result is a preliminary research summary, not a diagnosis."
    )

    # ---------------------------------------------------------
    # 1. DISPLAY INFORMATION EXTRACTED FROM THE UPLOADED IMAGE
    # ---------------------------------------------------------
    st.markdown("#### 1. Handwriting Image Analysis")

    dysgraphia_feature_aliases = {
        "Ink coverage (%)": [
            "ink_density_percent",
            "ink_density"
        ],
        "Writing region area (%)": [
            "writing_region_area_percent",
            "writing_area_percent"
        ],
        "Horizontal ink bands": [
            "horizontal_ink_bands"
        ],
        "Connected components": [
            "visible_connected_components",
            "connected_components"
        ],
        "Median row gap (pixels)": [
            "median_row_gap_px",
            "median_row_gap"
        ]
    }

    extracted_features = {}

    for label, possible_keys in dysgraphia_feature_aliases.items():
        value = None

        for feature_key in possible_keys:
            if dysgraphia_details.get(feature_key) is not None:
                value = dysgraphia_details.get(feature_key)
                break

        extracted_features[label] = value

    available_features = {
        label: value
        for label, value in extracted_features.items()
        if value is not None
    }

    if available_features:
        st.dataframe(
            pd.DataFrame([
                {
                    "Image measurement": label,
                    "Extracted value": value
                }
                for label, value in available_features.items()
            ]),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Measurements are extracted from the uploaded handwriting "
            "image. Their meaning depends on image resolution, writing "
            "task, age, and other factors."
        )
    else:
        st.warning(
            "No saved handwriting-image measurements were found. "
            "Run the handwriting image analysis first and ensure its "
            "features are saved under the selected child's ID."
        )

    # ---------------------------------------------------------
    # 2. DISPLAY OCR TEXT EXTRACTED FROM THE SAME INPUT
    # ---------------------------------------------------------
    st.markdown("#### 2. Recognized Handwriting Content")

    handwriting_text = str(recognized_text or "").strip()

    if handwriting_text:
        st.text_area(
            "Text recognized from the handwriting image",
            value=handwriting_text,
            height=120,
            disabled=True,
            key="dysgraphia_evidence_ocr_text"
        )
    else:
        st.info(
            "No recognized text is available. Run handwriting OCR "
            "on the uploaded image if text recognition is required."
        )

    st.caption(
        "OCR text helps review what the system recognized, but text "
        "recognition errors alone do not indicate dysgraphia."
    )

    # ---------------------------------------------------------
    # 3. HUMAN REVIEW CHECKLIST
    # ---------------------------------------------------------
    st.markdown("#### 3. Handwriting Observation Checklist")

    st.write(
        "Review the original uploaded image. Select an observation "
        "only if you can actually see it in the handwriting. "
        "Do not select an item simply because a measurement is unusual."
    )

    review_completed = st.checkbox(
        "I have reviewed the original handwriting image",
        key="dysgraphia_image_review_completed"
    )

    observation_labels = [
        "Letter shapes or formation appear inconsistent",
        "Letter size or spacing appears inconsistent",
        "Writing alignment on the line appears inconsistent",
        "Letters or words are difficult to distinguish",
        "Writing appears unusually effortful or poorly organized"
    ]

    selected_observations = []

    for index, observation in enumerate(observation_labels):
        checked = st.checkbox(
            observation,
            key=f"dysgraphia_observation_{index}"
        )

        if checked:
            selected_observations.append(observation)

    observation_count = len(selected_observations)

    st.metric(
        "Selected handwriting observations",
        f"{observation_count}/{len(observation_labels)}"
    )

    # ---------------------------------------------------------
    # 4. COMBINED EVIDENCE DECISION
    # ---------------------------------------------------------
    st.markdown("#### 4. Combined Dysgraphia Evidence Result")

    has_image_evidence = bool(available_features)
    has_text_evidence = bool(handwriting_text)
    has_any_image_information = (
        has_image_evidence or has_text_evidence
    )

    if not has_any_image_information:
        dysgraphia_evidence_result = "INSUFFICIENT EVIDENCE"

        dysgraphia_evidence_reason = (
            "No saved image measurements or recognized handwriting "
            "text are available for this child."
        )

    elif not review_completed:
        dysgraphia_evidence_result = "INSUFFICIENT EVIDENCE"

        dysgraphia_evidence_reason = (
            "The uploaded image must be reviewed before the checklist "
            "can contribute to the combined result."
        )

    elif observation_count >= 2:
        dysgraphia_evidence_result = "YES — INDICATORS FLAGGED FOR REVIEW"

        dysgraphia_evidence_reason = (
            "Multiple handwriting observations were selected during "
            "image review. Further assessment may be appropriate; "
            "these observations do not establish dysgraphia."
        )

    elif observation_count == 0:
        dysgraphia_evidence_result = (
            "NO — NO CHECKLIST INDICATORS SELECTED"
        )

        dysgraphia_evidence_reason = (
            "No listed observations were selected during this review. "
            "This does not rule out dysgraphia."
        )

    else:
        dysgraphia_evidence_result = "INCONCLUSIVE — REVIEW NEEDED"

        dysgraphia_evidence_reason = (
            "One handwriting observation was selected. Review the "
            "writing sample and consider additional age-appropriate "
            "writing tasks before interpreting the result."
        )

    if dysgraphia_evidence_result.startswith("YES"):
        st.error(f"### {dysgraphia_evidence_result}")

    elif dysgraphia_evidence_result.startswith("NO"):
        st.success(f"### {dysgraphia_evidence_result}")

    else:
        st.warning(f"### {dysgraphia_evidence_result}")

    st.write(dysgraphia_evidence_reason)

    # Show exactly what informed the result.
    st.markdown("#### Evidence Summary")

    evidence_summary_rows = [
        {
            "Evidence source": "Handwriting image measurements",
            "Status": (
                f"{len(available_features)} measurement(s) available"
                if has_image_evidence
                else "Unavailable"
            )
        },
        {
            "Evidence source": "OCR-recognized text",
            "Status": (
                "Available"
                if has_text_evidence
                else "Unavailable"
            )
        },
        {
            "Evidence source": "Manual image review",
            "Status": (
                "Completed"
                if review_completed
                else "Not completed"
            )
        },
        {
            "Evidence source": "Selected observations",
            "Status": f"{observation_count} selected"
        }
    ]

    st.dataframe(
        pd.DataFrame(evidence_summary_rows),
        use_container_width=True,
        hide_index=True
    )

    if selected_observations:
        st.markdown("**Selected observations:**")

        for observation in selected_observations:
            st.write(f"- {observation}")

    st.warning(
        "Important: This checklist is a prototype review aid, not a "
        "validated dysgraphia screening instrument. The number of "
        "selected observations is not a clinical score. A qualified "
        "professional should interpret persistent handwriting "
        "difficulties in context."
    )

    # ---------------------------------------------------------
    # 5. SAVE DYSGRAPHIA EVIDENCE REPORT
    # ---------------------------------------------------------
    if child_id and st.button(
        "💾 Save Dysgraphia Evidence Report",
        key="save_dysgraphia_evidence_report"
    ):
        dysgraphia_report = {
            "child_id": child_id,
            "result": dysgraphia_evidence_result,
            "reason": dysgraphia_evidence_reason,
            "image_measurements": extracted_features,
            "recognized_handwriting_text": handwriting_text,
            "image_review_completed": review_completed,
            "selected_observations": selected_observations,
            "observation_count": observation_count,
            "image_measurements_available": has_image_evidence,
            "ocr_text_available": has_text_evidence,
            "interpretation": (
                "AI-assisted research evidence review. Not a diagnosis. "
                "The checklist is not clinically validated."
            )
        }

        try:
            save_assessment_result(
                child_id=child_id,
                assessment_type="dysgraphia_evidence",
                result=dysgraphia_evidence_result,
                confidence=None,
                details=json.dumps(
                    dysgraphia_report,
                    ensure_ascii=False
                )
            )

            st.success(
                f"Dysgraphia evidence report saved for child {child_id}."
            )

        except Exception as save_error:
            st.error("Could not save the dysgraphia evidence report.")
            st.exception(save_error)



    # =========================================================
    # COMBINED DYSCALCULIA EVIDENCE ANALYSIS
    # =========================================================
    st.divider()
    st.subheader("🔢 Combined Dyscalculia Evidence Analysis")

    st.write(
        "This section combines the saved dyscalculia screening model "
        "output with the child's mathematics assessment performance."
    )

    # ---------------------------------------------------------
    # 1. DISPLAY DYSCALCULIA MODEL RESULT
    # ---------------------------------------------------------
    st.markdown("#### 1. Dyscalculia Screening Model")

    dyscalculia_model_available = (
        dyscalculia_prediction is not None
    )

    dyscalculia_confidence_value = safe_float(
        dyscalculia_confidence
    )

    if dyscalculia_model_available:
        d1, d2 = st.columns(2)

        with d1:
            metric_card(
                "Recorded model output",
                str(dyscalculia_prediction)
            )

        with d2:
            metric_card(
                "Model confidence",
                (
                    f"{dyscalculia_confidence_value:.1f}%"
                    if dyscalculia_confidence_value is not None
                    else "Unavailable"
                )
            )

        display_saved_date(dyscalculia_record)

    else:
        st.info(
            "No saved dyscalculia model result was found for this child."
        )

    # ---------------------------------------------------------
    # 2. DISPLAY MATHEMATICS PERFORMANCE
    # ---------------------------------------------------------
    st.markdown("#### 2. Mathematics Assessment")

    mathematics_available = (
        math_accuracy is not None
        and correct_answers is not None
        and total_questions is not None
    )

    if math_accuracy is not None:
        m1, m2, m3 = st.columns(3)

        with m1:
            metric_card(
                "Mathematics accuracy",
                f"{math_accuracy:.1f}%"
            )

        with m2:
            metric_card(
                "Correct answers",
                (
                    f"{correct_answers}/{total_questions}"
                    if correct_answers is not None
                    and total_questions is not None
                    else "Unavailable"
                )
            )

        with m3:
            metric_card(
                "Average response time",
                (
                    f"{response_time:.2f}s"
                    if response_time is not None
                    else "Unavailable"
                )
            )

        display_saved_date(math_record)

    else:
        st.info(
            "No saved mathematics accuracy result was found for this child."
        )

    # ---------------------------------------------------------
    # 3. COMBINED EVIDENCE DECISION
    # ---------------------------------------------------------
    st.markdown("#### 3. Combined Evidence Result")

    # Prototype demonstration threshold only.
    # This is NOT a validated clinical threshold.
    MATH_REVIEW_THRESHOLD = 60.0

    model_output = str(
        dyscalculia_prediction or ""
    ).strip().lower()

    # Adapt these terms to the labels returned by your trained model.
    model_suggests_difficulty = any(
        phrase in model_output
        for phrase in (
            "dyscalculia pattern detected",
            "dyscalculia detected",
            "dyscalculia",
            "positive",
            "high risk"
        )
    )

    model_suggests_no_difficulty = any(
        phrase in model_output
        for phrase in (
            "no dyscalculia",
            "no dyscalculia pattern",
            "control",
            "negative",
            "low risk"
        )
    )

    # Explicit negative labels take precedence over generic
    # substring matches such as "no dyscalculia".
    if any(
        phrase in model_output
        for phrase in (
            "no dyscalculia",
            "no dyscalculia pattern",
            "control",
            "negative",
            "low risk"
        )
    ):
        model_suggests_difficulty = False
        model_suggests_no_difficulty = True

    math_below_threshold = (
        math_accuracy is not None
        and math_accuracy < MATH_REVIEW_THRESHOLD
    )

    if (
        not dyscalculia_model_available
        or math_accuracy is None
    ):
        dyscalculia_combined_result = "INSUFFICIENT EVIDENCE"

        dyscalculia_combined_reason = (
            "Both a saved dyscalculia model result and mathematics "
            "accuracy are needed for this prototype's combined review."
        )

    elif (
        model_suggests_difficulty
        and math_below_threshold
    ):
        dyscalculia_combined_result = (
            "YES — INDICATORS DETECTED"
        )

        dyscalculia_combined_reason = (
            "The model output suggests a dyscalculia-related pattern, "
            "and mathematics accuracy is below the prototype review "
            "threshold. Further assessment may be appropriate."
        )

    elif (
        model_suggests_no_difficulty
        and not math_below_threshold
    ):
        dyscalculia_combined_result = (
            "NO — INDICATORS NOT DETECTED"
        )

        dyscalculia_combined_reason = (
            "The model output does not flag a dyscalculia-related "
            "pattern, and mathematics accuracy is at or above the "
            "prototype review threshold. This does not rule out "
            "dyscalculia."
        )

    else:
        dyscalculia_combined_result = (
            "INCONCLUSIVE — REVIEW NEEDED"
        )

        dyscalculia_combined_reason = (
            "The model output and mathematics performance do not "
            "agree under the prototype decision rules, or the model "
            "label could not be interpreted reliably. Review the "
            "assessment results before drawing conclusions."
        )

    if dyscalculia_combined_result.startswith("YES"):
        st.error(f"### {dyscalculia_combined_result}")

    elif dyscalculia_combined_result.startswith("NO"):
        st.success(f"### {dyscalculia_combined_result}")

    else:
        st.warning(f"### {dyscalculia_combined_result}")

    st.write(dyscalculia_combined_reason)

    # ---------------------------------------------------------
    # 4. TRANSPARENT EVIDENCE TABLE
    # ---------------------------------------------------------
    st.markdown("#### Evidence Summary")

    dyscalculia_evidence_rows = [
        {
            "Evidence source": "Dyscalculia screening model",
            "Observed result": (
                str(dyscalculia_prediction)
                if dyscalculia_model_available
                else "Unavailable"
            ),
            "Interpretation": (
                "Model suggests a pattern"
                if model_suggests_difficulty
                else (
                    "Model does not flag a pattern"
                    if model_suggests_no_difficulty
                    else "Unclear model label"
                )
            )
        },
        {
            "Evidence source": "Mathematics assessment",
            "Observed result": (
                f"{math_accuracy:.1f}% accuracy"
                if math_accuracy is not None
                else "Unavailable"
            ),
            "Interpretation": (
                "Below prototype review threshold"
                if math_below_threshold
                else (
                    "At/above prototype review threshold"
                    if math_accuracy is not None
                    else "Cannot assess"
                )
            )
        },
        {
            "Evidence source": "Response time",
            "Observed result": (
                f"{response_time:.2f} seconds"
                if response_time is not None
                else "Unavailable"
            ),
            "Interpretation": (
                "Context only; no response-time cutoff applied"
            )
        }
    ]

    st.dataframe(
        pd.DataFrame(dyscalculia_evidence_rows),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"The mathematics review threshold of {MATH_REVIEW_THRESHOLD:.0f}% "
        "is a prototype setting, not a clinically validated cutoff. "
        "Performance also depends on age, grade, question difficulty, "
        "language, and assessment conditions."
    )

    st.warning(
        "This combined result is an experimental research summary, "
        "not a diagnosis. Model confidence is not the probability that "
        "a child has dyscalculia. A NO result does not rule it out."
    )

    # ---------------------------------------------------------
    # 5. SAVE COMBINED DYSCALCULIA EVIDENCE REPORT
    # ---------------------------------------------------------
    if child_id and st.button(
        "💾 Save Combined Dyscalculia Evidence Report",
        key="save_combined_dyscalculia_evidence_report"
    ):
        dyscalculia_report = {
            "child_id": child_id,
            "combined_result": dyscalculia_combined_result,
            "combined_reason": dyscalculia_combined_reason,
            "dyscalculia_model": {
                "prediction": (
                    str(dyscalculia_prediction)
                    if dyscalculia_prediction is not None
                    else None
                ),
                "confidence": dyscalculia_confidence_value
            },
            "mathematics": {
                "accuracy_percent": math_accuracy,
                "correct_answers": correct_answers,
                "total_questions": total_questions,
                "average_response_time_seconds": response_time
            },
            "prototype_rule": {
                "math_review_threshold_percent": (
                    MATH_REVIEW_THRESHOLD
                ),
                "limitations": (
                    "Prototype-only rules; not clinically validated. "
                    "A negative result does not rule out dyscalculia."
                )
            },
            "interpretation": (
                "AI-assisted research evidence summary. Not a diagnosis."
            )
        }

        try:
            save_assessment_result(
                child_id=child_id,
                assessment_type="combined_dyscalculia_evidence",
                result=dyscalculia_combined_result,
                confidence=None,
                details=json.dumps(
                    dyscalculia_report,
                    ensure_ascii=False
                )
            )

            st.success(
                "Combined dyscalculia evidence report saved successfully."
            )

        except Exception as save_error:
            st.error(
                "Could not save the combined dyscalculia evidence report."
            )
            st.exception(save_error)


    # =========================================================
    # SAVE THE COMBINED DYSLEXIA EVIDENCE REPORT
    # =========================================================
    st.divider()

    if child_id and st.button(
        "💾 Save Combined Dyslexia Evidence Report",
        key="save_combined_dyslexia_evidence_report"
    ):
        report = {
            "child_id": child_id,
            "combined_result": combined_result,
            "combined_reason": combined_reason,
            "reading": {
                "exact_word_match_percent": exact_match,
                "word_error_rate_percent": word_error_rate,
                "transcribed_words_per_minute": reading_speed,
                "available": reading_available
            },
            "handwriting": {
                "predicted_dataset_class": (
                    str(handwriting_prediction)
                    if handwriting_prediction is not None
                    else None
                ),
                "model_confidence": safe_float(handwriting_confidence),
                "available": handwriting_available
            },
            "eye_tracking": {
                "available": eye_tracking_available,
                "mean_target_error": mean_error,
                "median_target_error": median_error,
                "valid_gaze_samples": valid_samples,
                "face_detection_rate": face_detection,
                "used_as_dyslexia_decision_indicator": False
            },
            "prototype_rule": {
                "reading_review_threshold_percent": (
                    READING_REVIEW_THRESHOLD
                ),
                "yes_rule": (
                    "Reading below threshold AND handwriting class "
                    "is not Normal"
                ),
                "no_rule": (
                    "Reading at/above threshold AND handwriting class "
                    "is Normal"
                ),
                "limitations": (
                    "Prototype-only rules; not clinically validated. "
                    "A negative result does not rule out dyslexia."
                )
            },
            "recognized_handwriting_text": str(
                recognized_text or ""
            ),
            "interpretation": (
                "AI-assisted research screening summary. Not a "
                "medical diagnosis."
            )
        }

        try:
            save_assessment_result(
                child_id=child_id,
                assessment_type="combined_dyslexia_evidence",
                result=combined_result,
                confidence=None,
                details=json.dumps(report, ensure_ascii=False)
            )

            st.success(
                "Combined dyslexia evidence report saved successfully."
            )

        except Exception as save_error:
            st.error("Could not save the combined evidence report.")
            st.exception(save_error)



# ============================================================
# FINAL REPORT
# ============================================================


elif page == "📄 Final Report":

    import io
    import json
    import html
    from datetime import datetime

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak
    )

    hero(
        "Final Screening Report",
        "Combined assessment evidence, prototype findings, and downloadable report."
    )

    # --------------------------------------------------
    # 1. CHECK CHILD SELECTION
    # --------------------------------------------------
    child_id = st.session_state.get("child_id")

    if not child_id:

        st.info(
            "Please complete the Child Profile and select a child "
            "before generating the Final Report."
        )

    else:

        child_age = st.session_state.get("child_age")
        child_grade = st.session_state.get("child_grade")

        st.markdown("### Child information")

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card("Child ID", str(child_id))

        with c2:
            metric_card(
                "Age",
                f"{child_age} years" if child_age is not None else "Not available"
            )

        with c3:
            metric_card(
                "Grade",
                str(child_grade) if child_grade is not None else "Not available"
            )

        # --------------------------------------------------
        # 2. LOAD SAVED ASSESSMENTS FROM DATABASE
        # --------------------------------------------------
        assessment_records = {}
        database_error = None

        try:
            with get_db_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT assessment_type, result, confidence, details,
                           result_json, created_at
                    FROM assessment_results
                    WHERE child_id = ?
                    ORDER BY created_at DESC
                    """,
                    (child_id,)
                ).fetchall()

            # Rows are newest first. Keep the latest record for each type.
            for row in rows:
                row_data = dict(row)
                assessment_type = row_data.get("assessment_type")

                if assessment_type and assessment_type not in assessment_records:
                    assessment_records[assessment_type] = row_data

        except Exception as exc:
            database_error = str(exc)
            st.error(
                "The saved assessments could not be loaded from the database. "
                "Check the database connection and assessment_results table."
            )

        if database_error:
            st.caption(f"Database detail: {database_error}")

        # --------------------------------------------------
        # 3. SAFE HELPERS
        # --------------------------------------------------
        def report_details(record):
            """Return assessment details as a dictionary when possible."""
            if not record:
                return {}

            for field in ("details", "result_json"):
                value = record.get(field)

                if isinstance(value, dict):
                    return value

                if isinstance(value, str) and value.strip():
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, dict):
                            return parsed
                    except (json.JSONDecodeError, TypeError):
                        pass

            return {}

        def report_number(value):
            try:
                if value is None or value == "":
                    return None
                return float(value)
            except (ValueError, TypeError):
                return None

        def report_text(value, default="Not available"):
            if value is None or str(value).strip() == "":
                return default
            return str(value)

        def latest_record(*possible_types):
            for assessment_type in possible_types:
                if assessment_type in assessment_records:
                    return assessment_records[assessment_type]
            return None

        def record_result(record):
            if not record:
                return None
            return record.get("result")

        def record_confidence(record):
            if not record:
                return None
            return report_number(record.get("confidence"))

        def evidence_verdict(record):
            """
            Interpret an explicitly saved evidence-review result.
            Avoid treating arbitrary raw model labels as a final verdict.
            """
            if not record:
                return "Insufficient evidence"

            result = str(record.get("result") or "").strip()
            lowered = result.lower()

            if (
                lowered.startswith("yes")
                or "indicators flagged" in lowered
                or "indicators detected" in lowered
                or "further evaluation recommended" in lowered
            ):
                return "Indicators flagged for further review"

            if (
                lowered.startswith("no")
                or "no indicators flagged" in lowered
                or "no indicators detected" in lowered
            ):
                return "No indicators flagged by the prototype review"

            if "inconclusive" in lowered:
                return "Inconclusive"

            if "insufficient" in lowered or not result:
                return "Insufficient evidence"

            # A saved result may be explanatory text rather than a verdict.
            return "Review required — result needs interpretation"

        def record_summary(record):
            if not record:
                return "Not completed or no saved result found."

            result = report_text(record.get("result"))
            confidence = report_confidence(record)

            if confidence is not None:
                result += f" (recorded confidence: {confidence:.1f}%)"

            return result

        # --------------------------------------------------
        # 4. FIND THE SAVED EVIDENCE REVIEWS
        # --------------------------------------------------
        dyslexia_evidence_record = latest_record(
            "combined_dyslexia_evidence",
            "dyslexia_evidence"
        )

        dysgraphia_evidence_record = latest_record(
            "dysgraphia_evidence"
        )

        dyscalculia_evidence_record = latest_record(
            "combined_dyscalculia_evidence",
            "dyscalculia_evidence"
        )

        handwriting_record = latest_record("handwriting")
        dysgraphia_record = latest_record("dysgraphia")
        ocr_record = latest_record("handwriting_ocr")
        reading_record = latest_record("reading")
        dyscalculia_record = latest_record("dyscalculia")
        math_record = latest_record(
            "mathematics",
            "math",
            "math_assessment"
        )
        eye_record = latest_record("eye_tracking")

        # --------------------------------------------------
        # 5. COMBINE SESSION DATA WITH SAVED DATA
        #    Session values are used only when they belong to
        #    the currently selected child.
        # --------------------------------------------------
        session_child_matches = (
            st.session_state.get("child_id") == child_id
        )

        def session_value(key, owner_key=None):
            if not session_child_matches:
                return None

            if owner_key:
                owner = st.session_state.get(owner_key)
                if owner != child_id:
                    return None

            return st.session_state.get(key)

        handwriting_prediction = (
            record_result(handwriting_record)
            or session_value("handwriting_prediction", "handwriting_child_id")
        )

        handwriting_confidence = record_confidence(handwriting_record)
        if handwriting_confidence is None:
            handwriting_confidence = report_number(
                session_value("handwriting_confidence", "handwriting_child_id")
            )

        dysgraphia_details = report_details(dysgraphia_record)
        ocr_details = report_details(ocr_record)
        reading_details = report_details(reading_record)
        dyscalculia_details = report_details(dyscalculia_record)
        math_details = report_details(math_record)
        eye_details = report_details(eye_record)

        # Some older results may exist in the matching session.
        if not ocr_details:
            session_ocr = session_value(
                "handwriting_ocr_text",
                "handwriting_ocr_child_id"
            )
            if session_ocr:
                ocr_details = {"recognized_text": session_ocr}

        if not reading_details:
            session_reading = session_value(
                "reading_results",
                "reading_results_child_id"
            )
            if isinstance(session_reading, dict):
                reading_details = session_reading

        if not dysgraphia_details:
            session_dysgraphia = session_value(
                "dysgraphia_features",
                "dysgraphia_features_child_id"
            )
            if isinstance(session_dysgraphia, dict):
                dysgraphia_details = session_dysgraphia

        if not math_details:
            session_math = session_value("math_accuracy", "math_child_id")
            if session_math is not None:
                math_details = {
                    "accuracy": session_math,
                    "correct_answers": session_value(
                        "math_correct_answers", "math_child_id"
                    ),
                    "total_questions": session_value(
                        "math_total_questions", "math_child_id"
                    ),
                    "average_response_time": session_value(
                        "math_avg_response_time", "math_child_id"
                    )
                }

        dyscalculia_model_result = record_result(dyscalculia_record)
        dyscalculia_model_confidence = record_confidence(dyscalculia_record)

        if dyscalculia_model_result is None:
            dyscalculia_model_result = session_value(
                "dyscalculia_prediction",
                "dyscalculia_child_id"
            )

        if dyscalculia_model_confidence is None:
            dyscalculia_model_confidence = report_number(
                session_value(
                    "dyscalculia_confidence",
                    "dyscalculia_child_id"
                )
            )

        if dyscalculia_model_confidence is None:
            dyscalculia_model_confidence = report_number(
                session_value(
                    "dyscalculia_probability",
                    "dyscalculia_child_id"
                )
            )

        # --------------------------------------------------
        # 6. EXTRACT ASSESSMENT METRICS
        # --------------------------------------------------
        reading_exact_match = report_number(
            reading_details.get("exact_word_match_percent")
        )
        reading_error_rate = report_number(
            reading_details.get("word_error_rate_percent")
        )
        reading_wpm = report_number(
            reading_details.get("transcribed_words_per_minute")
        )

        math_accuracy = report_number(
            math_details.get(
                "accuracy",
                math_details.get("math_accuracy")
            )
        )
        math_correct = math_details.get(
            "correct_answers",
            math_details.get("math_correct_answers")
        )
        math_total = math_details.get(
            "total_questions",
            math_details.get("math_total_questions")
        )
        math_response_time = report_number(
            math_details.get(
                "average_response_time",
                math_details.get("avg_response_time",
                                 math_details.get("math_avg_response_time"))
            )
        )

        ink_density = report_number(
            dysgraphia_details.get("ink_density_percent")
        )
        ink_bands = dysgraphia_details.get("horizontal_ink_bands")
        ink_components = dysgraphia_details.get("visible_connected_components")

        recognized_text = ocr_details.get("recognized_text")
        if not recognized_text:
            recognized_text = ocr_details.get("text")

        # --------------------------------------------------
        # 7. INDIVIDUAL EVIDENCE VERDICTS
        # --------------------------------------------------
        dyslexia_verdict = evidence_verdict(dyslexia_evidence_record)
        dysgraphia_verdict = evidence_verdict(dysgraphia_evidence_record)
        dyscalculia_verdict = evidence_verdict(dyscalculia_evidence_record)

        # A raw handwriting class is not enough to establish dyslexia.
        # A raw dyscalculia model prediction is not itself the combined verdict.
        evidence_verdicts = [
            dyslexia_verdict,
            dysgraphia_verdict,
            dyscalculia_verdict
        ]

        flagged_count = sum(
            verdict == "Indicators flagged for further review"
            for verdict in evidence_verdicts
        )

        all_negative = all(
            verdict == "No indicators flagged by the prototype review"
            for verdict in evidence_verdicts
        )

        if flagged_count > 0:
            overall_verdict = (
                "Further evaluation recommended: one or more prototype "
                "evidence reviews flagged indicators for follow-up."
            )
        elif all_negative:
            overall_verdict = (
                "No indicators were flagged by the available prototype "
                "reviews. This does not rule out a learning difficulty."
            )
        else:
            overall_verdict = (
                "Insufficient evidence for an overall conclusion. "
                "Review the available results and complete missing assessments."
            )

        # --------------------------------------------------
        # 8. DISPLAY THE OVERALL VERDICT
        # --------------------------------------------------
        st.divider()
        st.markdown("### Overall screening summary")

        if flagged_count > 0:
            st.warning(overall_verdict)
        elif all_negative:
            st.success(overall_verdict)
        else:
            st.info(overall_verdict)

        st.caption(
            "This overall summary uses saved evidence-review results. "
            "Raw model labels, confidence scores, handwriting classes, and "
            "eye-tracking measurements are not treated as clinical diagnoses."
        )

        st.markdown("### Evidence review by area")

        area_rows = [
            ["Screening area", "Evidence review status"],
            ["Dyslexia-related evidence", dyslexia_verdict],
            ["Dysgraphia-related evidence", dysgraphia_verdict],
            ["Dyscalculia-related evidence", dyscalculia_verdict]
        ]

        st.table(area_rows)

        # --------------------------------------------------
        # 9. HANDWRITING AND DYSGRAPHIA DETAILS
        # --------------------------------------------------
        st.markdown("### ✍️ Handwriting and writing analysis")

        st.write(
            "**Handwriting model output:** "
            + report_text(handwriting_prediction, "Not completed")
        )

        if handwriting_confidence is not None:
            st.write(
                f"**Recorded model confidence:** {handwriting_confidence:.1f}%"
            )

        st.caption(
            "The handwriting model's Corrected, Normal, and Reversal classes "
            "describe dataset patterns; they do not independently diagnose dyslexia."
        )

        if dysgraphia_details:
            st.write("**Writing image features recorded:**")

            if ink_density is not None:
                st.write(f"- Ink coverage: {ink_density:.2f}%")
            if ink_bands is not None:
                st.write(f"- Horizontal ink bands: {ink_bands}")
            if ink_components is not None:
                st.write(f"- Visible connected components: {ink_components}")

            st.caption(
                "These are descriptive image measurements, not a validated "
                "dysgraphia classifier."
            )
        else:
            st.write("**Writing image features:** Not available")

        st.write(
            "**Dysgraphia evidence review:** " + dysgraphia_verdict
        )

        if dysgraphia_evidence_record:
            st.write(
                "**Saved review result:** "
                + report_text(dysgraphia_evidence_record.get("result"))
            )

        # --------------------------------------------------
        # 10. OCR RECOGNIZED TEXT
        # --------------------------------------------------
        st.markdown("### 🔤 Handwriting text recognition")

        if recognized_text:
            st.text_area(
                "Recognized handwriting text",
                value=str(recognized_text),
                height=120,
                disabled=True,
                key="final_report_recognized_text"
            )
        else:
            st.write("No saved handwriting OCR text was found.")

        st.caption(
            "OCR may misread handwriting. Recognized text should be checked "
            "against the original image before drawing conclusions."
        )

        # --------------------------------------------------
        # 11. READING ASSESSMENT
        # --------------------------------------------------
        st.markdown("### 📚 Reading assessment")

        if reading_record or reading_details:
            if reading_exact_match is not None:
                st.write(f"- Exact word matches: {reading_exact_match:.1f}%")
            if reading_error_rate is not None:
                st.write(f"- Word error rate: {reading_error_rate:.1f}%")
            if reading_wpm is not None:
                st.write(f"- Transcribed words per minute: {reading_wpm:.1f}")

            st.write(
                "**Dyslexia-related evidence review:** " + dyslexia_verdict
            )

            if dyslexia_evidence_record:
                st.write(
                    "**Saved review result:** "
                    + report_text(dyslexia_evidence_record.get("result"))
                )
            else:
                st.info(
                    "A combined dyslexia evidence review has not been saved. "
                    "Reading and handwriting measurements alone are not a diagnosis."
                )
        else:
            st.write("Reading assessment: Not completed or no saved record found.")

        # --------------------------------------------------
        # 12. DYSCALCULIA AND MATHEMATICS
        # --------------------------------------------------
        st.markdown("### 🔢 Dyscalculia and mathematics")

        st.write(
            "**Dyscalculia model output:** "
            + report_text(dyscalculia_model_result, "Not completed")
        )

        if dyscalculia_model_confidence is not None:
            st.write(
                f"**Recorded model score/confidence:** "
                f"{dyscalculia_model_confidence:.1f}%"
            )

        if math_accuracy is not None:
            st.write(f"**Mathematics accuracy:** {math_accuracy:.1f}%")

        if math_correct is not None and math_total is not None:
            st.write(
                f"**Correct answers:** {math_correct} out of {math_total}"
            )

        if math_response_time is not None:
            st.write(
                f"**Average response time:** {math_response_time:.2f} seconds"
            )

        st.write(
            "**Combined dyscalculia evidence review:** " + dyscalculia_verdict
        )

        if dyscalculia_evidence_record:
            st.write(
                "**Saved review result:** "
                + report_text(dyscalculia_evidence_record.get("result"))
            )
        else:
            st.info(
                "A combined dyscalculia evidence review has not been saved. "
                "The raw model output and mathematics score are shown separately."
            )

        # --------------------------------------------------
        # 13. EYE TRACKING
        # --------------------------------------------------
        st.markdown("### 👁️ Eye-tracking assessment")

        if eye_record:
            st.write(
                "**Saved result:** "
                + report_text(eye_record.get("result"))
            )

            if eye_details:
                for key, value in eye_details.items():
                    if isinstance(value, (str, int, float, bool)):
                        st.write(f"- {key.replace('_', ' ').title()}: {value}")

            st.caption(
                "This prototype uses an experimental moving-target gaze-following "
                "task. It is not a validated dyslexia test and is not treated as "
                "evidence for or against a learning disability."
            )
        else:
            st.write("Eye-tracking assessment: Not completed or no saved record found.")

        # --------------------------------------------------
        # 14. ALL SAVED ASSESSMENTS
        # --------------------------------------------------
        st.markdown("### 🗂️ Saved assessment records")

        if assessment_records:
            raw_rows = []

            for assessment_type, record in assessment_records.items():
                raw_rows.append({
                    "Assessment": assessment_type.replace("_", " ").title(),
                    "Result": report_text(record.get("result")),
                    "Confidence": (
                        f"{record_confidence(record):.1f}%"
                        if record_confidence(record) is not None
                        else "N/A"
                    ),
                    "Saved at": report_text(record.get("created_at"))
                })

            st.dataframe(
                raw_rows,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "No saved assessment records were found for this child. "
                "Complete assessments and save their results before exporting."
            )

        # --------------------------------------------------
        # 15. RECOMMENDATIONS AND LIMITATIONS
        # --------------------------------------------------
        st.markdown("### Recommendations")

        recommendations = [
            "Review each assessment result alongside the child's classroom performance and learning history.",
            "Verify OCR text and any automated handwriting interpretation against the original input.",
            "If learning difficulties persist, discuss the concerns with the child's teacher and a qualified educational or clinical professional.",
            "Do not use this prototype's confidence values or provisional thresholds as clinical probabilities."
        ]

        for recommendation in recommendations:
            st.write(f"- {recommendation}")

        st.info(
            "NeuroScreen AI is an AI-assisted research screening prototype, "
            "not a medical diagnosis. It cannot diagnose or rule out dyslexia, "
            "dysgraphia, dyscalculia, or any other condition. Its experimental "
            "models, image measurements, and assessment rules require appropriate "
            "validation before real-world clinical or educational decisions."
        )

        # --------------------------------------------------
        # 16. GENERATE DOWNLOADABLE PDF
        # --------------------------------------------------
        st.divider()
        st.markdown("### 📄 Download final report")

        if not assessment_records:
            st.warning(
                "No saved database assessment records were found. "
                "The PDF can still be generated, but it will clearly indicate "
                "that saved evidence is unavailable."
            )

        def pdf_safe(value):
            """Escape text for ReportLab Paragraphs."""
            return html.escape(report_text(value))

        def pdf_paragraph(text, style):
            return Paragraph(pdf_safe(text), style)

        def make_final_report_pdf():
            buffer = io.BytesIO()

            document = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=1.8 * cm,
                leftMargin=1.8 * cm,
                topMargin=1.7 * cm,
                bottomMargin=1.7 * cm,
                title="NeuroScreen AI - Final Screening Report",
                author="NeuroScreen AI Research Prototype"
            )

            styles = getSampleStyleSheet()

            styles.add(ParagraphStyle(
                name="ReportTitleCustom",
                parent=styles["Title"],
                alignment=TA_CENTER,
                fontSize=19,
                leading=24,
                textColor=colors.HexColor("#17365D"),
                spaceAfter=8
            ))

            styles.add(ParagraphStyle(
                name="ReportSectionCustom",
                parent=styles["Heading2"],
                fontSize=12,
                leading=16,
                textColor=colors.HexColor("#17365D"),
                spaceBefore=12,
                spaceAfter=6
            ))

            styles.add(ParagraphStyle(
                name="ReportBodyCustom",
                parent=styles["BodyText"],
                fontSize=9,
                leading=13,
                spaceAfter=5
            ))

            story = []

            story.append(Paragraph(
                "NeuroScreen AI",
                styles["ReportTitleCustom"]
            ))
            story.append(Paragraph(
                "Final Screening Report",
                styles["Heading2"]
            ))
            story.append(Spacer(1, 6))

            generated_at = datetime.now().strftime("%d %b %Y, %I:%M %p")

            profile_data = [
                ["Child ID", str(child_id)],
                ["Age", f"{child_age} years" if child_age is not None else "Not available"],
                ["Grade", str(child_grade) if child_grade is not None else "Not available"],
                ["Report generated", generated_at]
            ]

            profile_table = Table(
                profile_data,
                colWidths=[4 * cm, 12 * cm]
            )
            profile_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F8")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#17365D")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
            ]))

            story.append(profile_table)
            story.append(Spacer(1, 12))

            story.append(Paragraph(
                "Overall screening summary",
                styles["ReportSectionCustom"]
            ))
            story.append(Paragraph(
                pdf_safe(overall_verdict),
                styles["ReportBodyCustom"]
            ))

            story.append(Paragraph(
                "Evidence review by area",
                styles["ReportSectionCustom"]
            ))

            verdict_table_data = [
                ["Screening area", "Evidence review status"],
                ["Dyslexia-related", dyslexia_verdict],
                ["Dysgraphia-related", dysgraphia_verdict],
                ["Dyscalculia-related", dyscalculia_verdict]
            ]

            verdict_table_data = [
                [pdf_safe(cell) for cell in row]
                for row in verdict_table_data
            ]

            verdict_table = Table(
                verdict_table_data,
                colWidths=[5 * cm, 11 * cm],
                repeatRows=1
            )
            verdict_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
            ]))
            story.append(verdict_table)

            # All saved assessment types, including combined evidence reviews.
            story.append(Paragraph(
                "Saved assessment results",
                styles["ReportSectionCustom"]
            ))

            if assessment_records:
                for assessment_type, record in assessment_records.items():
                    story.append(Paragraph(
                        pdf_safe(assessment_type.replace("_", " ").title()),
                        styles["Heading3"]
                    ))
                    story.append(Paragraph(
                        "<b>Result:</b> " + pdf_safe(record.get("result")),
                        styles["ReportBodyCustom"]
                    ))

                    confidence = record_confidence(record)
                    if confidence is not None:
                        story.append(Paragraph(
                            f"<b>Recorded confidence:</b> {confidence:.1f}%",
                            styles["ReportBodyCustom"]
                        ))

                    if record.get("created_at"):
                        story.append(Paragraph(
                            "<b>Saved at:</b> " + pdf_safe(record.get("created_at")),
                            styles["ReportBodyCustom"]
                        ))

                    details = report_details(record)
                    if details:
                        story.append(Paragraph(
                            "<b>Recorded details:</b>",
                            styles["ReportBodyCustom"]
                        ))

                        for key, value in details.items():
                            if isinstance(value, (dict, list)):
                                value = json.dumps(value, ensure_ascii=False)

                            story.append(Paragraph(
                                pdf_safe(f"{key.replace('_', ' ').title()}: {value}"),
                                styles["ReportBodyCustom"]
                            ))

            else:
                story.append(Paragraph(
                    "No saved assessment records were found for this child.",
                    styles["ReportBodyCustom"]
                ))

            # Important derived metrics that may be in the current session
            # but not separately present in the database details.
            story.append(Paragraph(
                "Additional assessment summary",
                styles["ReportSectionCustom"]
            ))

            additional_items = [
                ("Handwriting model output", handwriting_prediction),
                (
                    "Handwriting model confidence",
                    f"{handwriting_confidence:.1f}%"
                    if handwriting_confidence is not None else None
                ),
                ("Recognized handwriting text", recognized_text),
                (
                    "Writing ink coverage",
                    f"{ink_density:.2f}%"
                    if ink_density is not None else None
                ),
                ("Horizontal ink bands", ink_bands),
                ("Visible connected components", ink_components),
                (
                    "Reading exact word matches",
                    f"{reading_exact_match:.1f}%"
                    if reading_exact_match is not None else None
                ),
                (
                    "Reading word error rate",
                    f"{reading_error_rate:.1f}%"
                    if reading_error_rate is not None else None
                ),
                (
                    "Reading words per minute",
                    f"{reading_wpm:.1f}"
                    if reading_wpm is not None else None
                ),
                ("Dyscalculia model output", dyscalculia_model_result),
                (
                    "Dyscalculia model score/confidence",
                    f"{dyscalculia_model_confidence:.1f}%"
                    if dyscalculia_model_confidence is not None else None
                ),
                (
                    "Mathematics accuracy",
                    f"{math_accuracy:.1f}%"
                    if math_accuracy is not None else None
                ),
                (
                    "Mathematics correct answers",
                    f"{math_correct} out of {math_total}"
                    if math_correct is not None and math_total is not None
                    else None
                ),
                (
                    "Average mathematics response time",
                    f"{math_response_time:.2f} seconds"
                    if math_response_time is not None else None
                )
            ]

            for label, value in additional_items:
                if value is not None and str(value).strip():
                    story.append(Paragraph(
                        "<b>" + pdf_safe(label) + ":</b> " + pdf_safe(value),
                        styles["ReportBodyCustom"]
                    ))

            story.append(Paragraph(
                "Recommendations",
                styles["ReportSectionCustom"]
            ))

            for recommendation in recommendations:
                story.append(Paragraph(
                    "• " + pdf_safe(recommendation),
                    styles["ReportBodyCustom"]
                ))

            story.append(Paragraph(
                "Important disclaimer",
                styles["ReportSectionCustom"]
            ))
            story.append(Paragraph(
                "This document summarizes an AI-assisted research screening "
                "prototype. It is not a medical diagnosis and cannot diagnose "
                "or rule out dyslexia, dysgraphia, dyscalculia, or another "
                "condition. Experimental model outputs, image measurements, "
                "and prototype thresholds are not validated clinical risk "
                "estimates. Results should be interpreted by appropriately "
                "qualified professionals alongside other relevant information.",
                styles["ReportBodyCustom"]
            ))

            def add_page_number(canvas, doc):
                canvas.saveState()
                canvas.setFont("Helvetica", 8)
                canvas.setFillColor(colors.grey)
                canvas.drawString(
                    1.8 * cm,
                    1.0 * cm,
                    "NeuroScreen AI | Research prototype | Not a medical diagnosis"
                )
                canvas.drawRightString(
                    A4[0] - 1.8 * cm,
                    1.0 * cm,
                    f"Page {doc.page}"
                )
                canvas.restoreState()

            document.build(
                story,
                onFirstPage=add_page_number,
                onLaterPages=add_page_number
            )

            buffer.seek(0)
            return buffer.getvalue()

        try:
            pdf_bytes = make_final_report_pdf()

            safe_child_id = "".join(
                char if char.isalnum() or char in ("-", "_") else "_"
                for char in str(child_id)
            )

            st.download_button(
                label="📄 Download Final Report PDF",
                data=pdf_bytes,
                file_name=f"NeuroScreen_Final_Report_{safe_child_id}.pdf",
                mime="application/pdf",
                key="download_neuroscreen_final_report_pdf",
                use_container_width=True
            )

            st.caption(
                "The PDF includes the overall summary, individual evidence-review "
                "statuses, saved assessment records, available metrics, recommendations, "
                "and the research disclaimer."
            )

        except Exception as exc:
            st.error(
                "The PDF could not be generated. Please check that ReportLab is "
                "installed and that the report data does not contain an unexpected format."
            )
            st.caption(f"Technical detail: {exc}")
