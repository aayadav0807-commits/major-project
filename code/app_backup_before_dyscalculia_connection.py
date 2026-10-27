
import streamlit as st
import torch
import pandas as pd
import time
import joblib
from PIL import Image
from torchvision import transforms
import sys
import os

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
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
<div class="brand">
    <div class="brand-title">🧠 NeuroScreen AI</div>
    <div class="brand-subtitle">Child Learning Screening Platform</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "🏠 Dashboard",
        "👤 Child Profile",
        "✍️ Handwriting Screening",
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

# ============================================================
# DEVICE + MODELS
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

HANDWRITING_MODEL_PATH = "models/dyslexia_cnn.pth"
DYSCALCULIA_MODEL_PATH = "models/dyscalculia_model.pkl"

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
        "Child Assessment Profile",
        "Set up the basic assessment information before starting the screening workflow."
    )

    with st.container(border=True):

        st.markdown("### Profile details")

        c1, c2, c3 = st.columns([1.5, 1, 1.2])

        with c1:
            child_id = st.text_input(
                "Child ID",
                placeholder="Example: CHILD001"
            )

        with c2:
            age = st.slider(
                "Age",
                min_value=6,
                max_value=12,
                value=8
            )

        with c3:
            grade = st.selectbox(
                "Grade",
                [
                    "Grade 1", "Grade 2", "Grade 3",
                    "Grade 4", "Grade 5", "Grade 6", "Grade 7"
                ]
            )

        st.caption("Age range configured for this project: 6–12 years.")

        if st.button("Save assessment profile", type="primary", use_container_width=True):

            if child_id.strip() == "":
                st.warning("Please enter a Child ID.")

            else:
                st.session_state["child_id"] = child_id.strip()
                st.session_state["child_age"] = age
                st.session_state["child_grade"] = grade

                st.success("Assessment profile saved.")

    if "child_id" in st.session_state:

        st.markdown("### Current profile")

        a, b, c = st.columns(3)

        with a:
            metric_card("Child ID", st.session_state["child_id"])

        with b:
            metric_card("Age", f"{st.session_state['child_age']} years")

        with c:
            metric_card("Grade", st.session_state["child_grade"])


# ============================================================
# HANDWRITING SCREENING
# ============================================================

elif page == "✍️ Handwriting Screening":

    hero(
        "Handwriting Screening",
        "Upload a handwriting sample and review the CNN classification."
    )

    uploaded_file = st.file_uploader(
        "Upload handwriting image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear handwriting sample with minimal background clutter."
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        left, right = st.columns([1, 1.2])

        with left:
            st.image(
                image,
                caption="Uploaded sample",
                width="stretch"
            )

        with right:

            st.markdown("### AI analysis")

            if st.button(
                "Analyze handwriting",
                type="primary",
                use_container_width=True
            ):

                image_tensor = handwriting_transform(image)
                image_tensor = image_tensor.unsqueeze(0).to(device)

                model = load_handwriting_model()

                with torch.no_grad():

                    output = model(image_tensor)

                    probabilities = torch.softmax(
                        output,
                        dim=1
                    )

                    predicted_class = torch.argmax(
                        probabilities,
                        dim=1
                    ).item()

                classes = [
                    "Corrected",
                    "Normal",
                    "Reversal"
                ]

                prediction = classes[predicted_class]

                confidence = (
                    probabilities[0][predicted_class].item() * 100
                )

                # Save handwriting result
                st.session_state["handwriting_prediction"] = prediction
                st.session_state["handwriting_confidence"] = confidence

                # Interpret handwriting pattern
                if prediction == "Normal":

                    handwriting_indicator = (
                        "No notable reversal/correction pattern detected."
                    )

                elif prediction == "Corrected":

                    handwriting_indicator = (
                        "A corrected handwriting pattern was detected. "
                        "This may indicate difficulty during writing and should "
                        "be considered along with other assessment results."
                    )

                else:

                    handwriting_indicator = (
                        "A reversal handwriting pattern was detected. "
                        "This may be relevant to further screening and should "
                        "be considered along with other assessment results."
                    )

                # Save interpretation for later modules
                st.session_state["handwriting_indicator"] = handwriting_indicator

                st.success(
                    f"Model output: {prediction}"
                )

                st.metric(
                    "AI confidence",
                    f"{confidence:.2f}%"
                )

                st.markdown("### Screening interpretation")

                if prediction == "Normal":

                    st.success(handwriting_indicator)

                else:

                    st.warning(handwriting_indicator)

                st.caption(
                    "The handwriting model identifies visual writing patterns. "
                    "This result is a screening indicator and is not a diagnosis."
                )

                st.markdown("#### Class probabilities")

                for i, class_name in enumerate(classes):

                    probability = (
                        probabilities[0][i].item() * 100
                    )

                    st.write(
                        f"**{class_name}** — {probability:.2f}%"
                    )

                    st.progress(
                        min(probability / 100, 1.0)
                    )

    else:

        st.info(
            "Upload a handwriting image to begin the analysis."
        )


# ============================================================
# DYSCALCULIA SCREENING
# ============================================================



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
        "Webcam-based eye tracking will be integrated into this module."
    )

    card(
        "Coming next",
        "This section will collect gaze-related measures such as fixation duration, gaze movement and reading efficiency once the webcam module is implemented."
    )

    st.info(
        "Eye tracking is currently a planned research module and is not yet active."
    )


# ============================================================
# OVERALL ANALYSIS
# ============================================================

elif page == "📊 Overall Risk Analysis":

    hero(
        "Overall Screening Analysis",
        "A consolidated view of the results collected across the assessment modules."
    )

    if "child_id" in st.session_state:

        st.markdown("### Child")

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

    else:
        st.info("Complete the Child Profile to populate this section.")

    st.markdown("### Screening results")

    c1, c2 = st.columns(2)

    with c1:

        if "handwriting_prediction" in st.session_state:

            metric_card(
                "Handwriting",
                st.session_state["handwriting_prediction"]
            )

            st.caption(
                f"Confidence: {st.session_state['handwriting_confidence']:.1f}%"
            )

        else:
            card(
                "Handwriting",
                "Assessment not completed."
            )

    with c2:

        if "dyscalculia_probability" in st.session_state:

            metric_card(
                "Dyscalculia probability",
                f"{st.session_state['dyscalculia_probability']:.1f}%"
            )

            st.caption(
                f"Model confidence: {st.session_state['dyscalculia_confidence']:.1f}%"
            )

        else:
            card(
                "Dyscalculia",
                "Assessment not completed."
            )

    if "math_accuracy" in st.session_state:

        st.markdown("### Mathematics")

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "Accuracy",
                f"{st.session_state['math_accuracy']:.1f}%"
            )

        with c2:
            metric_card(
                "Correct",
                f"{st.session_state['math_correct_answers']}/{st.session_state['math_total_questions']}"
            )

        with c3:
            metric_card(
                "Avg. response",
                f"{st.session_state['math_avg_response_time']:.2f}s"
            )

    st.warning(
        "The combined screen is a research summary. It should not be interpreted as a medical diagnosis."
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

        st.divider()

        st.info(
            "This report summarizes AI-assisted screening outputs and performance measures. "
            "It is not a clinical diagnosis."
        )

st.markdown(
    '<div class="footer-note">NeuroScreen AI · Final-year research prototype · Screening support only</div>',
    unsafe_allow_html=True
)
