import streamlit as st
import torch
import pandas as pd
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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Child Learning Disability Screening",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🧠 AI Child Learning Disability Screening System")

st.write(
    "An AI-assisted research system for analyzing handwriting "
    "and mathematical performance patterns."
)

st.warning(
    "⚠️ This system is an AI-assisted screening/research tool "
    "and is NOT a medical diagnosis."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📋 Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Dashboard",
        "👤 Child Profile",
        "✍️ Handwriting Screening",
        "🔢 Dyscalculia Screening",
        "👁️ Eye Tracking Assessment",
        "📊 Overall Risk Analysis",
        "📄 Final Report"
    ]
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HANDWRITING MODEL
# ============================================================

HANDWRITING_MODEL_PATH = "models/dyslexia_cnn.pth"


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


# ============================================================
# DYSCALCULIA MODEL
# ============================================================

DYSCALCULIA_MODEL_PATH = "models/dyscalculia_model.pkl"


@st.cache_resource
def load_dyscalculia_model():

    model = joblib.load(
        DYSCALCULIA_MODEL_PATH
    )

    return model
# ============================================================
# CHILD PROFILE
# ============================================================

if page == "👤 Child Profile":

    st.header("👤 Child Assessment Profile")

    st.write(
        "Enter basic information before starting the screening assessment."
    )

    st.divider()

    child_id = st.text_input(
        "Child ID",
        placeholder="Example: CHILD001"
    )

    age = st.number_input(
        "Age",
        min_value=6,
        max_value=12,
        value=8,
        step=1
    )

    grade = st.selectbox(
        "Grade",
        [
            "Grade 1",
            "Grade 2",
            "Grade 3",
            "Grade 4",
            "Grade 5",
            "Grade 6",
            "Grade 7"
        ]
    )

    st.divider()

    if st.button(
        "💾 Save Child Profile",
        type="primary"
    ):

        if child_id.strip() == "":
            st.warning("Please enter a Child ID.")

        else:

            st.session_state["child_id"] = child_id
            st.session_state["child_age"] = age
            st.session_state["child_grade"] = grade

            st.success(
                "Child profile saved successfully!"
            )

            st.write("### Assessment Profile")

            st.write(
                f"**Child ID:** {child_id}"
            )

            st.write(
                f"**Age:** {age} years"
            )

            st.write(
                f"**Grade:** {grade}"
            )

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("🏠 Project Dashboard")

    st.write(
        "Welcome to the AI Child Learning Disability "
        "Screening System."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("✍️ Handwriting Analysis")

        st.write(
            """
            The handwriting module uses a Convolutional
            Neural Network (CNN) to analyze handwriting images.

            The trained model classifies images into:

            • Normal
            • Corrected
            • Reversal
            """
        )

        st.info(
            "Handwriting model test accuracy: approximately 87.48%"
        )

    with col2:

        st.subheader("🔢 Dyscalculia Analysis")

        st.write(
            """
            The dyscalculia module analyzes mathematical
            performance features such as:

            • Accuracy
            • Reaction time
            • Enumeration performance
            • Symbolic comparison
            • Visuo-spatial working memory
            """
        )

        st.info(
            "Dyscalculia model trained using participant-level data."
        )

    st.divider()

    st.subheader("⚠️ Important")

    st.write(
        """
        The purpose of this project is to demonstrate how
        Artificial Intelligence can assist in identifying
        patterns associated with learning difficulties.

        The system should not be used to diagnose a child.
        Professional assessment by qualified specialists is
        required for diagnosis.
        """
    )


# ============================================================
# HANDWRITING SCREENING
# ============================================================

elif page == "✍️ Handwriting Screening":

    st.header("✍️ Handwriting Screening")

    st.write(
        "Upload a handwriting image for AI-based classification."
    )

    st.info(
        "The current handwriting model predicts the dataset "
        "categories Normal, Corrected, and Reversal."
    )

    uploaded_file = st.file_uploader(
        "Upload handwriting image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.subheader("📷 Uploaded Handwriting")

        st.image(
            image,
            caption="Handwriting Sample",
            width="stretch"
        )

        image_tensor = handwriting_transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)

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
            probabilities[0][predicted_class].item()
            * 100
        )

        st.divider()

        st.subheader("🔍 Screening Result")

        if prediction == "Normal":

            st.success(
                f"Prediction: {prediction}"
            )

        elif prediction == "Corrected":

            st.warning(
                f"Prediction: {prediction}"
            )

        else:

            st.error(
                f"Prediction: {prediction}"
            )

        st.metric(
            "AI Confidence",
            f"{confidence:.2f}%"
        )

        st.divider()

        st.subheader("📊 Class Probabilities")

        for i, class_name in enumerate(classes):

            probability = (
                probabilities[0][i].item()
                * 100
            )

            st.write(
                f"**{class_name}: {probability:.2f}%**"
            )

            st.progress(
                min(probability / 100, 1.0)
            )



# ============================================================
# DYSCALCULIA SCREENING
# ============================================================

elif page == "🔢 Dyscalculia Screening":

    st.header("🔢 Dyscalculia Screening")

    st.write(
        "Enter mathematical performance information "
        "to obtain an AI-assisted screening result."
    )

    st.warning(
        "⚠️ This is a research demonstration. "
        "It is not a clinical diagnosis."
    )

    st.divider()

    st.subheader("📊 Child Performance Information")

    st.write(
        "Enter the child's mathematical and cognitive "
        "performance values below."
    )

    # --------------------------------------------------------
    # COGNITIVE FEATURES
    # --------------------------------------------------------

    st.markdown("### 🧠 Cognitive Performance")

    col1, col2 = st.columns(2)

    with col1:
        similarities_iq = st.number_input(
            "Similarities IQ",
            min_value=0.0,
            value=0.0
        )

    with col2:
        matrixreas_iq = st.number_input(
            "Matrix Reasoning IQ",
            min_value=0.0,
            value=0.0
        )

    # --------------------------------------------------------
    # ANS FEATURES
    # --------------------------------------------------------

    st.markdown("### 🔢 Approximate Number System")

    col1, col2, col3 = st.columns(3)

    with col1:
        mean_acc_ans = st.number_input(
            "Mean ANS Accuracy",
            min_value=0.0,
            max_value=1.0,
            value=0.5
        )

    with col2:
        mean_rts_ans = st.number_input(
            "Mean ANS Reaction Time",
            min_value=0.0,
            value=1000.0
        )

    with col3:
        mean_rts_def = st.number_input(
            "Mean Default Reaction Time",
            min_value=0.0,
            value=1000.0
        )

    # --------------------------------------------------------
    # ACCURACY FEATURES
    # --------------------------------------------------------

    st.markdown("### 🎯 Accuracy Performance")

    col1, col2 = st.columns(2)

    with col1:
        mean_acc_13 = st.number_input(
            "Mean Accuracy 13",
            min_value=0.0,
            max_value=1.0,
            value=0.5
        )

    with col2:
        mean_acc_47 = st.number_input(
            "Mean Accuracy 47",
            min_value=0.0,
            max_value=1.0,
            value=0.5
        )

    # --------------------------------------------------------
    # REACTION TIME FEATURES
    # --------------------------------------------------------

    st.markdown("### ⏱️ Reaction Time Performance")

    col1, col2 = st.columns(2)

    with col1:
        meanrt_13 = st.number_input(
            "Mean Reaction Time 13",
            min_value=0.0,
            value=1000.0
        )

    with col2:
        meanrt_47 = st.number_input(
            "Mean Reaction Time 47",
            min_value=0.0,
            value=1000.0
        )

    # --------------------------------------------------------
    # EFFICIENCY FEATURES
    # --------------------------------------------------------

    st.markdown("### ⚡ Efficiency Scores")

    col1, col2 = st.columns(2)

    with col1:
        efficiency_13 = st.number_input(
            "Efficiency Score 13",
            min_value=0.0,
            value=0.0
        )

    with col2:
        efficiency_47 = st.number_input(
            "Efficiency Score 47",
            min_value=0.0,
            value=0.0
        )

    # --------------------------------------------------------
    # VARIABILITY FEATURES
    # --------------------------------------------------------

    st.markdown("### 📈 Response Variability")

    col1, col2 = st.columns(2)

    with col1:
        cov_13 = st.number_input(
            "CoV 13",
            min_value=0.0,
            value=0.0
        )

    with col2:
        cov_47 = st.number_input(
            "CoV 47",
            min_value=0.0,
            value=0.0
        )

    # --------------------------------------------------------
    # WORKING MEMORY
    # --------------------------------------------------------

    st.markdown("### 🧠 Working Memory")

    wm_k = st.number_input(
        "Working Memory K",
        min_value=0.0,
        value=0.0
    )

    # --------------------------------------------------------
    # CORSI / SPATIAL MEMORY
    # --------------------------------------------------------

    st.markdown("### 🧩 Visuo-Spatial Memory")

    col1, col2, col3 = st.columns(3)

    with col1:
        avanti = st.number_input(
            "Avanti",
            min_value=0.0,
            value=0.0
        )

    with col2:
        indietro = st.number_input(
            "Indietro",
            min_value=0.0,
            value=0.0
        )

    with col3:
        average_corsi = st.number_input(
            "Average Corsi",
            min_value=0.0,
            value=0.0
        )

    # --------------------------------------------------------
    # SYMBOLIC COMPARISON
    # --------------------------------------------------------

    st.markdown("### 🔤 Symbolic Comparison")

    col1, col2 = st.columns(2)

    with col1:
        accuracy_symbolic = st.number_input(
            "Symbolic Comparison Accuracy",
            min_value=0.0,
            max_value=1.0,
            value=0.5
        )

    with col2:
        rts_symbolic = st.number_input(
            "Symbolic Comparison Reaction Time",
            min_value=0.0,
            value=1000.0
        )

    # --------------------------------------------------------
    # EXTRA TASK FEATURES
    # --------------------------------------------------------

    st.markdown("### 📋 Additional Task Performance")

    col1, col2 = st.columns(2)

    with col1:
        enumeration_rts = st.number_input(
            "Enumeration Reaction Time",
            min_value=0.0,
            value=1000.0
        )

        enumeration_accuracy = st.number_input(
            "Enumeration Accuracy",
            min_value=0.0,
            max_value=1.0,
            value=0.5
        )

    with col2:
        comparison_rts = st.number_input(
            "Comparison Reaction Time",
            min_value=0.0,
            value=1000.0
        )

        comparison_accuracy = st.number_input(
            "Comparison Accuracy",
            min_value=0.0,
            max_value=1.0,
            value=0.5
        )

    # --------------------------------------------------------
    # VISUOSPATIAL TASK FEATURES
    # --------------------------------------------------------

    st.markdown("### 🧠 Visuo-Spatial Task")

    col1, col2 = st.columns(2)

    with col1:
        visuospatial_rts = st.number_input(
            "Visuo-Spatial Reaction Time",
            min_value=0.0,
            value=1000.0
        )

    with col2:
        visuospatial_accuracy = st.number_input(
            "Visuo-Spatial Accuracy",
            min_value=0.0,
            max_value=1.0,
            value=0.5
        )

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🔍 Run Dyscalculia Screening",
        type="primary"
    ):

        model = load_dyscalculia_model()

        # IMPORTANT:
        # These feature names MUST exactly match
        # the features used during model training.

        input_data = pd.DataFrame([{

            "Similarities_IQ": similarities_iq,
            "MatrixReas_IQ": matrixreas_iq,

            "Mean_ACC_ANS": mean_acc_ans,
            "Mean_RTs_ANS": mean_rts_ans,
            "Mean_RTs_def": mean_rts_def,

            "Mean_ACC_13": mean_acc_13,
            "Mean_ACC_47": mean_acc_47,

            "MeanRT_13": meanrt_13,
            "MeanRT_47": meanrt_47,

            "EfficiencyScore_13": efficiency_13,
            "EfficiencyScore_47": efficiency_47,

            "CoV_13": cov_13,
            "CoV_47": cov_47,

            "wm_K": wm_k,

            "Avanti": avanti,
            "Indietro": indietro,
            "Average_Corsi": average_corsi,

            "Accuracy_SymbolicComp": accuracy_symbolic,
            "RTs_SymbolicComp": rts_symbolic,

            "Enumeration_RTs": enumeration_rts,
            "Enumeration_Accuracy": enumeration_accuracy,

            "Comparison_RTs": comparison_rts,
            "Comparison_Accuracy": comparison_accuracy,

            "Visuospatial_RTs": visuospatial_rts,
            "Visuospatial_Accuracy": visuospatial_accuracy

        }])

        try:

            prediction = model.predict(
                input_data
            )[0]

            probabilities = model.predict_proba(
                input_data
            )[0]

            confidence = max(probabilities) * 100

            st.divider()

            st.subheader("🔍 AI Screening Result")

            if prediction == 1:

                st.error(
                    "AI Result: Dyscalculia Pattern Detected"
                )

            else:

                st.success(
                    "AI Result: Control / No Dyscalculia Pattern Detected"
                )

            st.metric(
                "AI Confidence",
                f"{confidence:.2f}%"
            )

            st.divider()

            st.subheader("📊 Prediction Probabilities")

            st.write(
                f"Dyscalculia Probability: "
                f"{probabilities[1] * 100:.2f}%"
            )

            st.progress(
                min(probabilities[1], 1.0)
            )

            st.write(
                f"Control Probability: "
                f"{probabilities[0] * 100:.2f}%"
            )

            st.progress(
                min(probabilities[0], 1.0)
            )

            st.info(
                "This result is an AI model prediction based "
                "on the entered performance features. It is "
                "not a medical diagnosis."
            )

        except Exception as e:

            st.error(
                "There was an error while making the prediction."
            )

            st.code(str(e))

