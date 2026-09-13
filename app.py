import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ChurnAI - Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Overall background */
.stApp {
    background: linear-gradient(
        135deg,
        #eef2ff 0%,
        #f8fafc 45%,
        #fdf2f8 100%
    );
}

/* Main content width */
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}


/* Main title */
h1 {
    color: #1e1b4b;
    font-weight: 800;
    font-size: 44px;
}


/* Headings */
h2 {
    color: #312e81;
    font-weight: 750;
}

h3 {
    color: #4338ca;
}


/* Metric styling */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.85);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #e0e7ff;
    box-shadow: 0px 8px 25px rgba(79, 70, 229, 0.08);
}


/* Metric value */
[data-testid="stMetricValue"] {
    color: #312e81;
    font-weight: 750;
}


/* Input boxes */
div[data-baseweb="select"] > div {
    border-radius: 10px;
}

div[data-testid="stNumberInput"] input {
    border-radius: 10px;
}


/* Button */
.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(
        90deg,
        #4f46e5,
        #7c3aed,
        #db2777
    );
    color: white;
    font-size: 18px;
    font-weight: 700;
    box-shadow: 0px 8px 20px rgba(79, 70, 229, 0.25);
}


/* Button hover */
.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #4338ca,
        #6d28d9,
        #be185d
    );
    color: white;
}


/* Progress bar */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(
        90deg,
        #4f46e5,
        #db2777
    );
}


/* Alert boxes */
div[data-testid="stAlert"] {
    border-radius: 14px;
}


/* Divider */
hr {
    border-color: #ddd6fe;
}


/* Caption */
.stCaption {
    color: #64748b;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")


# =========================================================
# HEADER
# =========================================================

st.title("📊 ChurnAI — Customer Churn Prediction")

st.markdown(
    "### Predict whether a customer is likely to leave using Machine Learning."
)

st.caption(
    "Interactive ML application powered by Logistic Regression"
)

st.divider()


# =========================================================
# TOP INFORMATION
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🤖 Model",
        value="Logistic Regression"
    )

with col2:
    st.metric(
        label="📊 Input Features",
        value="13"
    )

with col3:
    st.metric(
        label="🎯 Task",
        value="Classification"
    )

with col4:
    st.metric(
        label="⚡ Prediction",
        value="Real-Time"
    )


st.divider()


# =========================================================
# CUSTOMER PROFILE
# =========================================================

st.header("👤 Customer Profile")

st.caption(
    "Enter the customer's personal and account information."
)


col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input(
        "💳 Credit Score",
        min_value=300,
        max_value=850,
        value=650
    )

with col2:
    age = st.number_input(
        "🎂 Age",
        min_value=18,
        max_value=100,
        value=35
    )

with col3:
    gender = st.selectbox(
        "⚥ Gender",
        ["Female", "Male"]
    )
    col1, col2, col3 = st.columns(3)

with col1:
    geography = st.selectbox(
        "🌍 Geography",
        ["France", "Germany", "Spain"]
    )

with col2:
    tenure = st.number_input(
        "📅 Tenure",
        min_value=0,
        max_value=10,
        value=5
    )

with col3:
    is_active = st.selectbox(
        "🟢 Active Member",
        ["Yes", "No"]
    )


st.divider()


# =========================================================
# ACCOUNT INFORMATION
# =========================================================

st.header("💰 Account & Financial Information")

st.caption(
    "Provide the customer's financial and product-related details."
)


col1, col2, col3 = st.columns(3)

with col1:
    balance = st.number_input(
        "💰 Account Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

with col2:
    num_products = st.number_input(
        "📦 Number of Products",
        min_value=1,
        max_value=4,
        value=1
    )

with col3:
    has_cr_card = st.selectbox(
        "💳 Has Credit Card",
        ["Yes", "No"]
    )


col1, col2 = st.columns(2)

with col1:
    estimated_salary = st.number_input(
        "💵 Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )


st.divider()


# =========================================================
# PREPARE INPUT
# =========================================================

input_data = pd.DataFrame({

    "CreditScore": [credit_score],

    "Age": [age],

    "Tenure": [tenure],

    "Balance": [balance],

    "NumOfProducts": [num_products],

    "HasCrCard": [
        1 if has_cr_card == "Yes" else 0
    ],

    "IsActiveMember": [
        1 if is_active == "Yes" else 0
    ],

    "EstimatedSalary": [estimated_salary],

    "Geography_France": [
        1 if geography == "France" else 0
    ],

    "Geography_Germany": [
        1 if geography == "Germany" else 0
    ],

    "Geography_Spain": [
        1 if geography == "Spain" else 0
    ],

    "Gender_Female": [
        1 if gender == "Female" else 0
    ],

    "Gender_Male": [
        1 if gender == "Male" else 0
    ]
})


# =========================================================
# PREDICTION
# =========================================================

st.header("🔮 Customer Churn Prediction")

st.caption(
    "Click the button to generate a real-time prediction."
)

predict = st.button(
    "🚀 Predict Customer Churn"
)


if predict:

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Probability
    probability = model.predict_proba(input_scaled)[0][1]

    probability_percent = probability * 100


    # =====================================================
    # CHURN
    # =====================================================

    if prediction[0] == 1:

        st.error(
            "⚠️ Customer is likely to churn."
        )

        st.subheader(
            f"Churn Probability: {probability_percent:.1f}%"
        )

        st.progress(
            probability
        )

        st.warning(
            "💡 This customer may require additional attention "
            "to reduce the risk of churn."
        )


    # =====================================================
    # STAY
    # =====================================================

    else:

        st.success(
            "✅ Customer is likely to stay."
        )

        st.subheader(
            f"Churn Probability: {probability_percent:.1f}%"
        )

        st.progress(
            probability
        )

        st.info(
            "💡 The model currently predicts a lower likelihood "
            "of customer churn."
        )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.header("🧠 About the Model")

col1, col2 = st.columns(2)

with col1:

    st.info(
        """
        Algorithm

        Logistic Regression
        Used for binary classification to predict
        whether a customer is likely to churn.
        """
    )


with col2:

    st.info(
        """
        Prediction Pipeline

        Customer Input → Data Preparation →
        Standard Scaling → Logistic Regression →
        Churn Prediction
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Built with Python • Pandas • Scikit-learn • Logistic Regression • Streamlit"
)

st.caption(
    "Customer Churn Prediction | Machine Learning Portfolio Project"
)
# =========================================================
# EXTRA PREMIUM DESIGN
# =========================================================

st.markdown("""
<style>

/* Make main headings larger and stronger */
h1 {
    font-size: 48px !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2 {
    font-size: 30px !important;
    font-weight: 750 !important;
    color: #312e81 !important;
}

h3 {
    font-size: 23px !important;
    font-weight: 700 !important;
}


/* Make normal text easier to read */
p {
    font-size: 16px;
}


/* Make input labels larger */
label {
    font-size: 15px !important;
    font-weight: 600 !important;
}


/* Make metric values larger */
[data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: 800 !important;
}


/* Darker metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        #1e1b4b,
        #312e81
    ) !important;

    border: 1px solid #4338ca !important;
    border-radius: 18px !important;

    box-shadow:
        0 10px 25px rgba(30, 27, 75, 0.20) !important;
}


/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #c7d2fe !important;
}


/* Metric values */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
}


/* Make select boxes slightly bigger */
div[data-baseweb="select"] {
    font-size: 16px !important;
}


/* Make number input text bigger */
input {
    font-size: 16px !important;
}


/* Prediction button */
.stButton > button {
    font-size: 20px !important;
    font-weight: 800 !important;
    letter-spacing: 0.3px;
    height: 62px !important;

    background: linear-gradient(
        90deg,
        #312e81,
        #4f46e5,
        #7c3aed,
        #be185d
    ) !important;

    border-radius: 16px !important;

    box-shadow:
        0 10px 30px rgba(79, 70, 229, 0.30) !important;
}


/* Button hover */
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow:
        0 15px 35px rgba(79, 70, 229, 0.40) !important;
}


/* Alerts */
div[data-testid="stAlert"] {
    font-size: 16px !important;
    border-radius: 15px !important;
}


/* Progress bar height */
div[data-testid="stProgress"] {
    height: 12px !important;
}


/* Add more spacing around sections */
[data-testid="stVerticalBlock"] {
    gap: 0.8rem;
}


/* Footer text */
.stCaption {
    font-size: 14px !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PROJECT CREATOR
# =========================================================

st.divider()

st.markdown("### 👩‍💻 Project Creator")

st.info(
    """
    Customer Churn Prediction

    An interactive Machine Learning application built using
    Python, Pandas, Scikit-learn and Streamlit.

    Created by: Khushi Meena

    🎯 Machine Learning • 📊 Data Science • 🐍 Python
    """
)


# =========================================================
# FINAL PROJECT NOTE
# =========================================================

st.caption(
    "© 2026 Khushi Meena | Customer Churn Prediction ML Project"
)
# =========================================================
# PREMIUM PORTFOLIO SHOWCASE
# =========================================================

st.divider()

st.header("✨ Project Showcase")

st.markdown(
    """
    Welcome to ChurnAI, an interactive Machine Learning
    application designed to predict customer churn.

    This application takes customer information as input,
    processes the data using the same preprocessing pipeline
    used during model training, and generates a real-time
    churn prediction using Logistic Regression.
    """
)


# =========================================================
# PROJECT HIGHLIGHTS
# =========================================================

st.subheader("🚀 What This Application Does")

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("### 🎯 Predict")
        st.write(
            "Predict whether a customer is likely to churn "
            "based on their profile and account information."
        )

with c2:
    with st.container(border=True):
        st.markdown("### 📊 Analyze")
        st.write(
            "Use customer attributes such as age, credit score, "
            "balance, products and activity to generate predictions."
        )

with c3:
    with st.container(border=True):
        st.markdown("### ⚡ Respond")
        st.write(
            "Generate an immediate prediction and estimated "
            "churn probability from the trained ML model."
        )


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# MACHINE LEARNING PIPELINE
# =========================================================

st.header("🔬 Machine Learning Pipeline")

st.write(
    "The application follows a complete machine learning "
    "prediction workflow:"
)


pipeline1, pipeline2, pipeline3, pipeline4, pipeline5 = st.columns(5)

with pipeline1:
    st.markdown("### 01")
    st.markdown("📥")
    st.write("Customer Input")

with pipeline2:
    st.markdown("### 02")
    st.markdown("🧹")
    st.write("Data Preparation")

with pipeline3:
    st.markdown("### 03")
    st.markdown("⚖️")
    st.write("Feature Scaling")

with pipeline4:
    st.markdown("### 04")
    st.markdown("🤖")
    st.write("Logistic Regression")

with pipeline5:
    st.markdown("### 05")
    st.markdown("🎯")
    st.write("Prediction")


st.info(
    "Customer Input → Data Preparation → Standard Scaling → "
    "Logistic Regression → Churn Prediction"
)


# =========================================================
# PREDICTION INTERPRETATION
# =========================================================

st.header("📈 Understanding the Prediction")

st.write(
    "The application provides an estimated probability of "
    "customer churn along with the predicted class."
)


risk1, risk2, risk3 = st.columns(3)

with risk1:
    with st.container(border=True):
        st.markdown("### 🟢 Lower Risk")
        st.write(
            "A lower churn probability indicates that the "
            "customer is currently less likely to leave."
        )

with risk2:
    with st.container(border=True):
        st.markdown("### 🟡 Moderate Risk")
        st.write(
            "A moderate probability suggests that the "
            "customer may require closer attention."
        )

with risk3:
    with st.container(border=True):
        st.markdown("### 🔴 Higher Risk")
        st.write(
            "A higher churn probability indicates a greater "
            "likelihood that the customer may leave."
        )


st.caption(
    "Note: The probability is a model prediction and should "
    "be interpreted as an estimated risk rather than a certainty."
)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.header("🧠 Model & Data Information")

info1, info2 = st.columns(2)

with info1:

    with st.container(border=True):

        st.subheader("🤖 Machine Learning Model")

        st.write("Algorithm: Logistic Regression")
        st.write("Problem Type: Binary Classification")

        st.write(
            "Target: Customer Churn (Exited)"
        )

        st.write(
            "Output: Stay / Churn + Probability"
        )


with info2:

    with st.container(border=True):

        st.subheader("📊 Features Used")

        st.write("Credit Score")

        st.write("Age")

        st.write("Tenure")

        st.write("Balance")

        st.write("Number of Products")

        st.write("Credit Card Status")

        st.write("Active Member Status")

        st.write("Estimated Salary")

        st.write("Geography & Gender")


# =========================================================
# TECHNOLOGY STACK
# =========================================================

st.header("🛠️ Technology Stack")

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:
    with st.container(border=True):
        st.markdown("## 🐍")
        st.markdown("### Python")
        st.write("Programming & ML workflow")

with tech2:
    with st.container(border=True):
        st.markdown("## 🐼")
        st.markdown("### Pandas")
        st.write("Data preparation")

with tech3:
    with st.container(border=True):
        st.markdown("## 🤖")
        st.markdown("### Scikit-learn")
        st.write("Machine Learning")

with tech4:
    with st.container(border=True):
        st.markdown("## ⚡")
        st.markdown("### Streamlit")
        st.write("Interactive web application")


# =========================================================
# PROJECT JOURNEY
# =========================================================

st.header("📚 Project Journey")

st.write(
    "This project follows the complete journey from raw customer "
    "data to an interactive Machine Learning application."
)


with st.expander("🔎 Step 1 — Exploratory Data Analysis"):

    st.write(
        "The customer dataset was explored to understand "
        "its structure, distributions, relationships and "
        "important patterns."
    )


with st.expander("📊 Step 2 — Data Analysis"):

    st.write(
        "Univariate, bivariate and multivariate analysis "
        "were performed to understand customer behaviour "
        "and its relationship with churn."
    )

    with st.expander("⚙️ Step 3 — Preprocessing"):
        pass

    # ==============================
# FINAL DARK THEME
# ==============================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: #0B1120 !important;
}

/* Main content area */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* All normal text */
.stApp p,
.stApp label,
.stApp span {
    color: #E5E7EB;
}

/* Headings */
.stApp h1 {
    color: #F8FAFC !important;
    font-size: 46px !important;
    font-weight: 800 !important;
}

.stApp h2 {
    color: #C4B5FD !important;
    font-size: 30px !important;
    font-weight: 750 !important;
}

.stApp h3 {
    color: #A5B4FC !important;
    font-size: 23px !important;
    font-weight: 700 !important;
}

/* Input boxes */
div[data-baseweb="input"] {
    background-color: #1E293B !important;
    border: 1px solid #475569 !important;
    border-radius: 10px !important;
}

div[data-baseweb="input"] input {
    color: #F8FAFC !important;
    background-color: transparent !important;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background-color: #1E293B !important;
    border: 1px solid #475569 !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] * {
    color: #F8FAFC !important;
}

/* Number input buttons */
button {
    color: #E2E8F0 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #172554, #312E81) !important;
    border: 1px solid #4F46E5 !important;
    border-radius: 16px !important;
    padding: 18px !important;
}

[data-testid="stMetricLabel"] {
    color: #C7D2FE !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #4F46E5, #7C3AED) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    min-height: 52px !important;
    transition: 0.2s ease-in-out;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
    transform: translateY(-2px);
}

/* Info boxes */
[data-testid="stAlert"] {
    background-color: #172554 !important;
    color: #E0E7FF !important;
    border-radius: 12px !important;
}

/* Success message */
div[data-testid="stAlert"][kind="success"] {
    background-color: #052E16 !important;
}

/* Error message */
div[data-testid="stAlert"][kind="error"] {
    background-color: #450A0A !important;
}

/* Divider */
hr {
    border-color: #334155 !important;
}

/* Sidebar, if present */
section[data-testid="stSidebar"] {
    background-color: #0F172A !important;
}

</style>
""", unsafe_allow_html=True)