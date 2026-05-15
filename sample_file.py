import streamlit as st
import pandas as pd
import joblib
from function_file import standardize_columns, normalize_data, one_hot_encoding
import time


# 1. Load the saved model
model = joblib.load('tech_survey_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

try:
    model = joblib.load('tech_survey_model.pkl')
    # Check if the model has the 'classes_' attribute (only exists if fitted)
    if hasattr(model, "classes_"):
        pass
    else:
        st.error("The loaded model file is empty/not trained.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="Mental Health Survey",
    page_icon="🧠",
    layout="centered"
)

# 1. Apply Custom CSS to match your PDF Theme
st.markdown("""
    <style>
    /* Side bar (The Dark Blue Divider) */
    [data-testid="stSidebar"] {
        background-color: #3d654c !important; /* Dark Blue */
        color: white;
    }

    /* ---------- Section headers ---------- */
    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 35px;
        margin-bottom: 15px;
        color: var(--text-color);
    }
            
    
    /* ---------- Card / Question Box ---------- */
    .question-box {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 18px;
        margin-bottom: 18px;
        border: 1px solid rgba(120,120,120,0.25);
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        color: var(--text-color);
    }
    /* Main Background (The Faint Blue) */
    .stApp {
        background-color: #CDE9E8; /* Faint green  */
    }

    /* Adjusting text colors for readability */
    [data-testid="stSidebar"] .css-17l2qt2 {
        color: white;
    }
    
    h1, h2, h3, p {
        color: #002366; /* Keep text dark blue for contrast */
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Left Side Content (Sidebar)
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/ffffff/brain.png", width=30)
    #st.title("Project Details")
    st.markdown("---")
    st.subheader("📊 **Model:** RandomForestClassifier")
    st.subheader("🎯 **Goal:** Early Stress Detection")
    st.subheader("🏢 **Sector:** Workplace Wellness")


# 3. Main Body Content
st.title("Mental Health Treatment?")
st.subheader("Predicting Well-being through Machine Learning")

# ------------------------------
# SURVEY QUESTIONS
# ------------------------------

st.markdown("<div class='section-title'>Personal Details</div>", unsafe_allow_html=True)
name = st.text_input("Your Name")
gender = st.selectbox("Gender", ["Female", "Male", "Non-binary", "Prefer not to say"])

st.write("---")

st.markdown("### Psychological Questions")

def ask_question(text):
    st.markdown(f"<div class='question-box'>{text}</div>", unsafe_allow_html=True)
    #return st.slider("Select intensity:", 1, 5, 3, key=text)
    return text
    st.caption("1 = Very low   •   3 = Moderate   •   5 = Very high")


# 1. Work Interference (The most important question)
# We use a selectbox because the categories are specific
work_interfere_label = "How often does your mental health interfere with your work?"
work_interfere_text = st.selectbox(
    work_interfere_label,
    ["Never", "Rarely", "Sometimes", "Often"]
)

# 2. Family History
family_history_text = st.selectbox("Do you have a family history of mental illness?", ["No", "Yes"])

# 3. Medical Leave
leave_label = "How easy is it for you to take medical leave for a mental health condition?"
leave_text = st.select_slider(
    leave_label,
    options=['Very difficult', 'Somewhat difficult', "Don't know", 'Somewhat easy', 'Very easy']
)

# 4. Benefits & Care Options
benefits_text = st.selectbox("Does your employer provide mental health benefits?", ["No", "Yes", "Don't know"])

# 5. Workplace Stigma (Coworkers & Supervisor)
supervisor_text = st.selectbox("Would you be willing to discuss a mental health issue with your direct supervisor?", ["No", "Some of them", "Yes"])

coworkers_text = st.selectbox("Would you be willing to discuss a mental health issue with your coworkers?", ["No", "Some of them", "Yes"])

# 6. Physical Health Consequence
phys_health_text = st.selectbox("Do you feel that discussing a physical health issue with your employer would have negative consequences?", ["No", "Maybe", "Yes"])

# 7. Age (Numeric)
age = st.number_input("What is your age?", min_value=18, max_value=100, value=25)

st.write("---")

# ------------------------------
# SAVE & SHOW RESULT
# ------------------------------
if st.button("Predict"):
    # Organize the inputs into a list that matches your training columns
    # IMPORTANT: The order MUST be the same as your X_train.columns
    input_data = pd.DataFrame({
        'phys_health_consequence': [phys_health_text],
        'coworkers': [coworkers_text],
        'family_history': [family_history_text],
        'benefits': [benefits_text],
        'Age': [age],
        'work_interfere': [work_interfere_text]
     })
   # Clean the strings to numbers
    #cleaned_df = standardize_columns(input_data)
    with st.spinner("Processing..."):
        time.sleep(1)
    cleaned_df = one_hot_encoding(input_data, encoder)
    st.success("Encoding of data done!!")

    with st.spinner("Processing..."):
        time.sleep(2)
    try:
        # Since 'input_data' already has numbers from your UI logic, 
        # we can go straight to normalization
        normalized_df = normalize_data(cleaned_df, scaler)
        st.success("Normalization of data done!!")

        with st.spinner("Processing..."):
            time.sleep(2)
        # 3. Predict
        prediction = model.predict(normalized_df)
        st.success("Model ran for your data!!")
        with st.spinner("Processing..."):
            time.sleep(1)
        if prediction[0] == 1:
            message = f"Prediction: {name }, Seeking treatment is likely recommended."
            st.error(message)
            st.markdown(f"""
                ### Prediction Result
                **Name:** {name}

                Seeking treatment is likely recommended.
                """)
        else:
            message = f"Prediction: {name} , Seeking treatment is likely not required."
            st.success(message)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Check if your input_data columns match your Scaler's columns.")

st.write("---")
st.warning(
    "⚠️ **Disclaimer**: This tool is for educational and awareness purposes only. "
    "It does NOT provide medical advice or diagnosis. "
)