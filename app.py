import streamlit as st
import pandas as pd
import joblib
import os
from textblob import TextBlob

# ========================== PATH SETUP ==========================
# Base directory where this app.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "predictions_log.csv")

# Show sidebar hint only on first load
if 'show_sidebar_hint' not in st.session_state:
    st.session_state.show_sidebar_hint = True

# ========================== LOAD MODELS ==========================
model = joblib.load(os.path.join(BASE_DIR, "saved_models", "stacked_model.pkl"))
le_gender = joblib.load(os.path.join(BASE_DIR, "saved_models", "le_gender.pkl"))
le_reflections = joblib.load(os.path.join(BASE_DIR, "saved_models", "le_reflection.pkl"))
le_mood = joblib.load(os.path.join(BASE_DIR, "saved_models", "le_mood.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "saved_models", "minmax_scaler.pkl"))

# ========================== HEADER ==========================
st.markdown(
    """
    <style>
        .fixed-header {
            position: fixed;
            top: 3.5rem;
            left: 0;
            width: 100%;
            background-color: #004080;
            padding: 18px 0;
            z-index: 9999;
            text-align: center;
        }
        .fixed-header h1 {
            color: white;
            font-size: 24px;
            margin: 0;
        }
        .spacer { margin-top: 120px; }
    </style>

    <div class="fixed-header">
        <h1>🧠 Student's Mental Health Predictor</h1>
    </div>
    <div class="spacer"></div>
    """,
    unsafe_allow_html=True
)
st.markdown("Predict mental health status using academic and emotional inputs.")

# ========================== SIDEBAR INPUTS ==========================
if st.session_state.show_sidebar_hint:
    st.markdown(
        """
        <style>
            .sidebar-hint {
                position: fixed;
                top: 70px;
                left: 10px;
                color: white;
                padding: 6px 10px;
                border-radius: 8px;
                font-size: 14px;
                z-index: 9999;
                box-shadow: 0px 2px 5px rgba(0,0,0,0.3);
                animation: bounce 2s infinite;
            }
            @keyframes bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-5px); }
            }
            .arrow { display: block; font-size: 14px; text-align: center; margin-top: -10px; }
            @media (min-width: 768px) { .sidebar-hint { display: none; } }
        </style>
        <div class="sidebar-hint">
            <div class="arrow">🔼 Tap here</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.sidebar.markdown("## 🧑 Student Information")
student_name = st.sidebar.text_input("Student Name")
student_id = st.sidebar.text_input("Student ID")

st.sidebar.header("📋 Input Student Data")

# Dropdowns using encoder classes
gender = st.sidebar.selectbox("Gender", le_gender.classes_)
reflection_text = st.text_area("Write your reflection for the day")
mood = st.sidebar.selectbox("Mood Description", le_mood.classes_)

# Sentiment from reflection text
if reflection_text:
    sentiment = TextBlob(reflection_text).sentiment.polarity
    reflection_score = (sentiment + 1) * 5
    reflection_sentiment = sentiment
else:
    reflection_score = 5.0
    reflection_sentiment = 0.0  

# Other inputs
age = st.sidebar.number_input("Age", 10, 100, 21)
gpa = st.sidebar.slider("GPA", 0.0, 5.0, 3.0, 0.1)
stress = st.sidebar.slider("Stress Level", 1, 10, 5)
anxiety = st.sidebar.slider("Anxiety Score", 1, 10, 5)
depression = st.sidebar.slider("Depression Score", 1, 10, 5)
sleep = st.sidebar.slider("Sleep Hours", 0, 24, 7)
steps = st.sidebar.number_input("Steps Per Day", 0, 100000, 5000)
sentiment = st.sidebar.slider("Sentiment Score", -1.0, 1.0, 0.0, 0.01)

# Convert GPA from 5-point scale to 4-point scale
converted_gpa = (gpa / 5.0) * 4.0

# Encode categorical inputs
gender_enc = le_gender.transform([gender])[0]
mood_enc = le_mood.transform([mood])[0]

# Build input dataframe
input_data = pd.DataFrame([{
    'Age': age,
    'Gender': gender_enc,
    'GPA': converted_gpa,
    'Stress_Level': stress,
    'Anxiety_Score': anxiety,
    'Depression_Score': depression,
    'Daily_Reflections': reflection_score,
    'Sleep_Hours': sleep,
    'Steps_Per_Day': steps,
    'Mood_Description': mood_enc,
    'Sentiment_Score': sentiment,
    'Reflection_Sentiment': reflection_sentiment
}])

# Apply scaler
scaler_columns = list(dict.fromkeys(scaler.feature_names_in_))
scaled_values = scaler.transform(input_data[scaler_columns])
scaled_df = pd.DataFrame(scaled_values, columns=scaler_columns, index=input_data.index)
input_data[scaler_columns] = scaled_df

# ========================== PREDICTION ==========================
if st.button("🔮 Predict Mental Health"):
    st.session_state.show_sidebar_hint = False
    if not student_name.strip() or not student_id.strip():
        st.warning("⚠️ Please enter both your *Name* and *Student ID* before predicting.")
    else:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            prediction = model.predict(input_data)

            mental_health_labels = {
                0: "Healthy",
                1: "At Risk",
                2: "Struggling"
            }
            pred_label = mental_health_labels.get(prediction[0], "Unknown")
            st.success(f"🧠 Mental Health Status: **{pred_label}**")

            # Save results to CSV log
            result_dict = {
                "Timestamp": timestamp,
                "Student_ID": student_id,
                "Student_Name": student_name,
                "Mental_Health_Status": pred_label,
                "Age": age,
                "Gender": gender,
                "GPA": gpa,
                "Stress_Level": stress,
                "Anxiety_Score": anxiety,
                "Depression_Score": depression,
                "Daily_Reflections": reflection_text,
                "Sleep_Hours": sleep,
                "Steps_Per_Day": steps,
                "Mood_Description": mood,
                "Sentiment_Score": sentiment
            }
            log_df = pd.DataFrame([result_dict])
            log_df.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)
        except Exception as e:
            st.error(f"❌ Prediction failed:\n\n{e}")

# ========================== LOGS SECTION ==========================
st.markdown("---")
st.subheader("📁 Prediction Logs")

# Show logs
if st.button("📜 Show Saved Logs"):
    try:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
            logs = pd.read_csv(LOG_FILE)
            st.dataframe(logs)
        else:
            st.warning("⚠️ No logs found yet.")
    except Exception as e:
        st.error(f"❌ Could not read logs:\n\n{e}")
col1, col2 = st.columns([1, 1])
# Save logs
with col1:
    try:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
            logs = pd.read_csv(LOG_FILE)
            st.download_button("💾 Save Logs to File",
                               logs.to_csv(index=False),
                               file_name="mental_health_predictions.csv",
                               mime="text/csv")
        else:
            st.warning("⚠️ No logs found yet to save.")
    except Exception as e:
        st.error(f"❌ Could not save logs:\n\n{e}")


# Delete logs
with col2:
    if st.button("🗑️ Delete Logs"):
        try:
            os.remove(LOG_FILE)
            st.success("🗑️ Log file deleted successfully.")
        except FileNotFoundError:
            st.warning("⚠️ No log file found to delete.")
        except Exception as e:
            st.error(f"An error occurred while deleting: {e}")
