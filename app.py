import streamlit as st
import pandas as pd
import joblib
from textblob import TextBlob

# Load model and preprocessing tools
model = joblib.load("saved_models/stacked_model.pkl")
le_gender = joblib.load("saved_models/le_gender.pkl")
le_reflections = joblib.load("saved_models/le_reflection.pkl")
le_mood = joblib.load("saved_models/le_mood.pkl")
scaler = joblib.load("saved_models/minmax_scaler.pkl")

# Streamlit UI
st.markdown(
    """
    <style>
        .fixed-header {
            position: fixed;
            top: 3.5rem;  /* pushes it below Streamlit's top menu */
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

        .spacer {
            margin-top: 120px;  /* adjust based on navbar height + Streamlit top bar */
        }
    </style>

    <div class="fixed-header">
        <h1>🧠 Student's Mental Health Predictor</h1>
    </div>
    <div class="spacer"></div>
    """,
    unsafe_allow_html=True
)




# st.markdown(
#     """
#     <div style='background-color:#004080; padding:15px; border-radius:10px;'>
#         <h1 style='color:white; text-align:center;'>🧠 Student's Mental Health Predictor</h1>
#     </div>
#     <br>
#     """,
#     unsafe_allow_html=True
# )

st.markdown("Predict mental health status using academic and emotional inputs.")


st.sidebar.markdown("## 🧑 Student Information")
student_name = st.sidebar.text_input("Student Name")
student_id = st.sidebar.text_input("Student ID")

st.sidebar.header("📋 Input Student Data")

# Dropdowns using encoder classes
gender = st.sidebar.selectbox("Gender", le_gender.classes_)
# reflections = st.sidebar.selectbox("Daily Reflections", le_reflections.classes_)
reflection_text = st.text_area("Write your reflection for the day")
mood = st.sidebar.selectbox("Mood Description", le_mood.classes_)

# convert reflections to sentiment score
if reflection_text:
    sentiment = TextBlob(reflection_text).sentiment.polarity  # range -1 to +1
    # Normalize to 0–10 scale (similar to what your model expects)
    reflection_score = (sentiment + 1) * 5  # because (-1 to 1) -> (0 to 10)
else:
    reflection_score = 5.0  # Neutral fallback


# Other inputs
# student_id = st.sidebar.text_input("Student ID", "S001")
age = st.sidebar.number_input("Age", 10, 100, 21)
gpa = st.sidebar.slider("GPA", 0.0, 4.0, 3.0, 0.1)
stress = st.sidebar.slider("Stress Level", 1, 10, 5)
anxiety = st.sidebar.slider("Anxiety Score", 1, 10, 5)
depression = st.sidebar.slider("Depression Score", 1, 10, 5)
sleep = st.sidebar.slider("Sleep Hours", 0, 24, 7)
steps = st.sidebar.number_input("Steps Per Day", 0, 100000, 5000)
sentiment = st.sidebar.slider("Sentiment Score", -1.0, 1.0, 0.0, 0.01)

# Encode categorical inputs
gender_enc = le_gender.transform([gender])[0]
# reflections_enc = le_reflections.transform([reflections])[0]
mood_enc = le_mood.transform([mood])[0]

# Build DataFrame in exact model order
input_data = pd.DataFrame([{
    # 'Student_ID': student_id,
    'Age': age,
    'Gender': gender_enc,
    'GPA': gpa,
    'Stress_Level': stress,
    'Anxiety_Score': anxiety,
    'Depression_Score': depression,
    'Daily_Reflections': reflection_score,
    'Sleep_Hours': sleep,
    'Steps_Per_Day': steps,
    'Mood_Description': mood_enc,
    'Sentiment_Score': sentiment
}])

# Correct any duplicate columns in scaler
scaler_columns = list(dict.fromkeys(scaler.feature_names_in_))  # removes duplicates safely

# Apply scaler only to numeric columns
scaled_values = scaler.transform(input_data[scaler_columns])
scaled_df = pd.DataFrame(scaled_values, columns=scaler_columns, index=input_data.index)

# Replace original values with scaled
input_data[scaler_columns] = scaled_df

#Predict
if st.button("🔮 Predict Mental Health"):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        prediction = model.predict(input_data)

        # Friendly labels
        mental_health_labels = {
            0: "Healthy",
            1: "Mild Mental Health Issues",
            2: "Severe Mental Health Risk"
        }

        pred_label = mental_health_labels.get(prediction[0], "Unknown")
        st.success(f"🧠 Mental Health Status: **{pred_label}**")
        # Save to file
        result_dict = {
            "Timestamp": timestamp,
            "Student_ID": student_id,
            "Student_Name": student_name,
            "Mental_Health_Status": pred_label,
            "Age": age,
            "Gender": gender,  # use original
            "GPA": gpa,
            "Stress_Level": stress,
            "Anxiety_Score": anxiety,
            "Depression_Score": depression,
            "Daily_Reflections": reflection_text,  # use original
            "Sleep_Hours": sleep,
            "Steps_Per_Day": steps,
            "Mood_Description": mood,  # use original
            "Sentiment_Score": sentiment
        }


        # Append to CSV file
        log_df = pd.DataFrame([result_dict])
        log_df.to_csv("predictions_log.csv", mode="a", header=not pd.io.common.file_exists("predictions_log.csv"), index=False)


    except Exception as e:
        st.error(f"❌ Prediction failed:\n\n{e}")

    # Show buttons after prediction
st.markdown("---")
st.subheader("📁 Prediction Logs")

# Button to display logs
if st.button("📜 Show Saved Logs"):
    try:
        logs = pd.read_csv("predictions_log.csv")
        st.dataframe(logs)
    except FileNotFoundError:
        st.warning("⚠️ No logs found yet.")

import os

# Buttons for Save and Delete side by side
col1, col2 = st.columns([1, 1])

with col1:
    try:
        logs = pd.read_csv("predictions_log.csv")
        st.download_button("💾 Save Logs to File", logs.to_csv(index=False), file_name="mental_health_predictions.csv", mime="text/csv")
    except FileNotFoundError:
        st.warning("⚠️ No logs found yet to save.")

with col2:
    if st.button("🗑️ Delete Logs"):
        try:
            os.remove("predictions_log.csv")
            st.success("🗑️ Log file deleted successfully.")
        except FileNotFoundError:
            st.warning("⚠️ No log file found to delete.")
        except Exception as e:
            st.error(f"An error occurred while deleting: {e}")
            st.warning("⚠️ No logs found yet to save.")

