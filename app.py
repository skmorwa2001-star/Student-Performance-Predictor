import streamlit as st
import pandas as pd
import joblib

# Load Model

model= joblib.load("student_performance_model.pkl")


# Page config
st.set_page_config(
    page_title="Student Performace Predictor",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Student Performance Predictor")

st.write("Predict a student's final grade using academic and "
         "student-related factors.")

st.divider()


# Student inputs

st.subheader("📊 Enter Student Details")

col1, col2=st.columns(2)

with col1:
    G1= st.number_input(
        "G1 - First Period Grade",
        min_value=0,
        max_value=20,
        value=10
    )

    G2 =st.number_input(
        "G2 - Second Period Grade",
        min_value=0,
        max_value=20,
        value=10
    )
    studytime=st.number_input(
        "Study time",
        min_value=1,
        max_value=4,
        value=2
    )
    failures=st.number_input(
        "Past Failures",
        min_value=0,
        max_value=4,
        value=0
    )
    absences=st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=100,
        value=5
    )

with col2:

    freetime=st.number_input(
        "Free Time",
        min_value=1,
        max_value=5,
        value=3
    )
    goout=st.number_input(
        "Going Out with Friends",
        min_value=1,
        max_value=5,
        value=3
    )
    health=st.number_input(
        "Health Status",
        min_value=1,
        max_value=5,
        value=4
    )
    traveltime=st.number_input(
        "Travel Time",
        min_value=1,
        max_value=4,
        value=2
    )

# Prediction
st.divider()

if st.button("🔮 Predict Final Grade", use_container_width=True):
    input_data=pd.DataFrame({
        "G1":[G1],
        "G2":[G2],
        "studytime":[studytime],
        "failures":[failures],
        "absences":[absences],
        "freetime":[freetime],
        "goout":[goout],
        "health":[health],
        "traveltime":[traveltime]
    })

    # Match the exact features order used during training
    input_data=input_data[model.feature_names_in_]

    prediction=model.predict(input_data)[0]

    st.success(f"🚀 Predicted Final Grade: **{prediction:2f}/20**")

    # Performance category

    if prediction >=16:
        st.balloons()
        st.success("🌟 Excellent Performance")

    elif prediction >=12:
        st.info("👍 Good Performance")

    elif prediction >=10:
        st.warning("📚 Average Performance")

    else:
        st.error("⚠️ Needs Improvement")


# Model Information
# 
st.divider()

st.subheader("🤖 Model Information")

st.write("**Model :** Decision Tree Regressor")
st.write("**R^2 Score:** 0.8841")
st.write("**MAE :** 0.9933")
st.write("**RMSE :** 1.5726")

st.caption("The model predicts the final grade (G3) on a 0-20 scale.")