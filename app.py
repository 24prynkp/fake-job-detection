import streamlit as st
import joblib

st.title("Fake Job Posting Detector")

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

job_text = st.text_area("Enter Job Description")

if st.button("Check Job"):
    if job_text.strip() == "":
        st.warning("Please enter job description.")
    else:
        vector_input = vectorizer.transform([job_text])
        prediction = model.predict(vector_input)

        if prediction[0] == 1:
            st.error("⚠ Fake Job Posting Detected")
        else:
            st.success("✅ Real Job Posting")