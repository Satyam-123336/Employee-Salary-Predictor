import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

model = joblib.load("model/best_model.pkl")
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.markdown("""
    <style>
    .main {
    background-color: #121212;  
    color: #f5f5f5;             
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5em 2em;
    }
    </style>
""", unsafe_allow_html=True)
st.title("Employee Salary Predictor")
st.write("Predict whether an employee earns **>50K or <=50K** based on their input details.")
def user_input_form():
    st.sidebar.header("Input Features")
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", value=100000)
    educational_num = st.sidebar.number_input("Educational Number", min_value=1, max_value=16, value=10)
    capital_gain = st.sidebar.number_input("Capital Gain", value=0)        
    capital_loss = st.sidebar.number_input("Capital Loss", value=0)
    hours_per_week = st.sidebar.number_input("Hours per Week", value=40)
    with st.form("Input_form"):
        st.subheader("Enter Employee Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            workclass = st.selectbox("Workclass", [
                'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                'Local-gov', 'State-gov'
            ])

            education = st.selectbox("Education", [
                'Bachelors', 'Some-college', 'HS-grad', 'Prof-school',
                'Assoc-acdm', 'Assoc-voc','Masters', 'Doctorate'
            ])

            marital_status = st.selectbox("Marital Status", [
                'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                'Married-spouse-absent', 'Married-AF-spouse'
            ])
        with col2:
            occupation = st.selectbox("Occupation", [
                'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                'Armed-Forces'
            ])
            relationship = st.selectbox("Relationship", [
                'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'
            ])
            race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
        with col3:
            gender = st.selectbox("Gender", ['Male', 'Female'])
            native_country = st.selectbox("Native Country", [
                'United-States', 'Mexico', 'Philippines', 'Germany', 'Canada',
                'India', 'England', 'Cuba'
            ])
        submitted = st.form_submit_button("Predict Income")
        if submitted:
            data = {
                'age': age,
                'workclass': workclass,
                'fnlwgt': fnlwgt,
                'education': education,
                'educational-num': educational_num,
                'marital-status': marital_status,
                'occupation': occupation,
                'relationship': relationship,
                'race': race,
                'gender': gender,
                'capital-gain': capital_gain,
                'capital-loss': capital_loss,
                'hours-per-week': hours_per_week,
                'native-country': native_country
            }
            return pd.DataFrame([data])
    return None
input_df = user_input_form()
if input_df is not None:
    probs = model.predict_proba(input_df)[0]
    predicted_index = np.argmax(probs)
    label = model.classes_[predicted_index]
    confidence = round(probs[predicted_index] * 100, 2)
    st.markdown("### Model Confidence")
    st.markdown(f"""
    <div style="text-align:center; padding:30px; background-color:#ffffff; border-radius:15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h2 style="color:#007bff;">Predicted Income</h2>
        <h1 style="font-size:48px; color:#28a745;">{label}</h1>
        <p style="font-size:18px; color:#333;">Confidence: <b>{confidence}%</b></p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("### Prediction Confidence")
    import matplotlib.pyplot as plt
    labels = ['<=50K', '>50K']
    colors = ['#9b5de5', '#00f5d4']
    fig, ax = plt.subplots()
    bars = ax.bar(labels, probs, color=colors)
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{prob:.2%}', ha='center', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)
st.markdown("---")

st.header("Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])
if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)
        required_cols = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status',
                         'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
                         'hours-per-week', 'native-country']
        if not all(col in batch_df.columns for col in required_cols):
            st.error("Uploaded CSV is missing one or more required columns...")
        else:
            batch_pred = model.predict(batch_df)
            batch_proba = model.predict_proba(batch_df)
            batch_df['Predicted Income'] = ['>50K' if p == 1 else '<=50K' for p in batch_pred]
            batch_df['Confidence (%)'] = [round(100 * max(p), 2) for p in batch_proba]
            st.success("Batch prediction completed successfully!")
            st.write("### Preview of Results")
            st.dataframe(batch_df)
            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions CSV", csv, "batch_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error reading file: {e}")

