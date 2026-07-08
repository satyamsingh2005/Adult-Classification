#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib
import os


# In[2]:


st.set_page_config(page_title="Income Predictor", page_icon="💰", layout="centered")


# In[3]:


MODEL_PATH = "best_model.pkl"


# In[4]:


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file '{MODEL_PATH}' not found. "
            "Train it locally with train_model.py and place best_model.pkl "
            "in this same folder before deploying."
        )
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()


# In[6]:


st.title("💰 Adult Income Classifier")
st.write(
    "Predicts whether a person's annual income is **above or below $50K** "
    "based on census attributes, using a Gradient Boosting model "
    "(trained on the UCI Adult dataset)."
)


# In[7]:


with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 17, 90, 35)
        education_num = st.slider("Education level (years)", 1, 16, 10)
        hours_per_week = st.slider("Hours worked per week", 1, 99, 40)
        sex = st.selectbox("Sex", ["Male", "Female"])

    with col2:
        capital_gain = st.number_input("Capital gain", 0, 99999, 0)
        capital_loss = st.number_input("Capital loss", 0, 4356, 0)
        workclass = st.selectbox(
            "Workclass",
            ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
             "Local-gov", "State-gov", "Without-pay", "Never-worked"],
        )
        race = st.selectbox(
            "Race",
            ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
        )

    marital_status = st.selectbox(
        "Marital status",
        ["Married-civ-spouse", "Divorced", "Never-married", "Separated",
         "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
    )
    occupation = st.selectbox(
        "Occupation",
        ["Tech-support", "Craft-repair", "Other-service", "Sales",
         "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
         "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
         "Transport-moving", "Priv-house-serv", "Protective-serv",
         "Armed-Forces"],
    )
    relationship = st.selectbox(
        "Relationship",
        ["Husband", "Wife", "Own-child", "Not-in-family",
         "Other-relative", "Unmarried"],
    )

    submitted = st.form_submit_button("Predict income")

if submitted:
    row = pd.DataFrame([{
        "age": age,
        "education_num": education_num,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "workclass": workclass,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
    }])

    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]

    label = ">50K" if pred == 1 else "<=50K"
    st.subheader(f"Prediction: **{label}**")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(f"Model confidence (P(income > $50K)): {prob:.1%}")

st.divider()
st.caption(
    "Model: Gradient Boosting classifier trained on the UCI Adult Income "
    "dataset (ROC-AUC ≈ 0.93 on held-out test data)."
)


# In[ ]:




