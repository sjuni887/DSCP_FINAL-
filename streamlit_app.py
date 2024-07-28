import streamlit as st
import pandas as pd
import os

# Import tabs
from generative_ai_chat import generative_ai_chat
from risk_calculator import risk_calculator
from patient_data import patient_data
from data_visualization import data_visualization

# File path for storing the patient data CSV
csv_file_path = 'patient_data.csv'

# Load existing patient data if available
if os.path.exists(csv_file_path):
    st.session_state.patient_data = pd.read_csv(csv_file_path)
    st.session_state.patient_data['IndexNo'] = st.session_state.patient_data['IndexNo'].astype(str)
else:
    st.session_state.patient_data = pd.DataFrame(columns=[
        'IndexNo', 'Age', 'RCRI score', 'Anemia category', 'PreopEGFRMDRD', 'Grade of Kidney Disease',
        'Preoptransfusion within 30 days', 'Intraop', 'Postop within 30 days',
        'Transfusion intra and postop', 'Transfusion Intra and Postop Category', 'Surgical Risk Category',
        'Grade of Kidney Category', 'Anemia Category Binned', 'RDW15.7', 'ASA Category Binned',
        'Gender', 'Anaesthesia Type', 'Surgery Priority', 'Race', 'Creatine RCRI Category',
        'DM Insulin Category', 'CHF RCRI Category', 'IHD RCRI Category', 'CVA RCRI Category',
        'Death Prediction', 'Death Probability', 'ICU Prediction', 'ICU Probability'
    ])

def main():
    st.title('Healthcare Dashboard')

    # Define tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Generative AI Chat", "Risk Calculator", "Patient Data", "Data Visualization"])

    with tab1:
        generative_ai_chat()
    with tab2:
        risk_calculator()
    with tab3:
        patient_data()
    with tab4:
        data_visualization()

if __name__ == "__main__":
    main()
