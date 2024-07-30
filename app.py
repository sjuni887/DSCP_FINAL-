import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time
import os
import replicate
from streamlit_mic_recorder import mic_recorder, speech_to_text
from PyPDF2 import PdfReader
import io
from pdpbox import pdp, info_plots
import matplotlib.pyplot as plt

favicon_path = "chansey.png"

# Set page configuration
st.set_page_config(page_title="Healthcare Dashboard", layout="wide",page_icon="chansey.png")

# File path for storing the patient data CSV
csv_file_path = 'patient_data.csv'

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ''
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to summarize text
def summarize_text(text, model, temperature=0.1, top_p=0.9, max_length=120, chunk_size=4000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for chunk in chunks:
        prompt = f"Please summarize the following text:\n\n{chunk}\n\nSummary:"
        output = replicate.run(model, input={"prompt": prompt, "temperature": temperature, "top_p": top_p, "max_length": max_length})
        summaries.append(''.join(output))
    final_summary = ' '.join(summaries)
    return final_summary

# Function to answer query about text
def answer_query_about_text(query, text, model, temperature=0.1, top_p=0.9, max_length=120):
    prompt = f"Please answer the following query based on the given text:\n\nText: {text}\n\nQuery: {query}\n\nAnswer:"
    output = replicate.run(model, input={"prompt": prompt, "temperature": temperature, "top_p": top_p, "max_length": max_length})
    return ''.join(output)

# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input, llm, temperature, top_p, max_length):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run(llm, input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ", "temperature": temperature, "top_p": top_p, "max_length": max_length})
    return output

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
        'Death Prediction', 'Death Probability', 'ICU Prediction', 'ICU Probability',
        'Income Category', 'Loneliness'
    ])

def main():
    st.title('Healthcare Dashboard')

    # Define sections in the sidebar
    with st.sidebar:
        st.title("Navigation")
        if st.button("🦙💬 Generative AI Chat"):
            st.session_state.page = "chat"
        if st.button("📊 Data Visualization"):
            st.session_state.page = "visualization"
        if st.button("📋 Patient Data"):
            st.session_state.page = "patient_data"
        if st.button("🧮 Risk Calculator"):
            st.session_state.page = "risk_calculator"

        st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')
        replicate_api = None
        api_token_error = False

        try:
            if 'REPLICATE_API_TOKEN' in st.secrets:
                st.success('API key already provided!', icon='✅')
                replicate_api = st.secrets['REPLICATE_API_TOKEN']
        except Exception as e:
            api_token_error = True

        if not replicate_api:
            replicate_api = st.text_input('Enter Replicate API token:', type='password')
            if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                st.success('Proceed to entering your prompt message!', icon='👉')

        if replicate_api and (not api_token_error or (replicate_api.startswith('r8_') and len(replicate_api) == 40)):
            os.environ['REPLICATE_API_TOKEN'] = replicate_api

            st.subheader('Models and parameters')
            selected_model = st.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
            if selected_model == 'Llama2-7B':
                llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
            elif selected_model == 'Llama2-13B':
                llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
            temperature = st.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
            max_length = st.slider('max_length', min_value=32, max_value=128, value=120, step=8)
            st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

            st.subheader("Upload PDF for Analysis")
            uploaded_file = st.file_uploader("📎", type="pdf", label_visibility="collapsed")
            if uploaded_file:
                text = extract_text_from_pdf(uploaded_file)
                with st.spinner("Summarizing PDF..."):
                    summary = summarize_text(text, llm, temperature, top_p, max_length)
                    st.session_state.messages.append({"role": "assistant", "content": f"Summary of uploaded PDF:\n\n{summary}"})

    if "page" not in st.session_state:
        st.session_state.page = "chat"

    if st.session_state.page == "chat":
        st.title("🦙💬 Llama 2 Chatbot")

        # Store LLM generated responses
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

        # Display or clear chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        def clear_chat_history():
            st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

        # User-provided prompt
        col1, col2 = st.columns([4, 1])
        with col1:
            prompt = st.chat_input(disabled=not replicate_api)
        with col2:
            audio_text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
        
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_llama2_response(prompt, llm, temperature, top_p, max_length)
                    st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    elif st.session_state.page == "visualization":
        st.header("Data Visualization")
        power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=7ca96589-22e3-465d-9f5c-af24869867b4&autoAuth=true&ctid=cba9e115-3016-4462-a1ab-a565cba0cdf1"
        st.markdown(f"""
            <iframe width="100%" height="700" src="{power_bi_url}" frameborder="0" allowfullscreen="true"></iframe>
        """, unsafe_allow_html=True)

    elif st.session_state.page == "patient_data":
        st.header("Patient Data")

        patient_tab = st.tabs(["Add New Patient", "Edit Patient Data", "View All Patient Data"])

        with patient_tab[0]:
            st.subheader("Add New Patient")
            with st.form(key='add_form'):
                general_info_col1, general_info_col2 = st.columns(2)
                with general_info_col1:
                    indexno = st.text_input('Index Number')
                    age = st.number_input('Age', min_value=0, max_value=120, step=1)
                    rcri_score = st.number_input('RCRI score', min_value=0, max_value=10, step=1)
                with general_info_col2:
                    gender = st.selectbox('Gender', ['Male', 'Female'])
                    race = st.selectbox('Race', ['Chinese', 'Indian', 'Malay', 'Others'])
                    surgery_priority = st.selectbox('Surgery Priority', ['Elective', 'Emergency'])
                    anesthesia_type = st.selectbox('Anaesthesia Type', ['GA', 'RA'])

                st.subheader("Anemia Related Features🩸")
                anemia_col1, anemia_col2 = st.columns(2)
                with anemia_col1:
                    anemia_category = st.selectbox('Anemia category', ['none', 'mild', 'moderate', 'severe'])
                    anemia_binned = st.selectbox('Anemia Category Binned', ['none', 'mild', 'moderate/severe'])
                with anemia_col2:
                    rdw_15_7 = st.selectbox('RDW15.7', ['<=15.7', '>15.7'])

                st.subheader("Kidney Related Features 🩺")
                kidney_col1, kidney_col2 = st.columns(2)
                with kidney_col1:
                    preop_egfr = st.number_input('PreopEGFRMDRD', min_value=0, max_value=1000, step=1)
                    grade_kidney_disease = st.selectbox('Grade of Kidney Disease', ['G1', 'G2', 'G3', 'G4', 'G5'])
                with kidney_col2:
                    grade_kidney_category = st.selectbox('Grade of Kidney Category', ['G1', 'G2', 'G3', 'G4', 'G5'])

                st.subheader("Transfusion Related Features 💉")
                transfusion_col1, transfusion_col2 = st.columns(2)
                with transfusion_col1:
                    preop_transfusion = st.number_input('Preoptransfusion within 30 days', min_value=0, max_value=10, step=1)
                with transfusion_col2:
                    transfusion_intra_postop = st.number_input('Transfusion intra and postop', min_value=0, max_value=10, step=1)
                    transfusion_category = st.selectbox('Transfusion Intra and Postop Category', ['0 units', '1 unit', '2 or more units'])

                st.subheader("Surgical and Postoperative Features 🏥")
                surg_postop_col1, surg_postop_col2 = st.columns(2)
                with surg_postop_col1:
                    intraop = st.number_input('Intraop', min_value=0, max_value=10, step=1)
                with surg_postop_col2:
                    postop_within_30days = st.number_input('Postop within 30 days', min_value=0, max_value=10, step=1)

                st.subheader("Risk Factors ⚠️")
                risk_col1, risk_col2 = st.columns(2)
                with risk_col1:
                    surg_risk_category = st.selectbox('Surgical Risk Category', ['Low', 'Moderate', 'High'])
                    asa_category_binned = st.selectbox('ASA Category Binned', ['I', 'II', 'III', 'IV-VI'])
                with risk_col2:
                    creatine_rcri = st.selectbox('Creatine RCRI Category', ['Yes', 'No'])
                    dm_insulin = st.selectbox('DM Insulin Category', ['Yes', 'No'])
                    chf_rcri = st.selectbox('CHF RCRI Category', ['Yes', 'No'])
                    ihd_rcri = st.selectbox('IHD RCRI Category', ['Yes', 'No'])
                    cva_rcri = st.selectbox('CVA RCRI Category', ['Yes', 'No'])

                st.subheader("SDOH (Social Determinants of Health) 🌐")
                sdoh_col1, sdoh_col2 = st.columns(2)
                with sdoh_col1:
                    income_category = st.selectbox('Income Category', ['Low Income', 'Median Income', 'High Income'])
                with sdoh_col2:
                    loneliness = st.selectbox('Loneliness', ['Low', 'Moderate', 'Severe'])

                submit_add_button = st.form_submit_button(label='Add Patient Data')

            if submit_add_button:
                if not indexno:
                    st.error("Index Number is required. Please enter a valid Index Number.")
                elif indexno in st.session_state.patient_data['IndexNo'].values:
                    st.error("This Index Number already exists. Please enter a new Index Number.")
                else:
                    new_data = pd.DataFrame([{
                        'IndexNo': indexno, 'Age': age, 'RCRI score': rcri_score, 'Anemia category': anemia_category, 'PreopEGFRMDRD': preop_egfr,
                        'Grade of Kidney Disease': grade_kidney_disease, 'Preoptransfusion within 30 days': preop_transfusion,
                        'Intraop': intraop, 'Postop within 30 days': postop_within_30days, 'Transfusion intra and postop': transfusion_intra_postop,
                        'Transfusion Intra and Postop Category': transfusion_category, 'Surgical Risk Category': surg_risk_category,
                        'Grade of Kidney Category': grade_kidney_category, 'Anemia Category Binned': anemia_binned,
                        'RDW15.7': rdw_15_7, 'ASA Category Binned': asa_category_binned, 'Gender': gender, 'Anaesthesia Type': anesthesia_type,
                        'Surgery Priority': surgery_priority, 'Race': race, 'Creatine RCRI Category': creatine_rcri, 'DM Insulin Category': dm_insulin,
                        'CHF RCRI Category': chf_rcri, 'IHD RCRI Category': ihd_rcri, 'CVA RCRI Category': cva_rcri,
                        'Death Prediction': 0, 'Death Probability': 0.0, 'ICU Prediction': 0, 'ICU Probability': 0.0,
                        'Income Category': income_category, 'Loneliness': loneliness
                    }])
                    st.session_state.patient_data = pd.concat([st.session_state.patient_data, new_data], ignore_index=True)
                    st.session_state.patient_data.to_csv(csv_file_path, index=False)
                    st.success("Patient data added successfully.")

        with patient_tab[1]:
            st.subheader("Edit Patient Data")
            indexno_edit = st.text_input("Enter Index Number to edit a patient:")
            if indexno_edit:
                patient_data = st.session_state.patient_data[st.session_state.patient_data['IndexNo'] == indexno_edit]
                if not patient_data.empty:
                    st.write("Patient Data:")
                    st.dataframe(patient_data)

                    with st.form(key='edit_form'):
                        general_info_col1, general_info_col2 = st.columns(2)
                        with general_info_col1:
                            indexno = patient_data['IndexNo'].values[0]
                            age = st.number_input('Age', value=int(patient_data['Age'].values[0]))
                            rcri_score = st.number_input('RCRI score', value=int(patient_data['RCRI score'].values[0]))
                        with general_info_col2:
                            gender = st.selectbox('Gender', ['Male', 'Female'], index=['Male', 'Female'].index(patient_data['Gender'].values[0]))
                            race = st.selectbox('Race', ['Chinese', 'Indian', 'Malay', 'Others'], index=['Chinese', 'Indian', 'Malay', 'Others'].index(patient_data['Race'].values[0]))
                            surgery_priority = st.selectbox('Surgery Priority', ['Elective', 'Emergency'], index=['Elective', 'Emergency'].index(patient_data['Surgery Priority'].values[0]))
                            anesthesia_type = st.selectbox('Anaesthesia Type', ['GA', 'RA'], index=['GA', 'RA'].index(patient_data['Anaesthesia Type'].values[0]))

                        st.subheader("Anemia Related Features🩸")
                        anemia_col1, anemia_col2 = st.columns(2)
                        with anemia_col1:
                            anemia_category = st.selectbox('Anemia category', ['none', 'mild', 'moderate', 'severe'], index=['none', 'mild', 'moderate', 'severe'].index(patient_data['Anemia category'].values[0]))
                            anemia_binned = st.selectbox('Anemia Category Binned', ['none', 'mild', 'moderate/severe'], index=['none', 'mild', 'moderate/severe'].index(patient_data['Anemia Category Binned'].values[0]))
                        with anemia_col2:
                            rdw_15_7 = st.selectbox('RDW15.7', ['<=15.7', '>15.7'], index=['<=15.7', '>15.7'].index(patient_data['RDW15.7'].values[0]))

                        st.subheader("Kidney Related Features 🩺")
                        kidney_col1, kidney_col2 = st.columns(2)
                        with kidney_col1:
                            preop_egfr = st.number_input('PreopEGFRMDRD', value=int(patient_data['PreopEGFRMDRD'].values[0]))
                            grade_kidney_disease = st.selectbox('Grade of Kidney Disease', ['G1', 'G2', 'G3', 'G4', 'G5'], index=['G1', 'G2', 'G3', 'G4', 'G5'].index(patient_data['Grade of Kidney Disease'].values[0]))
                        with kidney_col2:
                            grade_kidney_category = st.selectbox('Grade of Kidney Category', ['G1', 'G2', 'G3', 'G4', 'G5'], index=['G1', 'G2', 'G3', 'G4', 'G5'].index(patient_data['Grade of Kidney Category'].values[0]))

                        st.subheader("Transfusion Related Features 💉")
                        transfusion_col1, transfusion_col2 = st.columns(2)
                        with transfusion_col1:
                            preop_transfusion = st.number_input('Preoptransfusion within 30 days', value=int(patient_data['Preoptransfusion within 30 days'].values[0]))
                        with transfusion_col2:
                            transfusion_intra_postop = st.number_input('Transfusion intra and postop', value=int(patient_data['Transfusion intra and postop'].values[0]))
                            transfusion_category = st.selectbox('Transfusion Intra and Postop Category', ['0 units', '1 unit', '2 or more units'], index=['0 units', '1 unit', '2 or more units'].index(patient_data['Transfusion Intra and Postop Category'].values[0]))

                        st.subheader("Surgical and Postoperative Features 🏥")
                        surg_postop_col1, surg_postop_col2 = st.columns(2)
                        with surg_postop_col1:
                            intraop = st.number_input('Intraop', value=int(patient_data['Intraop'].values[0]))
                        with surg_postop_col2:
                            postop_within_30days = st.number_input('Postop within 30 days', value=int(patient_data['Postop within 30 days'].values[0]))

                        st.subheader("Risk Factors ⚠️")
                        risk_col1, risk_col2 = st.columns(2)
                        with risk_col1:
                            surg_risk_category = st.selectbox('Surgical Risk Category', ['Low', 'Moderate', 'High'], index=['Low', 'Moderate', 'High'].index(patient_data['Surgical Risk Category'].values[0]))
                            asa_category_binned = st.selectbox('ASA Category Binned', ['I', 'II', 'III', 'IV-VI'], index=['I', 'II', 'III', 'IV-VI'].index(patient_data['ASA Category Binned'].values[0]))
                        with risk_col2:
                            creatine_rcri = st.selectbox('Creatine RCRI Category', ['Yes', 'No'], index=['Yes', 'No'].index(patient_data['Creatine RCRI Category'].values[0]))
                            dm_insulin = st.selectbox('DM Insulin Category', ['Yes', 'No'], index=['Yes', 'No'].index(patient_data['DM Insulin Category'].values[0]))
                            chf_rcri = st.selectbox('CHF RCRI Category', ['Yes', 'No'], index=['Yes', 'No'].index(patient_data['CHF RCRI Category'].values[0]))
                            ihd_rcri = st.selectbox('IHD RCRI Category', ['Yes', 'No'], index=['Yes', 'No'].index(patient_data['IHD RCRI Category'].values[0]))
                            cva_rcri = st.selectbox('CVA RCRI Category', ['Yes', 'No'], index=['Yes', 'No'].index(patient_data['CVA RCRI Category'].values[0]))
                        
                        st.subheader("SDOH (Social Determinants of Health) 🌐")
                        sdoh_col1, sdoh_col2 = st.columns(2)
                        with sdoh_col1:
                            income_category = st.selectbox('Income Category', ['Low Income', 'Median Income', 'High Income'], index=['Low Income', 'Median Income', 'High Income'].index(patient_data['Income Category'].values[0]))
                        with sdoh_col2:
                            loneliness = st.selectbox('Loneliness', ['Low', 'Moderate', 'Severe'], index=['Low', 'Moderate', 'Severe'].index(patient_data['Loneliness'].values[0]))
                        
                        submit_edit_button = st.form_submit_button(label='Save Changes')

                    if submit_edit_button:
                        st.session_state.patient_data.loc[st.session_state.patient_data['IndexNo'] == indexno, [
                            'Age', 'RCRI score', 'Anemia category', 'PreopEGFRMDRD', 'Grade of Kidney Disease',
                            'Preoptransfusion within 30 days', 'Intraop', 'Postop within 30 days',
                            'Transfusion intra and postop', 'Transfusion Intra and Postop Category', 'Surgical Risk Category',
                            'Grade of Kidney Category', 'Anemia Category Binned', 'RDW15.7', 'ASA Category Binned',
                            'Gender', 'Anaesthesia Type', 'Surgery Priority', 'Race', 'Creatine RCRI Category',
                            'DM Insulin Category', 'CHF RCRI Category', 'IHD RCRI Category', 'CVA RCRI Category',
                            'Income Category', 'Loneliness']] = [
                            age, rcri_score, anemia_category, preop_egfr, grade_kidney_disease,
                            preop_transfusion, intraop, postop_within_30days, transfusion_intra_postop,
                            transfusion_category, surg_risk_category, grade_kidney_category,
                            anemia_binned, rdw_15_7, asa_category_binned, gender, anesthesia_type,
                            surgery_priority, race, creatine_rcri, dm_insulin, chf_rcri,
                            ihd_rcri, cva_rcri, income_category, loneliness
                        ]
                        st.session_state.patient_data.to_csv(csv_file_path, index=False)
                        st.success("Patient data updated successfully.")

        with patient_tab[2]:
            st.subheader("View All Patient Data")
            st.dataframe(st.session_state.patient_data)

    elif st.session_state.page == "risk_calculator":
        st.header("Risk Calculator")

        # Define the sub-tabs for risk calculator
        calculator_tabs = st.tabs(["Calculate Risk for a Patient", "Quick Calculation", "Model Explanation"])

        # Load the models
        with open('draft_log_reg_mortality.pkl', 'rb') as file:
            death_model = pickle.load(file)

        with open('draft_log_reg_icu.pkl', 'rb') as file:
            icu_model = pickle.load(file)

        # Define mappings
        anemia_mapping = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
        grade_kidney_mapping = {'G1': 1, 'G2': 2, 'G3': 2, 'G4': 4, 'G5': 4}
        transfusion_mapping = {'0 units': 0, '1 unit': 1, '2 or more units': 2}
        surg_risk_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
        anemia_binned_mapping = {'none': 0, 'mild': 1, 'moderate/severe': 2}
        asa_mapping = {'I': 0, 'II': 1, 'III': 2, 'IV-VI': 3}
        rdw_mapping = {'<=15.7': 1, '>15.7': 0}
        loneliness_mapping = {'Low': 0, 'Moderate': 1, 'Severe': 2}
        income_mapping = {'Low Income': 0, 'Median Income': 1, 'High Income': 2}

        def risk_calculation(input_data):
            with st.spinner('Processing...'):
                time.sleep(3)
                # Prediction for death
                death_prediction = death_model.predict(input_data)
                death_prediction_proba = death_model.predict_proba(input_data)

                death_outcome_message = ""
                death_outcome_color = ""

                death_prob = death_prediction_proba[0][1]
                if death_prob <= 0.2:
                    death_outcome_message = "Low risk of 30-day mortality"
                    death_outcome_color = "green"
                    death_action = "Regular monitoring and standard care"
                elif 0.2 < death_prob <= 0.4:
                    death_outcome_message = "Moderate risk of 30-day mortality"
                    death_outcome_color = "yellow"
                    death_action = "Enhanced monitoring and consider additional interventions"
                elif 0.4 < death_prob <= 0.7:
                    death_outcome_message = "High risk of 30-day mortality"
                    death_outcome_color = "orange"
                    death_action = "Intensive monitoring and prepare for potential critical interventions"
                else:
                    death_outcome_message = "Very high risk of 30-day mortality"
                    death_outcome_color = "red"
                    death_action = "Immediate and intensive interventions, consider ICU admission and advanced life support measures"

                # Prediction for ICU admission
                icu_prediction = icu_model.predict(input_data)
                icu_prediction_proba = icu_model.predict_proba(input_data)

                icu_outcome_message = ""
                icu_outcome_color = ""

                icu_prob = icu_prediction_proba[0][1]
                if icu_prob <= 0.2:
                    icu_outcome_message = "No need for ICU"
                    icu_outcome_color = "green"
                    icu_action = "no ICU required"

                elif 0.2 < icu_prob <= 0.7:
                    icu_outcome_message = "Needs further assessment for ICU"
                    icu_outcome_color = "yellow"
                    icu_action = "Needs further assessment"

                else:
                    icu_outcome_message = "ICU priority"
                    icu_outcome_color = "red"
                    icu_action = "ICU Priority"

                # Display results side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                        <div style='text-align: center; padding: 10px; border: 2px solid {death_outcome_color}; border-radius: 10px;'>
                            <h2 style='color: {death_outcome_color};'>{death_outcome_message}</h2>
                            <p><strong>Probability:</strong> {death_prediction_proba[0][1]:.2f}</p>
                            <p><strong>Action:</strong> {death_action}</p>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div style='text-align: center; padding: 10px; border: 2px solid {icu_outcome_color}; border-radius: 10px;'>
                            <h2 style='color: {icu_outcome_color};'>{icu_outcome_message}</h2>
                            <p><strong>Probability:</strong> {icu_prediction_proba[0][1]:.2f}</p>
                            <p><strong>Action:</strong> {icu_action}</p>
                        </div>
                    """, unsafe_allow_html=True)

        with calculator_tabs[0]:
            st.subheader("Calculate Risk for a Patient")
            indexno_input = st.text_input("Enter Patient Index Number for Risk Calculation:")
            if indexno_input:
                patient_data = st.session_state.patient_data[st.session_state.patient_data['IndexNo'] == indexno_input]
                if not patient_data.empty:
                    st.write("Patient Data:")
                    st.dataframe(patient_data)
                    patient_data = patient_data.iloc[0]

                    # Mapping inputs
                    anemia_category_mapped = anemia_mapping[patient_data['Anemia category']]
                    grade_kidney_disease_mapped = grade_kidney_mapping[patient_data['Grade of Kidney Disease']]
                    transfusion_category_mapped = transfusion_mapping[patient_data['Transfusion Intra and Postop Category']]
                    surg_risk_category_mapped = surg_risk_mapping[patient_data['Surgical Risk Category']]
                    grade_kidney_category_mapped = grade_kidney_mapping[patient_data['Grade of Kidney Category']]
                    anemia_binned_mapped = anemia_binned_mapping[patient_data['Anemia Category Binned']]
                    rdw_15_7_mapped = rdw_mapping[patient_data['RDW15.7']]
                    asa_category_binned_mapped = asa_mapping[patient_data['ASA Category Binned']]
                    male_mapped = 1 if patient_data['Gender'] == 'Male' else 0
                    ga_mapped = 1 if patient_data['Anaesthesia Type'] == 'GA' else 0
                    emergency_mapped = 1 if patient_data['Surgery Priority'] == 'Emergency' else 0
                    race_chinese = 1 if patient_data['Race'] == 'Chinese' else 0
                    race_indian = 1 if patient_data['Race'] == 'Indian' else 0
                    race_malay = 1 if patient_data['Race'] == 'Malay' else 0
                    race_others = 1 if patient_data['Race'] == 'Others' else 0
                    creatine_rcri_mapped = 1 if patient_data['Creatine RCRI Category'] == 'Yes' else 0
                    dm_insulin_mapped = 1 if patient_data['DM Insulin Category'] == 'Yes' else 0
                    chf_rcri_mapped = 1 if patient_data['CHF RCRI Category'] == 'Yes' else 0
                    ihd_rcri_mapped = 1 if patient_data['IHD RCRI Category'] == 'Yes' else 0
                    cva_rcri_mapped = 1 if patient_data['CVA RCRI Category'] == 'Yes' else 0
                    loneliness_mapped = loneliness_mapping[patient_data['Loneliness']]
                    income_mapped = income_mapping[patient_data['Income Category']]

                    # Create input array for prediction
                    input_data = np.array([[
                        patient_data['Age'], patient_data['RCRI score'], anemia_category_mapped, patient_data['PreopEGFRMDRD'], grade_kidney_disease_mapped,
                        patient_data['Preoptransfusion within 30 days'], patient_data['Intraop'], patient_data['Postop within 30 days'], patient_data['Transfusion intra and postop'],
                        transfusion_category_mapped, surg_risk_category_mapped, grade_kidney_category_mapped,
                        anemia_binned_mapped, rdw_15_7_mapped, asa_category_binned_mapped, male_mapped, ga_mapped, emergency_mapped,
                        race_chinese, race_indian, race_malay, race_others, creatine_rcri_mapped, dm_insulin_mapped, chf_rcri_mapped,
                        ihd_rcri_mapped, cva_rcri_mapped, income_mapped, loneliness_mapped
                    ]])

                    # Ensure input data is a 2D array
                    input_data = input_data.reshape(1, -1)

                    risk_calculation(input_data)

                    # Generate Summary and Recommendations
                    st.subheader("Summary and Recommendations")
                    if st.button("Generate Summary"):
                        with st.spinner("Generating summary..."):
                            time.sleep(3)  # Simulating a 3-second delay, replace this with actual data processing

                            # Generating prompt for LLM
                            patient_summary_prompt = f"""
                            Here is the patient's data:
                            Age: {patient_data['Age']}
                            RCRI score: {patient_data['RCRI score']}
                            Anemia category: {patient_data['Anemia category']}
                            PreopEGFRMDRD: {patient_data['PreopEGFRMDRD']}
                            Grade of Kidney Disease: {patient_data['Grade of Kidney Disease']}
                            Preoptransfusion within 30 days: {patient_data['Preoptransfusion within 30 days']}
                            Intraop: {patient_data['Intraop']}
                            Postop within 30 days: {patient_data['Postop within 30 days']}
                            Transfusion intra and postop: {patient_data['Transfusion intra and postop']}
                            Transfusion Intra and Postop Category: {patient_data['Transfusion Intra and Postop Category']}
                            Surgical Risk Category: {patient_data['Surgical Risk Category']}
                            Grade of Kidney Category: {patient_data['Grade of Kidney Category']}
                            Anemia Category Binned: {patient_data['Anemia Category Binned']}
                            RDW15.7: {patient_data['RDW15.7']}
                            ASA Category Binned: {patient_data['ASA Category Binned']}
                            Gender: {patient_data['Gender']}
                            Anaesthesia Type: {patient_data['Anaesthesia Type']}
                            Surgery Priority: {patient_data['Surgery Priority']}
                            Race: {patient_data['Race']}
                            Creatine RCRI Category: {patient_data['Creatine RCRI Category']}
                            DM Insulin Category: {patient_data['DM Insulin Category']}
                            CHF RCRI Category: {patient_data['CHF RCRI Category']}
                            IHD RCRI Category: {patient_data['IHD RCRI Category']}
                            CVA RCRI Category: {patient_data['CVA RCRI Category']}
                            Income Category: {patient_data['Income Category']}
                            Loneliness: {patient_data['Loneliness']}
                            Death Prediction: {death_outcome_message}
                            Death Probability: {death_prediction_proba[0][1]:.2f}
                            ICU Prediction: {icu_outcome_message}
                            ICU Probability: {icu_prediction_proba[0][1]:.2f}

                            Based on the above data, provide a summary and recommendations for the doctor to communicate with the patient's relatives.
                            """

                            summary = generate_llama2_response(patient_summary_prompt, llm, temperature, top_p, max_length)
                            st.write(summary)

        with calculator_tabs[1]:
            st.subheader("Quick Calculation")
            with st.form(key='quick_calc_form'):
                general_info_col1, general_info_col2 = st.columns(2)
                with general_info_col1:
                    age = st.number_input('Age', min_value=0, max_value=120, step=1)
                    rcri_score = st.number_input('RCRI score', min_value=0, max_value=10, step=1)
                    gender = st.selectbox('Gender', ['Male', 'Female'])
                    race = st.selectbox('Race', ['Chinese', 'Indian', 'Malay', 'Others'])
                    surgery_priority = st.selectbox('Surgery Priority', ['Elective', 'Emergency'])
                    anesthesia_type = st.selectbox('Anaesthesia Type', ['GA', 'RA'])

                st.subheader("Anemia Related Features🩸")
                anemia_col1, anemia_col2 = st.columns(2)
                with anemia_col1:
                    anemia_category = st.selectbox('Anemia category', ['none', 'mild', 'moderate', 'severe'])
                    anemia_binned = st.selectbox('Anemia Category Binned', ['none', 'mild', 'moderate/severe'])
                with anemia_col2:
                    rdw_15_7 = st.selectbox('RDW15.7', ['<=15.7', '>15.7'])

                st.subheader("Kidney Related Features 🩺")
                kidney_col1, kidney_col2 = st.columns(2)
                with kidney_col1:
                    preop_egfr = st.number_input('PreopEGFRMDRD', min_value=0, max_value=1000, step=1)
                    grade_kidney_disease = st.selectbox('Grade of Kidney Disease', ['G1', 'G2', 'G3', 'G4', 'G5'])
                with kidney_col2:
                    grade_kidney_category = st.selectbox('Grade of Kidney Category', ['G1', 'G2', 'G3', 'G4', 'G5'])
                
                st.subheader("Transfusion Related Features 💉")
                transfusion_col1, transfusion_col2 = st.columns(2)
                with transfusion_col1:
                    preop_transfusion = st.number_input('Preoptransfusion within 30 days', min_value=0, max_value=10, step=1)
                with transfusion_col2:
                    transfusion_intra_postop = st.number_input('Transfusion intra and postop', min_value=0, max_value=10, step=1)
                    transfusion_category = st.selectbox('Transfusion Intra and Postop Category', ['0 units', '1 unit', '2 or more units'])

                st.subheader("Surgical and Postoperative Features 🏥")
                surg_postop_col1, surg_postop_col2 = st.columns(2)
                with surg_postop_col1:
                    intraop = st.number_input('Intraop', min_value=0, max_value=10, step=1)
                with surg_postop_col2:
                    postop_within_30days = st.number_input('Postop within 30 days', min_value=0, max_value=10, step=1)

                st.subheader("Risk Factors ⚠️")
                risk_col1, risk_col2 = st.columns(2)
                with risk_col1:
                    surg_risk_category = st.selectbox('Surgical Risk Category', ['Low', 'Moderate', 'High'])
                    asa_category_binned = st.selectbox('ASA Category Binned', ['I', 'II', 'III', 'IV-VI'])
                with risk_col2:
                    creatine_rcri = st.selectbox('Creatine RCRI Category', ['Yes', 'No'])
                    dm_insulin = st.selectbox('DM Insulin Category', ['Yes', 'No'])
                    chf_rcri = st.selectbox('CHF RCRI Category', ['Yes', 'No'])
                    ihd_rcri = st.selectbox('IHD RCRI Category', ['Yes', 'No'])
                    cva_rcri = st.selectbox('CVA RCRI Category', ['Yes', 'No'])

                st.subheader("SDOH (Social Determinants of Health) 🌐")
                sdoh_col1, sdoh_col2 = st.columns(2)
                with sdoh_col1:
                    income_category = st.selectbox('Income Category', ['Low Income', 'Median Income', 'High Income'])
                with sdoh_col2:
                    loneliness = st.selectbox('Loneliness', ['Low', 'Moderate', 'Severe'])

                submit_quick_calc_button = st.form_submit_button(label='Calculate Risk')

            if submit_quick_calc_button:
                # Mapping inputs
                anemia_category_mapped = anemia_mapping[anemia_category]
                grade_kidney_disease_mapped = grade_kidney_mapping[grade_kidney_disease]
                transfusion_category_mapped = transfusion_mapping[transfusion_category]
                surg_risk_category_mapped = surg_risk_mapping[surg_risk_category]
                grade_kidney_category_mapped = grade_kidney_mapping[grade_kidney_category]
                anemia_binned_mapped = anemia_binned_mapping[anemia_binned]
                rdw_15_7_mapped = rdw_mapping[rdw_15_7]
                asa_category_binned_mapped = asa_mapping[asa_category_binned]
                male_mapped = 1 if gender == 'Male' else 0
                ga_mapped = 1 if anesthesia_type == 'GA' else 0
                emergency_mapped = 1 if surgery_priority == 'Emergency' else 0
                race_chinese = 1 if race == 'Chinese' else 0
                race_indian = 1 if race == 'Indian' else 0
                race_malay = 1 if race == 'Malay' else 0
                race_others = 1 if race == 'Others' else 0
                creatine_rcri_mapped = 1 if creatine_rcri == 'Yes' else 0
                dm_insulin_mapped = 1 if dm_insulin == 'Yes' else 0
                chf_rcri_mapped = 1 if chf_rcri == 'Yes' else 0
                ihd_rcri_mapped = 1 if ihd_rcri == 'Yes' else 0
                cva_rcri_mapped = 1 if cva_rcri == 'Yes' else 0
                loneliness_mapped = loneliness_mapping[loneliness]
                income_mapped = income_mapping[income_category]

                # Create input array for prediction
                input_data = np.array([[
                    age, rcri_score, anemia_category_mapped, preop_egfr, grade_kidney_disease_mapped,
                    preop_transfusion, intraop, postop_within_30days, transfusion_intra_postop,
                    transfusion_category_mapped, surg_risk_category_mapped, grade_kidney_category_mapped,
                    anemia_binned_mapped, rdw_15_7_mapped, asa_category_binned_mapped, male_mapped, ga_mapped, emergency_mapped,
                    race_chinese, race_indian, race_malay, race_others, creatine_rcri_mapped, dm_insulin_mapped, chf_rcri_mapped,
                    ihd_rcri_mapped, cva_rcri_mapped, income_mapped, loneliness_mapped
                ]])

                # Ensure input data is a 2D array
                input_data = input_data.reshape(1, -1)

                risk_calculation(input_data)

        with calculator_tabs[2]:
            if 'patient_data' not in st.session_state:
                st.session_state.patient_data = pd.DataFrame()  # or load your actual data here

            # Ensure patient_data has the correct columns
            icu_feature_names = icu_model.feature_names_in_
            death_feature_names = death_model.feature_names_in_

            # Feature importances calculation
            icu_feature_importances = pd.Series(icu_model.coef_[0], index=icu_feature_names)
            death_feature_importances = pd.Series(death_model.coef_[0], index=death_feature_names)

            icu_df = pd.DataFrame({
                'Feature': icu_feature_importances.index,
                'Importance': icu_feature_importances.values
            })

            mortality_df = pd.DataFrame({
                'Feature': death_feature_importances.index,
                'Importance': death_feature_importances.values
            })

            # Streamlit tabs setup
            explanation_tabs = st.tabs(["ICU Model", "Mortality Model"])

            with explanation_tabs[0]:
                st.subheader("ICU Model")
                st.write("Feature importances are a measure of how much each feature (or variable) in a dataset contributes to the prediction power of a machine learning model. They provide insight into which features are most influential in making predictions and can help to interpret the model's behavior. High feature importance means that a feature has a strong impact on the model's predictions, whereas low importance indicates a lesser influence. By analyzing feature importances, you can understand which aspects of the data are driving the model's decisions, and it can also help in feature selection, simplifying the model without sacrificing performance.")
                st.bar_chart(icu_df.set_index('Feature'),horizontal=True)

            with explanation_tabs[1]:
                st.subheader("Mortality Model")
                st.write("Feature importances are a measure of how much each feature (or variable) in a dataset contributes to the prediction power of a machine learning model. They provide insight into which features are most influential in making predictions and can help to interpret the model's behavior. High feature importance means that a feature has a strong impact on the model's predictions, whereas low importance indicates a lesser influence. By analyzing feature importances, you can understand which aspects of the data are driving the model's decisions, and it can also help in feature selection, simplifying the model without sacrificing performance.")
                st.bar_chart(mortality_df.set_index('Feature'),horizontal=True)

if __name__ == "__main__":
    main()
