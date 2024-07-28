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

# Set page configuration
st.set_page_config(page_title="Healthcare Dashboard")

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
        output = replicate.run(model, input={"prompt": prompt, "temperature": temperature, "top_p": top_p, "max_length": max_length, "repetition_penalty": 1})
        summaries.append(''.join(output))
    final_summary = ' '.join(summaries)
    return final_summary

# Function to answer query about text
def answer_query_about_text(query, text, model, temperature=0.1, top_p=0.9, max_length=120):
    prompt = f"Please answer the following query based on the given text:\n\nText: {text}\n\nQuery: {query}\n\nAnswer:"
    output = replicate.run(model, input={"prompt": prompt, "temperature": temperature, "top_p": top_p, "max_length": max_length, "repetition_penalty": 1})
    return ''.join(output)

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
        st.title("🦙💬 Llama 2 Chatbot")

        # Replicate Credentials
        with st.sidebar:
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

        # Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
        def generate_llama2_response(prompt_input):
            string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
            for dict_message in st.session_state.messages:
                if dict_message["role"] == "user":
                    string_dialogue += "User: " + dict_message["content"] + "\n\n"
                else:
                    string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
            output = replicate.run(llm, input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ", "temperature": temperature, "top_p": top_p, "max_length": max_length, "repetition_penalty": 1})
            return output

        # User-provided prompt
        col1, col2 = st.columns([4,1])
        with col1:
            prompt = st.chat_input(disabled=not replicate_api)
        with col2:
            audio_text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
        
        if audio_text:
            prompt = audio_text

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_llama2_response(st.session_state.messages[-1]["content"])
                    placeholder = st.empty()
                    full_response = ''.join(response)
                    placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Upload PDF and extract text
        st.subheader("Upload PDF for Analysis")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file:
            text = extract_text_from_pdf(uploaded_file)
            st.text_area("Extracted Text", text, height=300)

            # Summarize PDF text
            if st.button("Summarize PDF"):
                with st.spinner("Summarizing..."):
                    summary = summarize_text(text, llm, temperature, top_p, max_length)
                    st.write("Summary:")
                    st.write(summary)

            # Query PDF text
            query = st.text_input("Ask a question about the PDF")
            if st.button("Get Answer"):
                with st.spinner("Generating answer..."):
                    answer = answer_query_about_text(query, text, llm, temperature, top_p, max_length)
                    st.write("Answer:")
                    st.write(answer)

    with tab2:
        st.header("Risk Calculator")
        # Load the models
        with open('rf_model_death.pkl', 'rb') as file:
            death_model = pickle.load(file)

        with open('rf_model_icu.pkl', 'rb') as file:
            icu_model = pickle.load(file)

        # Define mappings
        anemia_mapping = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
        grade_kidney_mapping = {'G1': 1, 'G2': 2, 'G3': 2, 'G4': 4, 'G5': 4}
        transfusion_mapping = {'0 units': 0, '1 unit': 1, '2 or more units': 2}
        surg_risk_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
        anemia_binned_mapping = {'none': 0, 'mild': 1, 'moderate/severe': 2}
        asa_mapping = {'I': 0, 'II': 1, 'III': 2, 'IV-VI': 3}
        rdw_mapping = {'<=15.7': 1, '>15.7': 0}

        with st.form(key='risk_form'):
            st.subheader("General Information")
            col1, col2 = st.columns(2)
            with col1:
                indexno = st.text_input('Index Number')
                age = st.slider('Age', min_value=0, max_value=120, step=1)
                rcri_score = st.number_input('RCRI score', min_value=0, max_value=10, step=1)
                gender = st.selectbox('Gender', ['Male', 'Female'])
                race = st.selectbox('Race', ['Chinese', 'Indian', 'Malay', 'Others'])

            with col2:
                surgery_priority = st.selectbox('Surgery Priority', ['Elective', 'Emergency'])
                anesthesia_type = st.selectbox('Anaesthesia Type', ['GA', 'RA'])

            st.subheader("Kidney Related Features 🩺")
            col1, col2 = st.columns(2)
            with col1:
                preop_egfr = st.slider('PreopEGFRMDRD', min_value=0, max_value=1000, step=1)
                grade_kidney_disease = st.selectbox('Grade of Kidney Disease', list(grade_kidney_mapping.keys()))
            with col2:
                grade_kidney_category = st.selectbox('Grade of Kidney Category', list(grade_kidney_mapping.keys()))

            st.subheader("Anemia Related Features 🩸")
            col1, col2 = st.columns(2)
            with col1:
                anemia_category = st.selectbox('Anemia category', list(anemia_mapping.keys()))
                anemia_binned = st.selectbox('Anemia Category Binned', list(anemia_binned_mapping.keys()))
            with col2:
                rdw_15_7 = st.selectbox('RDW15.7', list(rdw_mapping.keys()))

            st.subheader("Transfusion Related Features 💉")
            col1, col2 = st.columns(2)
            with col1:
                preop_transfusion = st.number_input('Preoptransfusion within 30 days', min_value=0, max_value=10, step=1)
                transfusion_intra_postop = st.number_input('Transfusion intra and postop', min_value=0, max_value=10, step=1)
            with col2:
                transfusion_category = st.selectbox('Transfusion Intra and Postop Category', list(transfusion_mapping.keys()))

            st.subheader("Surgical and Postoperative Features 🏥")
            col1, col2 = st.columns(2)
            with col1:
                intraop = st.number_input('Intraop', min_value=0, max_value=10, step=1)
                postop_within_30days = st.number_input('Postop within 30 days', min_value=0, max_value=10, step=1)

            st.subheader("Risk Factors ⚠️")
            col1, col2 = st.columns(2)
            with col1:
                surg_risk_category = st.selectbox('Surgical Risk Category', list(surg_risk_mapping.keys()))
                asa_category_binned = st.selectbox('ASA Category Binned', list(asa_mapping.keys()))
            with col2:
                dm_insulin = st.selectbox('DM Insulin Category', ['Yes', 'No'])
                chf_rcri = st.selectbox('CHF RCRI Category', ['Yes', 'No'])
                ihd_rcri = st.selectbox('IHD RCRI Category', ['Yes', 'No'])
                cva_rcri = st.selectbox('CVA RCRI Category', ['Yes', 'No'])
                creatine_rcri = st.selectbox('Creatine RCRI Category', ['Yes', 'No'])

            submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            # Check for duplicate indexno
            if indexno in st.session_state.patient_data['IndexNo'].values:
                st.error("This Index Number already exists. Please enter a new Index Number.")
            else:
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

                # Create input array for prediction
                input_data = np.array([[
                    age, rcri_score, anemia_category_mapped, preop_egfr, grade_kidney_disease_mapped,
                    preop_transfusion, intraop, postop_within_30days, transfusion_intra_postop,
                    transfusion_category_mapped, surg_risk_category_mapped, grade_kidney_category_mapped,
                    anemia_binned_mapped, rdw_15_7_mapped, asa_category_binned_mapped, male_mapped, ga_mapped, emergency_mapped,
                    race_chinese, race_indian, race_malay, race_others, creatine_rcri_mapped, dm_insulin_mapped, chf_rcri_mapped,
                    ihd_rcri_mapped, cva_rcri_mapped
                ]])

                # Ensure input data is a 2D array
                input_data = input_data.reshape(1, -1)

                with st.spinner('Processing...'):
                    time.sleep(3)
                    # Prediction for death
                    death_prediction = death_model.predict(input_data)
                    death_prediction_proba = death_model.predict_proba(input_data)

                    death_outcome_message = "Death expectancy in 30 days is likely" if death_prediction[0] == 1 else "Death expectancy in 30 days is unlikely"
                    death_outcome_color = "red" if death_prediction[0] == 1 else "green"

                    # Prediction for ICU admission
                    icu_prediction = icu_model.predict(input_data)
                    icu_prediction_proba = icu_model.predict_proba(input_data)

                    icu_outcome_message = "ICU admission is likely" if icu_prediction[0] == 1 else "ICU admission is unlikely"
                    icu_outcome_color = "red" if icu_prediction[0] == 1 else "green"

                    # Display results side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                            <div style='text-align: center; padding: 10px; border: 2px solid {death_outcome_color}; border-radius: 10px;'>
                                <h2 style='color: {death_outcome_color};'>{death_outcome_message}</h2>
                                <p><strong>Probability:</strong> {death_prediction_proba[0][1]:.2f}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                            <div style='text-align: center; padding: 10px; border: 2px solid {icu_outcome_color}; border-radius: 10px;'>
                                <h2 style='color: {icu_outcome_color};'>{icu_outcome_message}</h2>
                                <p><strong>Probability:</strong> {icu_prediction_proba[0][1]:.2f}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    # Append raw input data and prediction to DataFrame
                    new_data = pd.DataFrame([{
                        'IndexNo': indexno, 'Age': age, 'RCRI score': rcri_score, 'Anemia category': anemia_category, 'PreopEGFRMDRD': preop_egfr,
                        'Grade of Kidney Disease': grade_kidney_disease, 'Preoptransfusion within 30 days': preop_transfusion,
                        'Intraop': intraop, 'Postop within 30 days': postop_within_30days, 'Transfusion intra and postop': transfusion_intra_postop,
                        'Transfusion Intra and Postop Category': transfusion_category, 'Surgical Risk Category': surg_risk_category,
                        'Grade of Kidney Category': grade_kidney_category, 'Anemia Category Binned': anemia_binned,
                        'RDW15.7': rdw_15_7, 'ASA Category Binned': asa_category_binned, 'Gender': gender, 'Anaesthesia Type': anesthesia_type,
                        'Surgery Priority': surgery_priority, 'Race': race, 'Creatine RCRI Category': creatine_rcri, 'DM Insulin Category': dm_insulin,
                        'CHF RCRI Category': chf_rcri, 'IHD RCRI Category': ihd_rcri, 'CVA RCRI Category': cva_rcri,
                        'Death Prediction': death_prediction[0], 'Death Probability': death_prediction_proba[0][1],
                        'ICU Prediction': icu_prediction[0], 'ICU Probability': icu_prediction_proba[0][1]
                    }])
                    st.session_state.patient_data = pd.concat([st.session_state.patient_data, new_data], ignore_index=True)
                    st.session_state.patient_data.to_csv(csv_file_path, index=False)

    with tab3:
        st.header("Patient Data")
        st.write("All Patient Data")
        st.dataframe(st.session_state.patient_data)

        indexno_query = st.text_input("Enter Index Number to view specific patient data:")
        if indexno_query:
            patient_data = st.session_state.patient_data[st.session_state.patient_data['IndexNo'] == indexno_query]
            if not patient_data.empty:
                st.write("Patient Data:")
                st.dataframe(patient_data)
                st.write("Summary:")
                st.write(patient_data.describe().T)

                if st.button("Generate Summary"):
                    with st.spinner("Processing data..."):
                        time.sleep(3)  # Simulating a 3-second delay, replace this with actual data processing

                        # Generating prompt for LLM
                        patient_data_dict = patient_data.to_dict(orient='records')[0]
                        patient_summary_prompt = f"""
                        Here is the patient's data:
                        Age: {patient_data_dict['Age']}
                        RCRI score: {patient_data_dict['RCRI score']}
                        Anemia category: {patient_data_dict['Anemia category']}
                        PreopEGFRMDRD: {patient_data_dict['PreopEGFRMDRD']}
                        Grade of Kidney Disease: {patient_data_dict['Grade of Kidney Disease']}
                        Preoptransfusion within 30 days: {patient_data_dict['Preoptransfusion within 30 days']}
                        Intraop: {patient_data_dict['Intraop']}
                        Postop within 30 days: {patient_data_dict['Postop within 30 days']}
                        Transfusion intra and postop: {patient_data_dict['Transfusion intra and postop']}
                        Transfusion Intra and Postop Category: {patient_data_dict['Transfusion Intra and Postop Category']}
                        Surgical Risk Category: {patient_data_dict['Surgical Risk Category']}
                        Grade of Kidney Category: {patient_data_dict['Grade of Kidney Category']}
                        Anemia Category Binned: {patient_data_dict['Anemia Category Binned']}
                        RDW15.7: {patient_data_dict['RDW15.7']}
                        ASA Category Binned: {patient_data_dict['ASA Category Binned']}
                        Gender: {patient_data_dict['Gender']}
                        Anaesthesia Type: {patient_data_dict['Anaesthesia Type']}
                        Surgery Priority: {patient_data_dict['Surgery Priority']}
                        Race: {patient_data_dict['Race']}
                        Creatine RCRI Category: {patient_data_dict['Creatine RCRI Category']}
                        DM Insulin Category: {patient_data_dict['DM Insulin Category']}
                        CHF RCRI Category: {patient_data_dict['CHF RCRI Category']}
                        IHD RCRI Category: {patient_data_dict['IHD RCRI Category']}
                        CVA RCRI Category: {patient_data_dict['CVA RCRI Category']}
                        Death Prediction: {patient_data_dict['Death Prediction']}
                        Death Probability: {patient_data_dict['Death Probability']}
                        ICU Prediction: {patient_data_dict['ICU Prediction']}
                        ICU Probability: {patient_data_dict['ICU Probability']}

                        Based on the above data, provide a summary and recommendations for the doctor to communicate with the patient's relatives.
                        """

                        summary = generate_llama2_response(patient_summary_prompt)
                        st.write(summary)

            else:
                st.write("No data found for the entered Index Number.")

    with tab4:
        st.header("Data Visualization")
        power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=7ca96589-22e3-465d-9f5c-af24869867b4&autoAuth=true&ctid=cba9e115-3016-4462-a1ab-a565cba0cdf1"
        st.markdown(f"""
            <iframe width="170%" height="700" src="{power_bi_url}" frameborder="0" allowfullscreen="true"></iframe>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
