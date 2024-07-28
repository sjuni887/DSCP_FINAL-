import streamlit as st
import pandas as pd
import replicate
import time

def generate_llama2_response(prompt_input, llm, temperature, top_p, max_length):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    prompt = f"{string_dialogue} {prompt_input} Assistant: "
    try:
        output = replicate.run(llm, input={"prompt": prompt, "temperature": temperature, "top_p": top_p, "max_length": max_length, "repetition_penalty": 1})
        return output
    except replicate.exceptions.ReplicateError as e:
        st.error(f"An error occurred: {e}")
        return None

def patient_data():
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

                    summary = generate_llama2_response(patient_summary_prompt, llm, temperature, top_p, max_length)
                    st.write(summary)

        else:
            st.write("No data found for the entered Index Number.")
