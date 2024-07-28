import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time

def risk_calculator():
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
                st.session_state.patient_data.to_csv('patient_data.csv', index=False)
