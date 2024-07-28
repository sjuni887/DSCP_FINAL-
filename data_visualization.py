import streamlit as st

def data_visualization():
    st.header("Data Visualization")
    power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=7ca96589-22e3-465d-9f5c-af24869867b4&autoAuth=true&ctid=cba9e115-3016-4462-a1ab-a565cba0cdf1"
    st.markdown(f"""
        <iframe width="170%" height="700" src="{power_bi_url}" frameborder="0" allowfullscreen="true"></iframe>
    """, unsafe_allow_html=True)
