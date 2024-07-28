import streamlit as st
import os
import replicate
from streamlit_mic_recorder import mic_recorder, speech_to_text
from PyPDF2 import PdfReader

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ''
    for page in pdf.pages:
        text += page.extract_text()
    return text

def summarize_text(text, model, temperature=0.1, top_p=0.9, max_length=120, chunk_size=4000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for chunk in chunks:
        prompt = f"Please summarize the following text:\n\n{chunk}\n\nSummary:"
        output = replicate.run(model, input={"prompt": prompt, "temperature": temperature, "top_p": top_p, "max_length": max_length, "repetition_penalty": 1})
        summaries.append(''.join(output))
    final_summary = ' '.join(summaries)
    return final_summary

def answer_query_about_text(query, text, model, temperature=0.1, top_p=0.9, max_length=120):
    prompt = f"Please answer the following query based on the given text:\n\nText: {text}\n\nQuery: {query}\n\nAnswer:"
    output = replicate.run(model, input={"prompt": prompt, "temperature": temperature, "top_p": top_p, "max_length": max_length, "repetition_penalty": 1})
    return ''.join(output)

def generate_llama2_response(prompt_input, llm, temperature, top_p, max_length):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    try:
        output = replicate.run(llm, input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ", "temperature": temperature, "top_p": top_p, "max_length": max_length, "repetition_penalty": 1})
        return output
    except replicate.exceptions.ReplicateError as e:
        st.error(f"An error occurred: {e}")
        return None

def generative_ai_chat():
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
                response = generate_llama2_response(st.session_state.messages[-1]["content"], llm, temperature, top_p, max_length)
                if response:
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
