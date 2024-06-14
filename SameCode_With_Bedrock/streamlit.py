import streamlit as st
import app

st.set_page_config(page_title="GenAI Proposals") #HTML title
st.title("Proposals") #page title

col1, col2 = st.columns(2)
    
with col1:
    prompt_text = st.text_area("Enter Job Description", height=350)
    user_name = st.text_input("Enter you name", placeholder="Your name here")
    client_name = st.text_input("Enter Clien name", placeholder="Your client name here")
    experience_year = st.text_input("Enter the years of experiance", placeholder="Your client name here")
    uploaded_file = st.file_uploader("Select Your Previously done job", type=['pdf', 'docs'], label_visibility="collapsed")
    process_button = st.button("Run", type="primary") 

with col2:
    if process_button:
        with st.spinner("Creating Proposal"):
            response = app.llm_and_prompt(prompt_text, user_name, client_name, uploaded_file)
            st.write(response)