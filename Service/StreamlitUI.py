#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename: StreamlitUI.py
# @Author: ikbal
# @Time: 6/1/2024 3:06 AM

import streamlit as st
import requests, os, json
import pandas as pd
from Utils import get_config

st.set_page_config(
    page_title="Gemma 2B Text Generation",
    page_icon=":sparkles:",
    layout="wide",
)

chat_dir = "Chats"
os.makedirs(chat_dir, exist_ok=True)

url = get_config('Config/serviceConfigs.json')['http']['ngrok']
def load_chats():
    chats_ = [f.replace('.json', '') for f in os.listdir(chat_dir) if f.endswith('.json')]
    return chats_

st.markdown("""
<style>
.title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    margin: 20px 0;
    color: #F7A008;
}
</style>
<div class='title'>Gemma 2B Text Generation</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
div.stButton > button {
    display: block;
    margin: 0 auto;
    width: 50%;
}
</style>""", unsafe_allow_html=True)

col1, col2 = st.columns(spec=[8, 8], gap="small")
response = None
chats = load_chats()
chat_file = None

selected_chat = st.sidebar.selectbox(":speech_balloon: :blue[Select Conversations]", chats)

if st.sidebar.button("Delete"):
    if selected_chat:
        os.remove(os.path.join(chat_dir, f"{selected_chat}.json"))
        chats.remove(selected_chat)
        st.experimental_rerun()

st.sidebar.markdown("---")

new_chat = st.sidebar.text_input("New conversation name", value="", help="Enter a name for the new conversation.")
if st.sidebar.button("Create"):
    if new_chat:
        new_chat_path = os.path.join(chat_dir, f"{new_chat}.json")
        if os.path.exists(new_chat_path):
            st.sidebar.error("A conversation with this name already exists. Please choose a different name.")
        else:
            selected_chat = new_chat
            chats.append(new_chat)
            with open(new_chat_path, 'w') as f:
                json.dump([], f)
            st.experimental_rerun()



try:
    if selected_chat:
        chat_file = os.path.join(chat_dir, f"{selected_chat}.json")
        with open(chat_file) as f:
            conversation = json.load(f)
    else:
        chat_file = None
        conversation = []
except Exception as e:
    st.error(f"An error occurred: {e}")

with col1:
    st.subheader(":arrow_forward: :blue[Input]")
    with st.form("text_gen_form"):
        try:
            selected_model = st.selectbox(":black_small_square: :blue[Model]", options=["Gemma 2B Default", "Gemma 2B Trained"], index=0,
                                           help="Select the model to generate text from.")
            input_text = st.text_area(value="", help="Input the text you want to generate from.", label=":black_small_square: :blue[Input Text]")
            max_new_tokens = st.number_input(":black_small_square: :blue[Max New Tokens]", min_value=10, max_value=3096, value=50, step=10,
                                             help="Maximum new tokens to generate.")
            temperature = st.slider(":black_small_square: :blue[Temperature]", min_value=0.0, max_value=1.0, value=0.7, step=0.05,
                                    help="Controls randomness in generation.")
            top_k = st.number_input(":black_small_square: :blue[Top K]", min_value=0, max_value=100, value=50,
                                    help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
            top_p = st.slider(":black_small_square: :blue[Top P]", min_value=0.0, max_value=1.0, value=0.92, step=0.01,
                              help="Nucleus sampling: selects tokens with cumulative probability of `top_p` or higher.")
            submit_button = st.form_submit_button(":sparkles: Generate")

            if submit_button:
                data = {
                    "model": selected_model,
                    "text": input_text,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p
                }

                # Flask API endpoint
                response = requests.post(url, json=data)

            if submit_button and selected_chat:

                if response and response.status_code == 200:
                    result = response.json().get("response")
                    conversation.append({"input": input_text, "output": result})
                    with open(chat_file, 'w') as f:
                        json.dump(conversation, f)

        except Exception as e:
            st.error(f"An error occurred: {e}")

with col2:
    st.subheader(":arrow_forward: :blue[Output]")
    try:
        if response:
            if response.status_code == 200:
                result = response.json().get("response")
                st.text_area(value=result, height=300, help="Generated text from the model.", disabled=True, label="Generated Text")
            else:
                st.error("Failed to generate text. Server responded with an error.")
        else:
            st.text_area(value="Submit a query to generate text.", height=300, disabled=True, label = "Generated Text")

    except Exception as e:
        st.error(f"An error occurred: {e}")


if selected_chat:
    try:
        chat_file = os.path.join(chat_dir, f"{selected_chat}.json")
        with open(chat_file) as f:
            conversation = json.load(f)

        if conversation:
            st.markdown("---")
            st.subheader(":speech_balloon: :blue[Conversation History]")
            df = pd.DataFrame(conversation)
            st.table(df[['input', 'output']])
        else:
            st.markdown("---")
            st.subheader(":speech_balloon: :blue[Conversation History]")
            st.warning("No conversation history yet.")
    except Exception as e:
        st.error(f"An error occurred: {e}")