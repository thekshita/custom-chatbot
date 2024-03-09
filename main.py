#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:08:29 2024

@author: deekshitadoli
"""

import streamlit as st
import shelve
from query_service import QueryService
import logging

st.title("EquityEngine")



USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

st.markdown(
        f"""
            <style>
                [data-testid="stSidebar"] {{
                    background-image: url(https://uw-s3-cdn.s3.us-west-2.amazonaws.com/wp-content/uploads/sites/230/2023/11/02134809/W-Logo_Gold_RGB.png);
                    background-repeat: no-repeat;
                    padding-top: 80px;
                    background-position: 20px 20px;
                    background-color: #4b2e83;
                    background-size: 200px;
                    width: 1000px;
                }}
                [data-testid="stApp"] {{
                    background-color: #e8e3d3;
                }}
                .sidebar-content {{
                    margin-top: 50px; /* Adjust top margin for content alignment */
                }}
                .sidebar-radio-group > div {{
                    margin-top: 20px; /* Adjust the margin between buttons */
                    color: white;
                }}
                .sidebar-radio-group label {{
                    color: white !important; /* Change text color of radio button labels */
                }}
            </style>
            """,
        unsafe_allow_html=True,
    )

#st.markdown(page_bg_img, unsafe_allow_html=True)

disclaimer_text = """
This tool is not an exhaustive source of information pertinent to the University of Washington. It may sometimes produce erroneous results. It is recommended that you verify any information.
"""


# Ensure openai_model is initialized in session state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar with a button to delete chat history
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])
    st.markdown("<div style='color: white;'>", unsafe_allow_html=True)
    user_choice = st.radio(":gray[Select an option:]", (":gray[General]", ":gray[DRS]", ":gray[HFS]"), key="user_choice")
    
    logging.debug("query service changed to "+ user_choice)
    query_service = QueryService(user_choice[6:-1]) 

with st.expander("Disclaimer"):
    st.write(disclaimer_text)
    
# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        


if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""        
        full_response = query_service.ask_agent(question=prompt)

        # Display the response
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})


    
# Save chat history after each interaction
save_chat_history(st.session_state.messages)