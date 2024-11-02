from dotenv import load_dotenv, find_dotenv
import os
import streamlit as st
from ai_functionality import generate_ai_reponse

# Load environment variables from the .env file
load_dotenv()
DATAFORSED_LOGIN = os.getenv("DATAFORSEO_LOGIN")
DATAFORSED_PASSWORD = os.getenv("DATAFORSEO_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import streamlit as st

# Create a selectbox and multiselect in the sidebar
with st.sidebar:
    selected_language = st.selectbox(
        "Select A Language", 
        ("English", "French", "Chinese", "Nepali", "Bahasa Indonesia", "Bahasa Melayu", "Tamil")
    )
    
    traits = st.multiselect(
        "Select a trait(s) of the bot",
        [
            'Funny', 'Rude',
            'Normal', 'Black',
            'White', 'News reporter',
            'Childish', "Mature",
            'Charismatic', 'Compassion',
            'Agreebleness', "creative",
            'optimism', 'confident',
            'ambitious' 
        ],
        ['Funny', 'Charismatic']  # Default selected traits
    )

# Display the selected language and traits
st.write(f"You selected: {selected_language}")
st.write(f"Selected traits: {', '.join(traits)}")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, How can I help you today!"}
    ]

#Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# Get user input
user_prompt = st.chat_input()

if user_prompt is not None:
    # Append user input to the session state
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

# Check if the last message is not from the assistant before generating a new response
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            # Generate the AI response
            ai_response, context = generate_ai_reponse(
                language=selected_language,  # Ensure `selected_language` is defined elsewhere in your code
                traits=traits,               # Ensure `traits` is defined and available
                user_prompt=user_prompt
            )
            st.write(ai_response)
            st.write(context)

    # Append the AI's response to the session state
    new_ai_response = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_response)