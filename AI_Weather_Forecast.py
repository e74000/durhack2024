from dotenv import load_dotenv
import os
import streamlit as st
from ai_functionality import generate_ai_reponse

# Load environment variables from the .env file
load_dotenv()
DATAFORSEO_LOGIN = os.getenv("DATAFORSEO_LOGIN")
DATAFORSEO_PASSWORD = os.getenv("DATAFORSEO_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Sidebar widgets for user inputs
with st.sidebar:
    selected_language = st.selectbox(
        "Select A Language", 
        ["English", "French", "Chinese", "Nepali", "Bahasa Indonesia", "Bahasa Melayu", "Tamil"]
    )
    
    traits = st.multiselect(
        "Select a trait(s) of the bot",
        [
            'Funny', 'Rude', 'Normal', 'Black', 'White', 'News reporter', 
            'Childish', 'Mature', 'Charismatic', 'Compassion', 'Agreebleness', 
            'Creative', 'Optimism', 'Confident', 'Ambitious'
        ],
        ['Funny', 'Charismatic']  # Default selected traits
    )

# Display the selected options
st.write(f"You selected: {selected_language}")
st.write(f"Selected traits: {', '.join(traits)}")

# Initialize session state for storing chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, How can I help you today!"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input from the chat interface
user_prompt = st.chat_input()

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    # Generate AI response if last message is from the user
    if st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    ai_response, _ = generate_ai_reponse(
                        language=selected_language,
                        traits=traits,
                        user_prompt=user_prompt
                    )
                    st.write(ai_response)
                    
                    # Append AI's response to session state
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                except Exception as error:
                    st.error(f"An error occurred while generating the response: {error}")
