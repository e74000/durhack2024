from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.utilities.dataforseo_api_search import DataForSeoAPIWrapper
from decouple import config
import os

# Load environment variables
os.environ["DATAFORSEO_LOGIN"] = config("DATAFORSEO_LOGIN")
os.environ["DATAFORSEO_PASSWORD"] = config("DATAFORSEO_PASSWORD")

# Initialize the AI model
model = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"))

# Create the prompt template
prompt = ChatPromptTemplate.from_template(
    """You are a highly intelligent and friendly AI assistant. Provide a clear, direct, and conversational response to the user's question using the context provided. Use the specified language and demonstrate the selected traits.

   Traits: {traits}
   Language: {language}
   Human: {question}
   Context: {context}

   AI:""")

# Create a chain to handle input/output
chain = prompt | model | StrOutputParser()

def generate_ai_reponse(
        language: str,
        traits: list,
        user_prompt: str) -> (str, list):
    """
    Generates an AI response based on the provided user input, language, and traits.

    :param language: Language for the response.
    :param traits: List of traits the AI should demonstrate.
    :param user_prompt: User's input query.

    :return: A tuple containing the AI response and a list of context dictionaries.
    """
    try:
        # Initialize the DataForSeo API wrapper
        json_wrapper = DataForSeoAPIWrapper(
            top_count=3,
            json_result_types=["organic", "local_pack"],
            json_result_fields=["title", "description", "type", "text"],
        )

        # Retrieve context from the API
        context = json_wrapper.results(user_prompt)

        # Ensure descriptions are fully captured and not truncated
        for item in context:
            if "description" in item:
                item["description"] = item["description"].strip()

        # Generate AI response
        response = chain.invoke(
            {"question": user_prompt, "context": context, "language": language, "traits": traits}
        )

        return response, context
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while generating a response. Please try again later.", []
