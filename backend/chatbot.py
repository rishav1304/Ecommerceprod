# Import Libraries
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, LLMChain
from langchain.llms import GooglePalm
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load environment variables from .env
load_dotenv()

# Access the Google API key from environment variable
google_api_key = os.getenv('GOOGLE_API_KEY')

# Data Loading
df = pd.read_csv('datatoworl.csv')

# Combine necessary columns into a single column
df['combined_info'] = df.apply(lambda row: f"Order time: {row['created_at']}. Customer Name: {row['name']}. Product Department: {row['product_department']}. Product: {row['product_name']}. Category: {row['product_category']}. Price: ${row['sale_price']}. Stock quantity: {row['stock_quantity']}", axis=1)

# Document splitting
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
loader = DataFrameLoader(df, page_content_column="combined_info")
texts = text_splitter.split_documents(loader.load())

# Embeddings model
embeddings = GooglePalmEmbeddings(api_key=google_api_key)

# Vector DB
vectorstore = FAISS.from_documents(texts, embeddings)

# Chatbot Prompt Engineering
chatbot_template = """
You are a friendly, conversational retail shopping assistant that helps customers find products that match their preferences. 
From the following context and chat history, assist customers in finding what they are looking for based on their input. 
For each question, suggest three products, including their category, price, and current stock quantity.
Sort the answer by the cheapest product.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Chat history: {history}

Input: {question}
Your Response:
"""
chatbot_prompt = PromptTemplate(
    input_variables=["context", "history", "question"],
    template=chatbot_template
)

# Create the LangChain conversational chain
memory = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=True)

qa = RetrievalQA.from_chain_type(
    llm=GooglePalm(api_key=google_api_key, temperature=0.1),
    chain_type='stuff',
    retriever=vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": chatbot_prompt,
        "memory": memory
    }
)

# Define the steps for the feedback loop
feedback_steps = [
    {"question": "Could you please specify the product department?", "key": "department"},
    {"question": "Could you please specify the product category?", "key": "category"},
    {"question": "Could you please specify the product brand?", "key": "brand"},
    {"question": "What is your maximum price range?", "key": "price"},
    {"question": "Do you have any color preferences?", "key": "color"}
]

def chatbot_conversation():
    print("Chatbot is ready to assist you. Type 'exit' to end the conversation.")
    user_inputs = {}
    step = 0

    while True:
        if step < len(feedback_steps):
            user_input = input(f"Chatbot: {feedback_steps[step]['question']} ")
            if user_input.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break
            user_inputs[feedback_steps[step]['key']] = user_input
            step += 1
        else:
            # All steps completed, make a request with gathered inputs
            query = f"Department: {user_inputs.get('department', '')}, Category: {user_inputs.get('category', '')}, Brand: {user_inputs.get('brand', '')}, Price: {user_inputs.get('price', '')}, Color: {user_inputs.get('color', '')}"
            response = qa({"query": query})
            print(f"Chatbot: {response['result']}")
            user_feedback = input("Was this suggestion helpful? (yes/no): ")
            if user_feedback.lower() == 'no':
                print("Chatbot: Could you please provide more details on what you are looking for?")
                additional_details = input("You: ")
                query += f" {additional_details}"
                response = qa({"query": query})
                print(f"Chatbot: {response['result']}")
            else:
                step = 0
                user_inputs = {}  # Reset inputs for new conversation

if __name__ == "__main__":
    print("Setup complete. Ready to use the QA chain.")
    chatbot_conversation()
