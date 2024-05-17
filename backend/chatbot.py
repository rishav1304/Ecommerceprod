# Import Libraries
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
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
df = pd.read_csv('bq-results-20240205-004748-1707094090486.csv').head(2000)

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

# Prompt Engineering
manual_template = """
Kindly suggest three similar products based on the description I have provided below:

Product Department: {department},
Product Category: {category},
Product Brand: {brand},
Maximum Price range: {price}.

Please provide complete answers including product department name, product category, product name, price, and stock quantity.
"""
prompt_manual = PromptTemplate(
    input_variables=["department", "category", "brand", "price"],
    template=manual_template
)

llm = GooglePalm(api_key=google_api_key, temperature=0.1)

chain = LLMChain(
    llm=llm,
    prompt=prompt_manual,
    verbose=True
)

# Chatbot Prompt Engineering
chatbot_template = """
You are a friendly, conversational retail shopping assistant that helps customers find products that match their preferences. 
From the following context and chat history, assist customers in finding what they are looking for based on their input. 
For each question, suggest three products, including their category, price, and current stock quantity.
Sort the answer by the cheapest product.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

chat history: {history}

input: {question}
Your Response:
"""
chatbot_prompt = PromptTemplate(
    input_variables=["context", "history", "question"],
    template=chatbot_template
)

# Create the LangChain conversational chain
memory = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=True)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": chatbot_prompt,
        "memory": memory
    }
)

if __name__ == "__main__":
    print("Setup complete. Ready to use the QA chain.")
