# Import Libraries
import streamlit as st
import requests

# Set up Streamlit app
st.title("Ecommerce Product Recommendation with ChatGPT")

# Define the FastAPI server URL
api_url = "http://127.0.0.1:8000"

# Sidebar for manual input
st.sidebar.header("Manual Product Recommendation")
department = st.sidebar.text_input("Product Department")
category = st.sidebar.text_input("Product Category")
brand = st.sidebar.text_input("Product Brand")
price = st.sidebar.text_input("Maximum Price Range")

if st.sidebar.button("Get Recommendations"):
    # Send a request to the /manual endpoint
    manual_payload = {
        "department": department,
        "category": category,
        "brand": brand,
        "price": price
    }
    response = requests.post(f"{api_url}/manual", json=manual_payload)
    if response.status_code == 200:
        st.sidebar.success("Recommendations received!")
        recommendations = response.json()
        st.sidebar.write(recommendations)
    else:
        st.sidebar.error("Error fetching recommendations")

# Main area for chatbot interaction
st.header("Chatbot Product Recommendation")
query = st.text_input("Ask a question to the chatbot")

if st.button("Get Answer"):
    # Send a request to the /chatbot endpoint
    chatbot_payload = {"query": query}
    response = requests.post(f"{api_url}/chatbot", json=chatbot_payload)
    if response.status_code == 200:
        st.success("Answer received!")
        answer = response.json()
        st.write(answer)
    else:
        st.error("Error fetching answer")

if __name__ == "__main__":
    st.write("Streamlit app is ready to interact with the FastAPI backend.")
