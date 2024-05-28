# Import libraries
import streamlit as st
import requests
from chatbot import qa, RetrievalQA

# Define the FastAPI endpoint URL
fastapi_url = "http://127.0.0.1:8000"

def main():
    # Sidebar contents
    st.sidebar.title("Product Recommendation App Demo")
    st.sidebar.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM model
        
        You can use the two methods listed below to use this application.
    ''')

    def manual():
        st.header('🛍️ Product Recommendation App 🛍️')
        st.write('')
        st.write('Please fill in the fields below.')
        st.write('')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            department = st.text_input("Product Department: ")
        with col2:
            category = st.text_input("Product Category: ")
        with col3:
            brand = st.text_input("Product Brand: ")
        with col4:
            price = st.number_input("Maximum price: ", min_value=0, max_value=1000)

        if st.button('Get recommendations'):
            with st.spinner("Just a moment..."):
                # Make a request to FastAPI endpoint
                item = {
                    'department': department,
                    'category': category,
                    'brand': brand,
                    'price': f"${price}"
                }
                response = requests.post(fastapi_url + "/manual", json=item)
                
                # Check if the request was successful
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('response', 'No response')
                    st.success(response_text)
                else:
                    st.error("Error fetching recommendations. Please try again.")

    def ask_follow_up(user_input):
        if 'product department' not in user_input.lower():
            return "Could you please specify the product department?"
        if 'product category' not in user_input.lower():
            return "Could you please specify the product category?"
        if 'brand' not in user_input.lower():
            return "Do you have a preferred brand?"
        if 'price' not in user_input.lower():
            return "What is your maximum price range?"
        return None

    def chatbot():
        st.header("🤖 Product Recommendation Chatbot 🤖")

        user = "User"
        assistant = "Assistant"
        message = "Messages"

        if message not in st.session_state:
            st.session_state[message] = [{"actor": assistant, "payload": "Hi! How can I help you? 😀"}]

        # Display messages
        for msg in st.session_state[message]:
            if msg["actor"] == assistant:
                st.write(f"**{assistant}:** {msg['payload']}")
            else:
                st.write(f"**{user}:** {msg['payload']}")

        # Prompt
        prompt: str = st.text_input("Enter a prompt here")

        if prompt:
            with st.spinner("Just a moment..."):
                follow_up = ask_follow_up(prompt)
                if follow_up:
                    st.session_state[message].append({"actor": assistant, "payload": follow_up})
                    st.write(f"**{assistant}:** {follow_up}")
                else:
                    # Response
                    response = requests.post(fastapi_url + "/chatbot", json={"query": prompt})
                    if response.status_code == 200:
                        result = response.json()
                        response_text = result.get('response', 'No response')

                        # Update session state
                        st.session_state[message].append({"actor": user, "payload": prompt})
                        st.write(f"**{user}:** {prompt}")
                        st.session_state[message].append({"actor": assistant, "payload": response_text})
                        st.write(f"**{assistant}:** {response_text}")
                    else:
                        st.error("Error fetching answer. Please try again.")

    # Radio button to choose between manual or chatbot:
    mode = st.sidebar.radio("Choose Mode", ["Manual Input 🛍️", "ChatBot 🤖"])

    # Conditionally display the appropriate prediction form
    if mode == "Manual Input 🛍️":
        manual()
    elif mode == "ChatBot 🤖":
        chatbot()

    st.sidebar.markdown(''' 
        ## Created by: 
        
    ''')

if __name__ == '__main__':
    main()
