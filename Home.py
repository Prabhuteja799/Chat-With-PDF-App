import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.markdown(
    
    
    """
    
     


    # Welcome to the Chat-With_PDF ! 👋

    ## Features of this App:
    - **Chat with PDF**: Upload a PDF and ask questions about its content.
    - **Summarization**: Generate concise summaries of your PDF documents.
    
    ## How to Use:
    1. Navigate to the "Chat with PDF" tab to ask questions about a document.
    2. Use the "Summarization" tab to create a summarized version of your PDF.
    3. Upload a valid PDF in the respective tab to get started.

    ## About:
    This app leverages AI-powered tools to make working with PDFs easy and efficient. Built using Streamlit, LangChain, and OpenAI APIs.
    
    """,
    unsafe_allow_html=True

)
