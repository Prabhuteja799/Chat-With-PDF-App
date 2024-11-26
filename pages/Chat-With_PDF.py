from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import hashlib

def initialize_session_state():
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "input" not in st.session_state:
        st.session_state["input"] = ""
    if "current_pdf_hash" not in st.session_state:
        st.session_state.current_pdf_hash = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "pdf_content" not in st.session_state:
        st.session_state.pdf_content = ""

def process_pdf(pdf, persist_directory):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    st.session_state.pdf_content = text
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)
    
    knowledge_base = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    knowledge_base.persist()
    return knowledge_base

def get_answer_from_chroma(knowledge_base, question, min_similarity_score=0.1):
    """Get answer from ChromaDB with adjusted thresholds"""
    embeddings = OpenAIEmbeddings()
    
    # Get more documents to ensure we don't miss relevant content
    docs = knowledge_base.similarity_search(question, k=4)
    
    if not docs:
        return None
    
    # Custom QA prompt that focuses on PDF content but is less restrictive
    qa_prompt = PromptTemplate(
        template="""Use the following pieces of context to answer the question. 
        If you cannot find the specific information in the context to answer the question accurately, 
        say "I cannot find the specific information in the PDF to answer this question."
        
        Context: {context}
        
        Question: {question}
        
        Answer:""",
        input_variables=["context", "question"]
    )
    
    llm = OpenAI(temperature=0.2)  # Slightly higher temperature for more natural responses
    chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
    
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question)
    
    return response

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chat", page_icon="📚")
    st.header("PDF Question Answering System 💬")
    
    initialize_session_state()
    persist_directory = "./chroma_storage"
    
    # File upload
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        current_hash = hashlib.md5(pdf.read()).hexdigest()
        pdf.seek(0)
        
        if current_hash != st.session_state.current_pdf_hash:
            with st.spinner("Processing PDF..."):
                knowledge_base = process_pdf(pdf, persist_directory)
                st.session_state.current_pdf_hash = current_hash
                st.session_state.pdf_processed = True
                st.success("PDF processed successfully!")
    
    # Question input
    user_question = st.text_input("Ask a question about your PDF:", value=st.session_state.input)
    
    if user_question:
        if not st.session_state.pdf_processed:
            st.warning("Please upload a PDF file first!")
            return
        
        with st.spinner("Searching for answer..."):
            embeddings = OpenAIEmbeddings()
            knowledge_base = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            
            # Get answer
            response = get_answer_from_chroma(knowledge_base, user_question)
            
            if response and not response.startswith("I cannot find"):
                # Update conversation history
                st.session_state.past.append(user_question)
                st.session_state.generated.append(response)
                st.session_state.input = ""
                
                # Display the response
                st.write("**Answer:**")
                st.write(response)
            else:
                st.warning("I cannot find the specific information in the PDF to answer this question.")
    
    # Display conversation history
    with st.expander("Conversation History", expanded=True):
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            st.info(st.session_state["past"][i], icon="🧐")
            st.success(st.session_state["generated"][i], icon="🤖")
    


if __name__ == '__main__':
    main()