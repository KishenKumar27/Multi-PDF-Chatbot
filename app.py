import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import bot_template, user_template, css
from dotenv import load_dotenv
import os
from langchain_community.llms import HuggingFaceHub

load_dotenv()


def get_pdf_text(pdf_docs):
    text = ""
    document_sources = []
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            pdf_text += page_text
            # Add source information for better tracking
            text += f"\n\n--- Document: {pdf.name} (Page {page_num + 1}) ---\n"
            text += page_text
        
        document_sources.append({
            'name': pdf.name,
            'content_length': len(pdf_text),
            'pages': len(pdf_reader.pages)
        })
    
    # Store document info in session state for reference
    st.session_state.document_sources = document_sources
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, existing_memory=None):
    llm = ChatOpenAI(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"))

    # llm = HuggingFaceHub(
    #     repo_id='google/flan-t5-small',
    #     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    #     model_kwargs={
    #         "temperature": 0.5,
    #         "max_new_tokens": 512,
    #         "top_k": 50,
    #         "top_p": 0.95,
    #         "do_sample": True,
    #     }
    # )

    # Use existing memory if available, otherwise create new one
    if existing_memory is not None:
        memory = existing_memory
    else:
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"  # Specify which output to store in memory
        )
    
    # Configure retriever to get more documents for better multi-document coverage
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}  # Retrieve more chunks to ensure coverage from multiple documents
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  # Include source documents in response
        verbose=True  # Enable verbose mode for better debugging
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.warning("Please upload and process your PDFs first before asking questions.")

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)    

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "memory" not in st.session_state:
        st.session_state.memory = None
    
    st.header("Chat with multiple PDFs :books:")
    
    
    user_input = st.text_input("Ask a question about your documents:")

    if user_input:
        handle_userinput(user_input)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    # Get PDF Text
                    raw_text = get_pdf_text(pdf_docs)

                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    st.success(f"Processed {len(pdf_docs)} documents into {len(text_chunks)} text chunks")

                    # Create the vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # Extract existing memory if conversation exists
                    existing_memory = None
                    if st.session_state.conversation is not None:
                        existing_memory = st.session_state.conversation.memory
                    
                    # Create the conversation chain with preserved memory
                    st.session_state.conversation = get_conversation_chain(vectorstore, existing_memory)
                    
                    # Store memory reference for future use
                    st.session_state.memory = st.session_state.conversation.memory
            else:
                st.warning("Please upload at least one PDF file before processing.")
        
        # Display processed documents info
        if hasattr(st.session_state, 'document_sources') and st.session_state.document_sources:
            st.subheader("Processed Documents:")
            for doc in st.session_state.document_sources:
                st.write(f"📄 **{doc['name']}**")
                st.write(f"   - Pages: {doc['pages']}")
                st.write(f"   - Content length: {doc['content_length']:,} characters")


if __name__ == "__main__":
    main()
