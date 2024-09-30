import streamlit as st
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import validators
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.schema import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

st.title("Chat with Youtube Video")

if 'chain' not in st.session_state:
    st.session_state['chain'] = None
if 'store' not in st.session_state:
    st.session_state['store'] = {}
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'content_retrieved' not in st.session_state:
    st.session_state['content_retrieved'] = False
if 'messages' not in st.session_state:
    st.session_state['messages'] = []


model = ChatGroq(model='llama3-70b-8192')

def get_document_content(url):
    if 'youtube.com' in url or 'youtu.be' in url:
        return YoutubeLoader.from_youtube_url(url, add_video_info=False, language=['en', 'hi'], translation="en").load()
    else:
        return UnstructuredURLLoader(url).load()

def get_vector_store(url):
    docs = get_document_content(url)
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    embeddings = OpenAIEmbeddings()
    st.session_state['vector_store'] = FAISS.from_documents(documents, embedding=embeddings)

# URL input and validation
url = st.text_input("🔗 Enter a URL", placeholder="https://...")

if url and validators.url(url) and not st.session_state['content_retrieved']:
    with st.spinner("Retrieving content..."):
        get_vector_store(url)
        
        if st.session_state['vector_store'] is not None:
            retriever = st.session_state['vector_store'].as_retriever()

            standalone_prompt = """Given chat history and user question which might reference context in the history, create a standalone question that can be understood without the chat history. Do not answer the question, just create a question. If not needed to reformulate the question, return the original question."""

            standalone_q = ChatPromptTemplate.from_messages(
                [
                    ("system", standalone_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )

            history_retriever = create_history_aware_retriever(model, retriever, standalone_q)

            prompt = """
                You are a helpful chatbot which helps the person know more about the content of a youtube video.
                Given a question, use the context to answer it.
                Never mention the word "context" in the answer, as it is the content of a video.
                Always answer using the context, dont make up any answer on your own.
                If you don't know the answer, say you don't know. 
                Try to answer the question in the same way and tone as the context.
                Keep the answer concise until and unless asked to answer in detail. 
            <context>{context}</context>
            """

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )

            document_retriever = create_stuff_documents_chain(model, prompt=prompt_template)

            rag_chain = create_retrieval_chain(history_retriever, document_retriever)

            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                return st.session_state['store'].get(session_id, ChatMessageHistory())

            chain = RunnableWithMessageHistory(
                rag_chain, 
                get_session_history,
                history_messages_key="chat_history",
                input_messages_key="input", 
                output_messages_key="answer"
            )
            
            st.session_state['chain'] = chain
            st.session_state['content_retrieved'] = True
            st.session_state['messages'].append({"role": "assistant", "content": "Hello! I am your assistant. What questions do you have about the content?"})
    
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state['chain'] is not None:
    user_input = st.chat_input("Ask a question")
    if user_input:
        st.session_state['messages'].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Thinking..."):
            response = st.session_state['chain'].invoke(
                {"input": user_input}, 
                config={"configurable": {"session_id": "1"}}
            )
        
        st.session_state['messages'].append({"role": "assistant", "content": response['answer']})
        with st.chat_message("assistant"):
            st.write(response['answer'])

elif not validators.url(url) and url:
    st.warning("⚠️ Please enter a valid URL.")
else:
    st.error("🚀 Please enter a URL to get started.")
