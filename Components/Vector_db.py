import streamlit as st
from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

# Assuming this function encodes the question into a vector representation
def encode_question(question,embeddings):
    # embeddings = HuggingFaceInstructEmbeddings()  # Instantiate the embeddings model
    question_vector = embeddings.embed_query(question)  # Encode the question into a vector
    return question_vector

def save_vector_store(text_chunks,embeddings):
    # embeddings = OpenAIEmbeddings()
    # model = INSTRUCTOR('hkunlp/instructor-base')
    # embeddings = model.encode(raw_text)
    # embeddings = HuggingFaceInstructEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    new_db = FAISS.load_local("faiss_index_V2", embeddings)
    new_db.merge_from(vectorstore)
    new_db.save_local('faiss_index_V2')

    return st.write("vector Store is Saved")