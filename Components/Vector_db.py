import streamlit as st
import pinecone
from langchain.vectorstores import FAISS, Pinecone
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

pinecone.init(
    api_key="657b2a72-267f-4a50-8e66-36f61ef9d30b",
    environment = "gcp-starter"
)
index_name="mybot-learn"

# Assuming this function encodes the question into a vector representation
def encode_question(question,embeddings):
    # embeddings = HuggingFaceInstructEmbeddings()  # Instantiate the embeddings model
    question_vector = embeddings.embed_query(question)  # Encode the question into a vector
    return question_vector

def save_vector_store(text_chunks,embeddings, update=True):
    # embeddings = OpenAIEmbeddings()
    # model = INSTRUCTOR('hkunlp/instructor-base')
    # embeddings = model.encode(raw_text)
    # embeddings = HuggingFaceInstructEmbeddings()
    
    # Faiss
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    new_db = FAISS.load_local("faiss_index_V2", embeddings)
    new_db.merge_from(vectorstore)
    new_db.save_local('faiss_index_V2')

    # Pinecone
    # if index_name not in pinecone.list_indexes():
    # # we create a new index
    #     pinecone.create_index(name=index_name, metric="cosine", dimension=768)

    # if update==False:
    # vectorstore = Pinecone.from_texts(texts=text_chunks, embedding=embeddings, index_name=index_name)
    # else:
    #     vectorstore = Pinecone(index_name, embeddings.embed_query, text_chunks)

    

    return st.write("vector Store is Saved")