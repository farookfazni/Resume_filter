import re
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from FindKeyword import FindKeyWords
from PreprocessText import preprocess_text
from model_Responce import model_prediction
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from InstructorEmbedding import INSTRUCTOR
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Assuming this function encodes the question into a vector representation
def encode_question(question):
    embeddings = HuggingFaceInstructEmbeddings()  # Instantiate the embeddings model
    question_vector = embeddings.embed_query(question)  # Encode the question into a vector
    return question_vector

# def handle_user_input(question):
#     response = st.session_state.conversation({'question':question})
#     st.session_state.chat_history = response('chat_history')

#     for i,message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)

# def get_conversation_chain(vector_store):
#     llm = ChatOpenAI()
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vector_store.as_retriever(),
#         memory = memory
#     )
#     return conversation_chain

def save_vector_store(text_chunks):
    # embeddings = OpenAIEmbeddings()
    # model = INSTRUCTOR('hkunlp/instructor-base')
    # embeddings = model.encode(raw_text)
    embeddings = HuggingFaceInstructEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    new_db.merge_from(vectorstore)
    new_db.save_local('faiss_index')

    return st.write("vector Store is Saved")

def button_function(all_text):
    # Add your desired functionality here
    # predictions = []
    for item in all_text:
        text = item['text']
        # filename = item['filename']
        pred = model_prediction(text)
        # predictions.append({"filename": filename, "prediction": pred})
        item['prediction'] = pred
    return all_text

def get_pdf_text(pdfs,preprocess=True):
    if preprocess:
        all_text = []
        for pdf in pdfs:
            # Process each uploaded PDF file
            # Reading PDF
            pdf_reader = PdfReader(pdf)

            # Get the filename of the PDF
            filename = pdf.name
            
            text = ""
            # Reading Each Page
            for page in pdf_reader.pages:
                # Extracting Text in Every Page
                text += page.extract_text()
            # Preprocess the text
            text = preprocess_text(text)
            # Appending to array
            all_text.append({"filename": filename, "text": text})
        return all_text
    
    else:
        text = ""
        for pdf in pdfs:
            # Process each uploaded PDF file
            # Reading PDF
            pdf_reader = PdfReader(pdf)

            # Reading Each Page
            for page in pdf_reader.pages:
                # Extracting Text in Every Page
                extract_text = page.extract_text()
                text += preprocess_text(extract_text)
        return text

def filter_keywords(all_text, keywords):
    filtered_text = []
    for item in all_text:
        filename = item['filename']
        text = item['text']
        filtered_text_with_keywords = FindKeyWords(keywords, text)
        filtered_text.append({"filename": filename, "text": filtered_text_with_keywords})
    return filtered_text

            
# Main body
def main():
    # vector_store = None
    load_dotenv()
    st.header("Resume Filter using Keywords 💬")

    # Sidebar contents
    with st.sidebar:
        st.title('🤗💬 LLM Chat App')
        # upload a PDF file
        pdfs = st.file_uploader("Upload your Resumes", type='pdf',accept_multiple_files=True)

        # Get user preference for matching keywords
        # match_all_keywords = st.checkbox("Match All Keywords")

        # Choose functionality: Prediction or Filtering
        functionality = st.radio("Choose functionality:", ("Make Predictions", "Filter Keywords","Predict the Suitable canditate","Ask Questions"))
        if functionality == "Ask Questions":
            if st.button('Process'):
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = get_pdf_text(pdfs, preprocess=False)

                    # get the text chunk
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    save_vector_store(text_chunks)
        add_vertical_space(5)
        st.write('Made with ❤️ by Fazni Farook')


    if pdfs is not None:
        all_text = get_pdf_text(pdfs)

        # if 'conversation' not in st.session_state:
        #     st.session_state.conversation = None

        # if 'chat_history' not in st.session_state:
        #     st.session_state.chat_history = None

        if functionality == "Make Predictions":
            if st.button('Make Prediction'):
                with st.spinner("Progressing"):
                    all_text = button_function(all_text)

                    for item in all_text:
                        filename = item["filename"]
                        text = item["text"]
                        pred = item["prediction"]
                        st.markdown(f"**Filename: {filename}**")
                        # st.markdown(text, unsafe_allow_html=True)
                        st.markdown(f"**Prediction: {pred}**")
                        st.markdown("---")

        elif functionality == "Filter Keywords":
            # getting the keywords
            keyword_input  = st.text_input("Keyword")
            keywords = [keyword.strip() for keyword in keyword_input.split(",")]

            if st.button('Filter Keywords'):
                with st.spinner("Progressing"):
                    filtered_text = filter_keywords(all_text, keywords)

                    for item in filtered_text:
                        filename = item["filename"]
                        text = item["text"]
                        st.markdown(f"**Filename: {filename}**")
                        st.markdown(text, unsafe_allow_html=True)
                        st.markdown("---")

        elif functionality == "Predict the Suitable canditate":
            # getting the keywords
            keyword  = st.text_input("Keyword")

            if st.button('Filter Resumes'):
                with st.spinner("Progressing"):
                    all_text = button_function(all_text)
                    # filtered_text = filter_keywords(all_text, keywords)
                    count = 0
                    for item in all_text:
                        filename = item["filename"]
                        prediction = item["prediction"]
                        if keyword.lower()==prediction.lower():
                            count+=1
                            st.markdown(f"**Filename: {filename}**")
                            st.markdown(prediction, unsafe_allow_html=True)
                            st.markdown("---")
                    
                    if count==0:
                        st.markdown("No match found")

        elif functionality == "Ask Questions":

            embeddings = HuggingFaceInstructEmbeddings()

            new_db = FAISS.load_local("faiss_index", embeddings)

            st.write(css,unsafe_allow_html=True)

            # create conversation chain
            # st.session_state.conversation = get_conversation_chain(vector_store)

            question = st.text_input("Ask Question")

            if st.button('Ask Question'):
                with st.spinner("Processing"):
                    if question:
                        # Convert the question to a vector
                        question_vector = encode_question(question)

                        # Convert the vector store to a compatible format
                        output = new_db.similarity_search_by_vector(question_vector)
                        st.write(output[0])
                
if __name__=='__main__': 
    main()