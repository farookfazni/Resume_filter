import streamlit as st
from dotenv import load_dotenv
from Components.FindKeyword import filter_keywords
from Components.PreprocessText import get_pdf_text
from Components.model_Responce import model_prediction
from Components.GooglePalmChat import get_qa_chain
from Components.Vector_db import encode_question, save_vector_store
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
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
        # if functionality == "Ask Questions":
            
        add_vertical_space(5)
        st.write('Made with ❤️ by Fazni Farook')

    vector_store = None
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

            # new_db = FAISS.load_local("faiss_index_V2", embeddings)

            if st.button('Create Knowledgebase'):
                with st.spinner("Processing"):
                    # embeddings = HuggingFaceInstructEmbeddings()
                    # get pdf text
                    raw_text = get_pdf_text(pdfs, preprocess=False)

                    # get the text chunk
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    save_vector_store(text_chunks,embeddings)

            st.write(css,unsafe_allow_html=True)

            # create conversation chain
            # st.session_state.conversation = get_conversation_chain(vector_store)

            question = st.text_input("Ask Question: ")

            if st.button('Ask Question'):
                with st.spinner("Processing"):
                    if question:
                        # Convert the question to a vector
                        # question_vector = encode_question(question,embeddings)

                        # Convert the vector store to a compatible format
                        # output = new_db.similarity_search_by_vector(question_vector)
                        # page_content = output[0].page_content

                        
                        # Asking Questions using Google Palm
                        chain = get_qa_chain(embeddings)
                        # docs = vector_store.similarity_search(question)
                        response = chain(question)
                        st.header("Answer: ")
                        st.write(response["result"])
                
if __name__=='__main__': 
    main()