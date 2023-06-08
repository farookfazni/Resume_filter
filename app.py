import streamlit as st
from PyPDF2 import PdfReader
from FindKeyword import FindKeyWords
from PreprocessText import preprocess_text
from model_Responce import model_prediction
from streamlit_extras.add_vertical_space import add_vertical_space

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - Custom Trained AI model 
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Fazni Farook')

def button_function(all_text):
    # Add your desired functionality here
    # predictions = []
    for item in all_text:
        text = item['text']
        # filename = item['filename']
        pred = model_prediction(text)
        # predictions.append({"filename": filename, "prediction": pred})
        item['prediction'] = pred
    # st.write(predictions)
    return all_text

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
    st.header("Resume Filter using Keywords 💬")

    # upload a PDF file
    pdfs = st.file_uploader("Upload your Resumes", type='pdf',accept_multiple_files=True)



    # Get user preference for matching keywords
    # match_all_keywords = st.checkbox("Match All Keywords")

     # Choose functionality: Prediction or Filtering
    functionality = st.radio("Choose functionality:", ("Make Predictions", "Filter Keywords"))

    if pdfs is not None:
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

        if functionality == "Make Predictions":
            if st.button('Make Prediction'):
                all_text = button_function(all_text)

                for item in all_text:
                    filename = item["filename"]
                    text = item["text"]
                    pred = item["prediction"]
                    st.markdown(f"**Filename: {filename}**")
                    st.markdown(text, unsafe_allow_html=True)
                    st.markdown(f"**Prediction: {pred}**")
                    st.markdown("---")
        elif functionality == "Filter Keywords":
            # getting the keywords
            keyword_input  = st.text_input("Keyword")
            keywords = [keyword.strip() for keyword in keyword_input.split(",")]

            if st.button('Filter Keywords'):
                filtered_text = filter_keywords(all_text, keywords)

                for item in filtered_text:
                    filename = item["filename"]
                    text = item["text"]
                    st.markdown(f"**Filename: {filename}**")
                    st.markdown(text, unsafe_allow_html=True)
                    st.markdown("---")

if __name__=='__main__':
    main()