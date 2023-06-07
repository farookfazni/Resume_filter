import streamlit as st
from PyPDF2 import PdfReader
from PreprocessText import preprocess_text
from FindKeyword import FindKeyWord
from streamlit_extras.add_vertical_space import add_vertical_space

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model 
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Fazni Farook')

# Main body
def main():
    st.header("Resume Filter using Keywords 💬")

    # upload a PDF file
    pdfs = st.file_uploader("Upload your Resumes", type='pdf',accept_multiple_files=True)
    
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
            keyword = "Computer Science"
            Text_with_Keyword = FindKeyWord(keyword, text)
            # Appending to array
            all_text.append({"filename": filename, "text": Text_with_Keyword})
        # st.write(all_text, unsafe_allow_html=True)
        # st.markdown(all_text, unsafe_allow_html=True)
        for item in all_text:
            st.markdown(f"**Filename: {item['filename']}**")
            st.markdown(item['text'], unsafe_allow_html=True)
            st.markdown("---")

if __name__=='__main__':
    main()