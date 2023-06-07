import re

def preprocess_text(text):
    # Remove newlines and tabs
    text = re.sub(r'\n|\t', '', text)

    # Remove letter combinations between spaces
    text = re.sub(r'\s[A-Z]\s', ' ', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove dates in the format DD-MM-YYYY or DD/MM/YYYY
    text = re.sub(r'\d{2}[-/]\d{2}[-/]\d{4}', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\+\d{2}\s?\d{2,3}\s?\d{3,4}\s?\d{4}', '', text)
    
    # Remove specific text format
    text = re.sub(r'Issued\s\w+\s\d{4}Credential ID \w+', '', text)

    # Remove extra spaces between words
    text = re.sub(r'\s+', ' ', text)
    
    # Add a space before a word containing a capital letter in the middle
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    
    return text