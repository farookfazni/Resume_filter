import re

def FindKeyWord(keyword, text):
    if re.search(r'\b({0})\b'.format(re.escape(keyword)), text, flags=re.IGNORECASE):
        highlighted_text = re.sub(r'\b({0})\b'.format(re.escape(keyword)), r'<mark style="background-color: yellow;">\1</mark>', text, flags=re.IGNORECASE)
        return highlighted_text
    else:
         return "Keyword not found in the Resume."