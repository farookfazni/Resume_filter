import re
def FindKeyWords(keywords, text):
    highlighted_text = text

    for keyword in keywords:
        if re.search(r'\b({0})\b'.format(re.escape(keyword)), highlighted_text, flags=re.IGNORECASE):
            highlighted_text = re.sub(r'\b({0})\b'.format(re.escape(keyword)), r'<mark style="background-color: yellow;">\1</mark>', highlighted_text, flags=re.IGNORECASE)
        else:
            return "Keyword not found in the Resume."

    return highlighted_text

def filter_keywords(all_text, keywords):
    filtered_text = []
    for item in all_text:
        filename = item['filename']
        text = item['text']
        filtered_text_with_keywords = FindKeyWords(keywords, text)
        filtered_text.append({"filename": filename, "text": filtered_text_with_keywords})
    return filtered_text