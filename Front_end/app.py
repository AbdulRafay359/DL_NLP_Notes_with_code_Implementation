import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import PyPDF2

nltk.download('punkt')
nltk.download('stopwords')

knn = pickle.load(open('knn.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))

stop_words = set(stopwords.words('english'))

def cleanResume(text):
    cleantext = re.sub(r'http\S+|www\S+', '', text)
    cleantext = re.sub(r'[@#]\w+', '', cleantext)
    cleantext = re.sub(r'[^A-Za-z0-9\s]', '', cleantext)
    cleantext = re.sub(r'[^\w\s]', '', cleantext)
    new_text = [w for w in cleantext.split() if w.lower() not in stop_words]
    cleantext = ' '.join(new_text)
    return cleantext

def main():
    st.title("Resume Classification App")
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #00bfff;
        color:black;
    }
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {background-color: #00509e !important;}
    div.stFileUploader > div:first-child {
        background-color: #0077b6 !important;
        color: black;
        border-radius: 10px;
        padding: 10px;
    }
    div.stFileUploader input[type="file"] {
        color: black;
        border-radius: 5px;
        padding: 5px;
        background-color: #0057b7 !important;
    }
    div.stFileUploader input[type="file"]::-webkit-file-upload-button {
        background-color: #0057b7 !important;
        color: black !important;
        border: none;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
    }
    div.stFileUploader label {
        font-weight: bold;
        font-size: 18px;
        color: black;
    }
    div.stFileUploader:hover > div:first-child {
        background-color: #0057b7 !important;
    }
    .subtitle {
        font-size: 24px;
        font-weight: bold;
        color: black;
        margin-bottom: 10px;
    }
    </style>
    """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="subtitle">Check What your Resume says about you</div>', unsafe_allow_html=True)

    upload_file = st.file_uploader('Upload Your Resume', type=['pdf', 'txt'])

    if upload_file is not None:
        resume_text = ""
        if upload_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(upload_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    resume_text += text
        else:
            try:
                resume_text = upload_file.read().decode('utf-8')
            except UnicodeDecodeError:
                resume_text = upload_file.read().decode('latin1')

        if resume_text:
            clean_resume = cleanResume(resume_text)
            vectorized_resume = tfidf.transform([clean_resume])
            prediction = knn.predict(vectorized_resume)[0]
            original_category = le.inverse_transform([prediction])[0]
            st.write('This person Belong from field : ', original_category)
if __name__ == "__main__":
    main()
