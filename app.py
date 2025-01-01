import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

# preprocessing data
category_map = {
    15: 'Java Developer',
    23: 'Testing',
    8: 'DevOps Engineer',
    20: 'Python Developer',
    24: 'Web Designing',
    12: 'HR',
    13: 'Hadoop',
    3: 'Blockchain',
    10: 'ETL Developer',
    18: 'Operations Manager',
    6: 'Data Science',
    22: 'Sales',
    16: 'Mechanical Engineer',
    1: 'Arts',
    7: 'Database',
    11: 'Electrical Engineering',
    14: 'Health and Fitness',
    19: 'PMO',
    4: 'Business Analyst',
    9: 'DotNet Developer',
    2: 'Automation Testing',
    17: 'Network Security Engineer',
    21: 'SAP Developer',
    5: 'Civil Engineer',
    0: 'Advocate'
}
def clean_resume(text):
    clean_text = re.sub(r'http\S+\s',' ', text)  #url
    clean_text = re.sub(r'#\S+\s',' ', clean_text)
    clean_text = re.sub(r'@\S+',' ', clean_text)
    clean_text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]',' ', clean_text)
    clean_text = re.sub(r'\s+',' ', clean_text)
    return clean_text

# web app
def main():
    st.title('Resume Analyzer')
    upload_file = st.file_uploader('Upload Resume', type=['txt'])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text =  resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # if fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        category_name = category_map.get(prediction_id,'Unknown')
        st.write("Predicted Category: ",category_name)

# main
if __name__ == '__main__' :
    main()