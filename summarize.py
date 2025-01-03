import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import math
import streamlit as st
from PyPDF2 import PdfReader
import docx

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Function to create frequency table
def __create_frequency_table(text_string) -> dict:
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    return freqTable

# Function to create frequency matrix for sentences
def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue
            if word in freq_table:
                freq_table[word] = 1 + freq_table.get(word, 0)
            else:
                freq_table[word] = 1
        frequency_matrix[sent[:15]] = freq_table
    return frequency_matrix

# Function to create tf matrix
def _create_tf_matrix(freq_matrix):
    tf_matrix = {}
    for sent, f_table in freq_matrix.items():
        tf_table = {}
        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence
        tf_matrix[sent] = tf_table
    return tf_matrix

# Function to create document per word count
def _create_document_per_words(freq_matrix):
    word_per_doc_table = {}
    for sent, f_table in freq_matrix.items():
        for word in f_table.keys():
            word_per_doc_table[word] = word_per_doc_table.get(word, 0) + 1
    return word_per_doc_table

# Function to create idf matrix
def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}
    for sent, f_table in freq_matrix.items():
        idf_table = {}
        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))
        idf_matrix[sent] = idf_table
    return idf_matrix

# Function to create tf-idf matrix
def create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)
        tf_idf_matrix[sent1] = tf_idf_table
    return tf_idf_matrix

# Function to score sentences based on tf-idf matrix
def _score_sentences(tf_idf_matrix) -> dict:
    sentenceValue = {}
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = sum(f_table.values())
        count_words_in_sentence = len(f_table)
        if count_words_in_sentence != 0:
            sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
    return sentenceValue

# Function to calculate average score
def _find_average_score(sentenceValue) -> float:
    return sum(sentenceValue.values()) / len(sentenceValue)

# Function to generate summary based on threshold
def _generate_summary(sentences, sentenceValue, threshold):
    summary = ''
    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= threshold:
            summary += " " + sentence
    return summary.strip()

# Function to read PDF file
def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

# Function to read DOCX file
def read_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Streamlit App
def main():
    st.title("Text Summarization with TF-IDF")

    # File uploader
    uploaded_file = st.file_uploader("Upload a file (TXT, PDF, DOCX):", type=["txt", "pdf", "docx"])
    text_input = ""

    if uploaded_file is not None:
        file_type = uploaded_file.type
        if file_type == "text/plain":
            text_input = uploaded_file.read().decode("utf-8")
        elif file_type == "application/pdf":
            text_input = read_pdf(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text_input = read_docx(uploaded_file)
        else:
            st.error("Unsupported file type")

    # Text area for manual input
    text_area_input = st.text_area("Or paste your text here:")

    if text_input == "" and text_area_input != "":
        text_input = text_area_input

    if text_input:
        st.write("### Input Text:")
        st.write(text_input)

        sentences = sent_tokenize(text_input)
        total_documents = len(sentences)

        # Process
        freq_matrix = _create_frequency_matrix(sentences)
        tf_matrix = _create_tf_matrix(freq_matrix)
        count_doc_per_words = _create_document_per_words(freq_matrix)
        idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
        tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)
        sentence_scores = _score_sentences(tf_idf_matrix)
        avg_score = _find_average_score(sentence_scores)

        # Adjust threshold for summarization
        threshold = st.slider("Select summarization threshold:", 
                              min_value=0.0, max_value=1.0, value=avg_score, step=0.01)

        if st.button("Generate Summary"):
            summary = _generate_summary(sentences, sentence_scores, threshold)

            # Display results
            st.subheader("Summary")
            st.write(summary)

if __name__ == "__main__":
    main()
