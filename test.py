import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
# Load the document as a string
with open('CISI.ALLnettoye', 'r') as file:
    text = file.read()

term_frequency = {}  #tạo BoW
documents = re.split(r'\.I\s+', text)[1:]
stemmer = PorterStemmer()

for doc in documents:
    lines = doc.strip().split('\n')
    doc_id = lines[0].strip()  # cắt chữ số đầu tiên của văn bản
    lines = lines[1:]
    doc_text = ' '.join(lines) # Join the remaining lines into a single string ( ghép các dòng lại thành 1 văn bản như cũ )
    tokens = nltk.word_tokenize(doc_text.lower())       # Tokenize the document text

