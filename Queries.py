import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from Document import vectorizer, indexing_terms,stop_words

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Load the document as a string
with open('CISI_dev.QRY', 'r') as file:
    query_text = file.read()
    query_texts = re.split(r"\.I \d+", query_text)[1:]

stemmer = PorterStemmer()
processed_tokens = []
queries=[]
vectorizer = TfidfVectorizer(vocabulary=indexing_terms, sublinear_tf=True)
# Print the individual queries but number start from 0 to n-1
for query in query_texts:
    # Extract title if available
    title_match = re.search(r"\.T\n(.+?)\n", query)
    title = title_match.group(1) if title_match else ""
    # Remove .A and its content if available
    if re.search(r"\.A\n", query):
        query = re.sub(r"\.A\n(.+?)\n", "", query)
    # Extract content if available
    content_match = re.search(r"\.W\n(.+?)\n", query)
    content = content_match.group(1) if content_match else ""
    # Remove .B and its content if available
    if re.search(r"\.B\n", query):
        query = re.sub(r"\.B\n(.+?)\n", "", query)
    queries.append((title, content))

    # Tokenize and preprocess the query
    tokens = nltk.word_tokenize(query)
    processed_tokens_query = []
    for token in tokens:
        token = token.lower()
        if token not in stop_words and re.match(r'^[a-zA-Z0-9]+$', token):
            token = stemmer.stem(token)
            # Remove unwanted characters
            processed_tokens_query.append(token)
    processed_tokens.append(processed_tokens_query)
    # print(processed_tokens_query)

    query_strings = [' '.join(tokens) for tokens in processed_tokens]    

    # Fit the vectorizer to the query strings and transform the strings into a sparse matrix
    #query_matrix = vectorizer.transform([' '.join(tokens) for tokens in processed_tokens_query])
    query_matrix = vectorizer.fit_transform(query_strings)
    # query_matrix = vectorizer.transform(query_strings)