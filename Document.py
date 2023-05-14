import nltk
import re
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
# Load the document as a string
with open('CISI.ALLnettoye', 'r') as file:
    text = file.read()

#1.2 Create a dictionary to store the frequency count of each term across all documents
term_frequency = {}

# Split the text into individual queries using the ".I " delimiter
documents = re.split(r'\.I\s+', text)[1:]
#text =re.split(r'\n')
#essayé coupé le titre
stemmer = PorterStemmer()

processed_tokens = []
# Print the individual queries but number start from 0 to n-1
for document in documents:
    #doc_id, doc_text = queries.split('\n', 1)
    #couper le titre
    #print(document.strip())
    tokens = nltk.word_tokenize(document)
    processed_tokens_doc = []
    for token in tokens:
        token = token.lower()
        token = stemmer.stem(token)
        # Add processed tokens to the list
        processed_tokens_doc.append(token)
    processed_tokens.append(processed_tokens_doc)
    #each doc has their own tokens has been processed
    #print(processed_tokens_doc)

for document in processed_tokens:
    for token in document:
        if token in term_frequency:
            term_frequency[token] += 1
        else:
            term_frequency[token] = 1

#rearrange the word with the most frequency
#thay vi (...1) (..10) thi sap xep lai
sorted_terms = sorted(term_frequency.items(), key=lambda x: x[1], reverse=True)
#print(sorted_terms)


# Choose the top k terms as indexing terms
#loc gia tri co frequency>=k
k = 100 # NEED TO CHANGE IF NEEDED
indexing_terms = [term[0] for term in sorted_terms[:k]]
#print(indexing_terms)

#1.3

"""
# Create a TfidfVectorizer object with the desired options
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the documents
vectorizer.fit(documents)

# Transform the documents into vectors
document_vectors = vectorizer.transform(documents)
print(document_vectors)
# Print the vectors for each document
#for i, document in enumerate(documents):
#print(f"Document {i+1} vector: {document_vectors[i]}")
"""
#number of documents
num_docs = len(processed_tokens)
print(num_docs)
# Create a dictionary to store the vectors for each word
vectors = {}

# Create a vector of zeros for each word
for word in term_frequency.keys():
    vectors[word] = np.zeros(num_docs)

# Fill in the vectors
for i, doc in enumerate(processed_tokens):
    for word in doc:
        if word in vectors:
            vectors[word][i] += 1
#print(vectors)

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer(vocabulary=indexing_terms)

# from Document import vectorizer
# document_matrix = vectorizer.transform([' '.join(tokens) for tokens in processed_tokens])

# Fit the vectorizer on the processed documents
document_matrix = vectorizer.fit_transform([' '.join(tokens) for tokens in processed_tokens])

"""
# Create a document-term matrix
document_matrix = np.array([vectors[word] for word in vectors.keys()]).T

# Calculate IDF values
num_docs_containing_term = np.sum(document_matrix > 0, axis=0)
idf = np.log((num_docs + 1) / (num_docs_containing_term + 1)) + 1
idf = np.nan_to_num(idf)

# Apply TF-IDF weighting
document_matrix = document_matrix * idf

# Normalize the matrix
document_matrix /= np.linalg.norm(document_matrix, axis=1)[:, np.newaxis]

# print(document_matrix)
"""