from sklearn.metrics.pairwise import cosine_similarity

from Document import document_matrix, documents
from Queries import query_matrix

print(query_matrix.shape)
print(document_matrix.shape)
cosine_similarities = cosine_similarity(query_matrix, document_matrix.T)
"""
# Compute the cosine similarity between the query matrix and the document matrix
cosine_similarities = cosine_similarity(query_matrix, document_matrix)

# Get the index of the most similar document
most_similar_doc_idx = cosine_similarities.argmax()

# Get the title and content of the most similar document
most_similar_doc_title, most_similar_doc_content = documents[most_similar_doc_idx]

# Print the most similar document's title and content
print("Title:", most_similar_doc_title)
print("Content:", most_similar_doc_content)
"""