import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from Document import document_matrix
from Queries import query_matrix,query_texts

#print(query_matrix.shape)
#print(document_matrix.shape)
# Set the similarity threshold
similarity_threshold = 0.45

rel_file_path = 'test.txt'
# Compute the cosine similarity between the query matrix and the document matrix
cosine_similarities = cosine_similarity(query_matrix, document_matrix)
#print(cosine_similarities)

# Initialize an empty list to store the relevant documents for each query
relevant_docs_for_query = [[] for _ in range(len(query_texts))]

# Find the relevant documents for each query
for i, cosine_similarity in enumerate(cosine_similarities):
    for j, similarity_score in enumerate(cosine_similarity):
        if similarity_score >= similarity_threshold:
            relevant_docs_for_query[i].append(j)

with open(rel_file_path, 'w') as f:
    for i, relevant_docs in enumerate(relevant_docs_for_query):
        for doc_index in relevant_docs:
            f.write(f"{i + 1} {doc_index + 1} {cosine_similarities[i][doc_index]}\n")

"""
print(cosine_similarities.shape) # (30, 1460)
for i, cosine_similarity in enumerate(cosine_similarities):
    # Get the index of the most similar document
    most_similar_doc_idx = cosine_similarity.argmax()

    # Get the title and content of the most similar document
    most_similar_doc_content = documents[most_similar_doc_idx]

    # Print the most similar document's title and content
    # print("Title:", most_similar_doc_title)
    #print("Query:", query_texts[i])
    #print("Content:", most_similar_doc_content)

# For ranking, maybe look to do something like this:

# Create a DataFrame to store the document rankings
#rankings = pd.DataFrame(cosine_similarity, columns=documents)

# Sort the rankings in descending order for each query
#rankings = rankings.apply(lambda x: x.sort_values(ascending=False).index, axis=1)
#print(rankings)
"""