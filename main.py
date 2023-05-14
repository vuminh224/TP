import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from Document import document_matrix, documents
from Queries import query_matrix, query_texts

#print(query_matrix.shape)
#print(document_matrix.shape)

rel_file_path = 'CISI_dev.REL'
# Compute the cosine similarity between the query matrix and the document matrix
cosine_similarities = cosine_similarity(query_matrix, document_matrix)
#print(cosine_similarities)

with open(rel_file_path, 'w') as f:
    for i, cosine_similarity in enumerate(cosine_similarities):
        # Get the top 5 most similar documents
        most_similar_doc_indices = cosine_similarity.argsort()[::-1][:5]

        for j, doc_idx in enumerate(most_similar_doc_indices):
            # Write the query number, document number, and similarity score to file
            f.write(f"{i+1} {doc_idx+1} {cosine_similarity[doc_idx]:.4f}\n")

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