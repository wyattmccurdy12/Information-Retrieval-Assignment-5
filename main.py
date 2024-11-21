'''
Wyatt McCurdy
Information Retrieval Homework 5

The goal of this system is to make the best possible predictions (measured by P@10) in a retrieval system 
answering questions about travel. To do this, we simply use the best possible sentence transformer model that we can find, 
out-of-the-box. 

Data in the data directory is in trec format. 
documents: Answers.json
queries: topics_1.json, topics_2.json
qrels:   qrel_1.tsv (qrel_2.tsv is reserved by the instructor)

models used: 
sentence-transformers/multi-qa-mpnet-base-dot-v1
'''

import re
from sentence_transformers import SentenceTransformer
import json
import argparse
import numpy as np
from tqdm import tqdm
import os

class Retriever:
    """
    A class to represent a retriever model which can be either a bi-encoder or a cross-encoder.

    Attributes:
    model_type (str): The type of the model ('bi-encoder' or 'cross-encoder').
    model (SentenceTransformer): The loaded model.

    Methods:
    __init__(model_name):
        Initializes the Retriever with the specified model name.
    encode(texts):
        Encodes a list of texts using a bi-encoder model.
    """
    def __init__(self, model_name):
        """
        Initializes the Retriever with the specified model type and name.

        Parameters:
        model_name (str): The name of the model to load.
        """
        self.model = SentenceTransformer(model_name)


    def encode(self, texts):
        """
        Encodes a list of texts using the bi-encoder model.

        Parameters:
        texts (list of str): The texts to encode.

        Returns:
        list: The encoded texts.
        """

        return self.model.encode(texts)

def load_data(queries_path, documents_path):
    """
    Loads the documents and queries from the specified files.

    Parameters:
    queries_path (str): The path to the queries file.
    documents_path (str): The path to the documents file.

    Returns:
    tuple: A tuple containing the documents and queries.
    """
    with open(documents_path, 'r') as f:
        documents = json.load(f)
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    return documents, queries

def write_results_to_tsv(results, output_filename, output_dir, model_type, model_status):
    """
    Writes the results to a TSV file in TREC format.

    Parameters:
    results (dict): The results to write.
    output_filename (str): The name of the output file.
    output_dir (str): The directory to write the output file.
    model_type (str): The type of the model ('simple' or 'reranked').
    model_status (str): The status of the model ('pretrained' or 'finetuned').
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the full output path
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        for query_id, doc_scores in results.items():
            for rank, (doc_id, score) in enumerate(doc_scores[:100], start=1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score} {model_type}_{model_status}\n")

def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)

def main():
    """
    Main function to run the ranking and re-ranking system.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Ranking and Re-Ranking System')
    parser.add_argument('-q', '--queries', default='data/inputs/topics_1.json', help='Path to the queries file')
    parser.add_argument('-d', '--documents', default='data/inputs/Answers.json', help='Path to the documents file')
    parser.add_argument('-be', '--bi_encoder', default='sentence-transformers/multi-qa-mpnet-base-dot-v1', help='Bi-encoder model string')
    parser.add_argument('-op', '--output_dir', default='data/outputs/', help='Path to the output directory')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    documents, queries = load_data(args.queries, args.documents)
    print("Data loaded successfully.")

    # Instantiate retrievers
    bi_encoder_retriever = Retriever(args.bi_encoder)

    # Process and encode queries using bi-encoder
    print("Processing and encoding queries...")
    processed_queries = []
    for query in tqdm(queries, desc="Processing queries"):
        query_id = query['Id']
        title = remove_html_tags(query['Title'])
        body = remove_html_tags(query['Body'])
        tags = ' '.join(query['Tags'])
        merged_query = f"{title} {body} {tags}"
        processed_queries.append((query_id, merged_query))
    
    encoded_queries = bi_encoder_retriever.encode([q[1] for q in processed_queries])
    print("Queries processed and encoded successfully.")
    
    # Process and encode documents using bi-encoder
    print("Processing and encoding documents...")
    processed_documents = {}
    for doc in tqdm(documents, desc="Processing documents"):
        doc_id = doc['Id']
        text = remove_html_tags(doc['Text'])
        processed_documents[doc_id] = text
    
    encoded_documents = bi_encoder_retriever.encode(list(processed_documents.values()))
    print("Documents processed and encoded successfully.")
    
    # Initial ranking using bi-encoder
    print("Performing initial ranking using bi-encoder...")
    initial_rankings = {}  # Dictionary to store initial rankings
    for query_id, query_text in tqdm(processed_queries, desc="Ranking queries"):
        query_embedding = bi_encoder_retriever.encode([query_text])[0]
        # Compute similarity scores and rank documents
        scores = np.dot(encoded_documents, query_embedding)
        ranked_doc_indices = np.argsort(scores)[::-1][:100]
        initial_rankings[query_id] = [(list(processed_documents.keys())[doc_id], scores[doc_id]) for doc_id in ranked_doc_indices]
    print("Initial ranking completed.")
    
    # Extract topic number from queries file name
    topic_number = args.queries.split('_')[-1].split('.')[0]

    # Determine output filename
    output_filename = f"result_bi_{topic_number}.tsv"
    # Write initial rankings to TSV
    write_results_to_tsv(initial_rankings, output_filename, args.output_dir, model_type='simple', model_status='pretrained')
    print(f"Initial rankings have been computed and saved to {os.path.join(args.output_dir, output_filename)}.")

if __name__ == "__main__":
    main()
