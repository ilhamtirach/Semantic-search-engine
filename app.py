import streamlit as st
import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings.base import Embeddings
import chromadb
import torch
import pandas as pd

# Fonction pour l'initialisation des embeddings pour la recherche de documents légaux
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0]


# Modèles pour la recherche de documents légaux
bi_encoder_legal = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder_legal = SentenceTransformer('all-MiniLM-L12-v2')

# Initialisation FAISS pour la recherche de documents légaux
embedding_dim = bi_encoder_legal.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(embedding_dim)

# Initialisation du SemanticChunker pour la recherche de documents légaux
embeddings_legal = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
semantic_chunker = SemanticChunker(embeddings=embeddings_legal)

# Modèles pour la recherche d'emails
bi_encoder_email = SentenceTransformer('avsolatorio/GIST-Embedding-v0')
cross_encoder_email = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

# Initialisation ChromaDB pour la recherche d'emails
client = chromadb.PersistentClient(path="data_embeddings")
collection_BERT = client.get_collection(name="ENRON_GIST")


# Fonction pour charger et découper les fichiers texte (documents légaux)
def load_and_chunk_txt_files(txt_files):
    texts = []
    doc_names = []
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as file:
            text = file.read()
            texts.append(text)
            # Extraire le nom du fichier sans l'extension
            doc_name = os.path.splitext(os.path.basename(txt_file))[0]
            doc_names.append(doc_name)

    # Appliquer le chunking sémantique
    all_chunks = []
    chunk_indices = []
    for i, text in enumerate(texts):
        chunks = semantic_chunker.split_text(text)
        all_chunks.extend(chunks)
        chunk_indices.extend([i] * len(chunks))

    return all_chunks, chunk_indices, doc_names

# Fonction pour sauvegarder les embeddings
def save_embeddings(embeddings, filename="embeddings.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)

# Fonction pour charger les embeddings
def load_embeddings(filename="embeddings.pkl"):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Interface utilisateur principale
st.title("Système de Recherche Sémantique")

# Choix du type de recherche
search_type = st.radio(
    "Choisissez le type de recherche que vous souhaitez effectuer :",
    ("Recherche sur les documents légaux", "Recherche sur les emails")
)

# Si l'utilisateur choisit la recherche sur les documents légaux
if search_type == "Recherche sur les documents légaux":
    txt_files = ["pdf1.txt",
                 "pdf2.txt"]

    with st.spinner("Traitement des fichiers..."):
        chunks, chunk_indices, doc_names = load_and_chunk_txt_files(txt_files)
        chunk_embeddings = bi_encoder_legal.encode(chunks, convert_to_tensor=True)

        # Ajout des embeddings à l'index FAISS
        faiss_index.add(chunk_embeddings.cpu().numpy())

        # Option pour sauvegarder les embeddings
        save_embeddings(chunk_embeddings.cpu().numpy(), filename="chunk_embeddings.pkl")

    # Saisie de la requête
    query = st.text_input("Entrez votre requête ici :")
    if query:
        with st.spinner("Recherche en cours..."):
            query_embedding = bi_encoder_legal.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

            # Recherche sémantique avec FAISS (Top 5 résultats)
            distances, indices = faiss_index.search(query_embedding, k=5)
            retrieved_chunks = [chunks[idx] for idx in indices[0]]
            retrieved_docs = [doc_names[chunk_indices[idx]] for idx in indices[0]]

            # Re-rank avec cross-encoder
            cross_input = [query] * len(retrieved_chunks)
            chunk_embeddings = cross_encoder_legal.encode(retrieved_chunks, convert_to_tensor=True)
            similarity_scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings).numpy().flatten()
            sorted_indices = np.argsort(similarity_scores)[::-1]

            # Affichage des résultats reclassés
            st.subheader("Résultats:")
            for idx in sorted_indices:
                st.write(f"**Document:** {retrieved_docs[idx]}")
                st.write(f"**Chunk:** {retrieved_chunks[idx]}")
                st.write(f"**Score de similarité:** {similarity_scores[idx]:.4f}")
                st.write("---")

# Si l'utilisateur choisit la recherche sur les emails
elif search_type == "Recherche sur les emails":
    top_k = 50

    # Saisie de la requête
    query = st.text_input("Entrez votre requête ici :")
    if query:
        with st.spinner("Recherche en cours..."):
            query_embedding = bi_encoder_email.encode(query).tolist()

            # Recherche dans ChromaDB
            results = collection_BERT.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            documents = results['documents'][0]
            pairs = [(query, doc) for doc in documents]
            scores = cross_encoder_email.predict(pairs)

            # Re-rank avec Cross-Encoder
            scores = [f"{nombre:.2f}" for nombre in scores]
                #sorting scores
   
        # Trier et récupérer les top N résultats
            sorted_scores_and_documents = sorted(
                zip(scores, documents, results['metadatas'][0], results['distances'][0]), 
                key=lambda x: x[0],  # Trier par le score (premier élément dans chaque tuple)
                reverse=True  # Trier dans l'ordre décroissant
            )

            top_n = 5
            sorted_scores_and_documents = sorted_scores_and_documents[:top_n]

            # Générer des IDs qui commencent à 1
            sorted_ids = list(range(1, top_n + 1))

            # Extraire les éléments triés
            sorted_scores = [item[0] for item in sorted_scores_and_documents]  # Extraire les scores triés
            sorted_documents = [item[1] for item in sorted_scores_and_documents]  # Extraire les documents triés
            sorted_metadata = [item[2] for item in sorted_scores_and_documents]  # Extraire les metadata triés
            sorted_distances = [item[3] for item in sorted_scores_and_documents]  # Extraire les distances triées

            # Créer un DataFrame avec l'ID des documents commençant à 1
        


            def truncate_text(text, max_lines=8):
                    lines = text.split('\n')
                    if len(lines) > max_lines:
                        return '\n'.join(lines[:max_lines]) + '...'
                    return text

                # Apply truncation to documents for display
            
            truncated_documents = [truncate_text(doc) for doc in sorted_documents]
        




        

        # Convert sorted results to a pandas DataFrame
            df = pd.DataFrame({
            'ID': sorted_ids,          # Ajout de l'ID des documents
            'Score': sorted_scores,
            'Document': truncated_documents,
            'Metadata': sorted_metadata,
            #'Distance_bi-encoder': sorted_distances
        })


            df_download = pd.DataFrame({
                'ID': sorted_ids,          # Ajout de l'ID des documents
                'Score': sorted_scores,
                'Document': sorted_documents,
                'Metadata': sorted_metadata,
                'Distance_bi-encoder': sorted_distances

            })

                # Display the DataFrame as a table
            st.subheader("Re-ranked Query Results:")
            st.table(df)




        # Add a download button to download the output data as a CSV file
            csv = df_download.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="query_results.csv",
                mime="text/csv"
            )