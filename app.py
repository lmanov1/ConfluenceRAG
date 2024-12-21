import os 
import sys
from dotenv import load_dotenv, find_dotenv
import logging
import pickle
from  confluence_reader import get_and_save_space
from config import (CONFLUENCE_SPACE_NAME, CONFLUENCE_SPACE_KEY)

import faiss
import gradio as gr
#from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from transformers import pipeline, AutoModelForCausalLM
#from transformers import  AutoTokenizer
import numpy as np
from injest import  load_embed_text_from_directory , run_embedding_pipeline
import torch

def print_cuda_memory(device):            
    stats = torch.cuda.memory_stats(device=device)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    if torch.cuda.is_available():
        # Return the global free and total GPU memory for a given device
        free, total = torch.cuda.mem_get_info()
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        print(f"Free CUDA memory: {free_gb:.2f} GB")
        print(f"Total CUDA memory: {total_gb:.2f} GB")


# Step 2: Generate embeddings and create FAISS index with preprocessed texts (chunks)
def build_faiss_index(embeddings):    
    # Ensure embeddings are on CPU and convert to numpy
    embeddings_cpu = [emb.cpu().numpy() for emb in embeddings]  # Convert each tensor to numpy (if on GPU)    
    # Convert the list of numpy arrays into a single numpy 2D array
    chunk_embeddings_np = np.vstack(embeddings_cpu)  # Stack them along the first axis (n_samples, n_features)    
    # Ensure the dimensions are correct
    dimension = chunk_embeddings_np.shape[1]  # Number of features (embedding dimension)    
    # Initialize FAISS index
    index = faiss.IndexFlatL2(dimension)    
    # Add embeddings to the FAISS index
    index.add(chunk_embeddings_np)    
    # Save the FAISS index to disk
    faiss.write_index(index, "faiss_index.bin")    
    return index


# SEarch for relevant data chunks and generate LLM answer with relevant context
def process_query(query, embedding_model, tokenizer, model, index):    
    """
    Process the input query by embedding it using the provided model and searching 
    for the most similar chunk in the FAISS index.
    Parameters:
    - query: The query string.
    - tokenizer: The tokenizer to use for encoding the query.
    - model: The model to use for generating embeddings.
    - index: The FAISS index containing precomputed chunk embeddings

    Returns:
    - The most relevant chunk(s) of text from the document.
    """
    query_embeddings = run_embedding_pipeline(query, embedding_model, tokenizer, chunk_size=128, overlap=20)
    # Ensure embeddings are on CPU and convert to numpy
    query_embeddings_cpu = [emb.cpu().numpy() for emb in query_embeddings]  # Convert each tensor to numpy (if on GPU)    
    # Convert the list of numpy arrays into a single numpy 2D array
    chunk_embeddings_np = np.vstack(query_embeddings_cpu)  # Stack them along the first axis (n_samples, n_features)    
    D, I = index.search(chunk_embeddings_np, k=3)  # Search for the top-k (3) most similar chunks
    # Print the distances and indices
    print("Distances (D):", D)
    print("Indices (I):", I)
    print(f"type of chunk_embeddings_np: {type(chunk_embeddings_np)})")
    print(f"chunk_embeddings_np: {chunk_embeddings_np}")
    # Print information about query embeddings
    
    # print(f"Length of query_embeddings: {len(query_embeddings)}")
    print(f"Type of query_embeddings: {type(query_embeddings)}")
    print(f"query_embeddings: {query_embeddings}")

    tensor_emb = query_embeddings[0]  # Get the first (and only) tensor in the list
    tensor_emb_cpu = tensor_emb.cpu()
    print(f"Length of tensor_emb_cpu: {len(tensor_emb_cpu)}")
    print(f"Type of tensor_emb_cpu: {type(tensor_emb_cpu)}")
    print(f"Shape of tensor_emb: {tensor_emb.shape}")
    print(f"Shape of tensor_emb_cpu: {tensor_emb_cpu.shape}")
    idx0 = I[0]
    for idx in idx0:
        print(f" : Value at index {idx}:")
        value_at_index_0 = tensor_emb_cpu[0,idx ].item()
        print(f"Value at index {idx}: {value_at_index_0}")
        
    # Print the actual chunks for each match
    decoded_chunks = []
    for i, indices in enumerate(I):
        print(f"\nQuery {i} matches:")
        for idx, chunk_idx in enumerate(indices):
            print(f"Match {idx} (distance: {D[i][idx]:.4f}):")
            print(f"Match {idx} (index: {I[i][idx]} {chunk_idx}):")
            print(f"query_embeddings at index {chunk_idx}: {tensor_emb_cpu[0,chunk_idx].item()}")           
            chunk_text = tensor_emb_cpu[0,chunk_idx].item()
            print(f"Chunk text: {chunk_text}")
            decoded_chunks.append(chunk_text)
            

    # Step 5: Get the most similar chunks as context
    #context = [query_embeddings[i.item()] for i in I[0]]  # Get the chunks corresponding to the top-k (3) results
    print(f"Most similar chunks: {decoded_chunks}")
    # Step 6: Concatenate the context chunks into one string (assuming the chunks are small enough)
    context_text = " ".join([tokenizer.decode(chunk) for chunk in decoded_chunks])
    # Step 7: Generate the answer using the query and the concatenated context
    input_text = f"Context: {context_text}\n\nQuestion: {query}\n\nAnswer:"    
    # Tokenize the input text (query + context) and ensure it fits within the model's token limit
    input_ids = tokenizer.encode(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')

    # Generate the answer from the model
    output = model.generate(input_ids, 
                            max_new_tokens=250, 
                            num_beams=5, 
                            no_repeat_ngram_size=2, 
                            early_stopping=True)

    # Decode the generated answer
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract the answer after the "Answer:" part
    generated_answer = answer.split("Answer:")[-1].strip()
    return generated_answer
    
# #############################################################################
if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    # Env variables
    sys.path.append('../')
    load_dotenv(find_dotenv())
        
    print_cuda_memory("cuda")
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    #tokenizer = AutoTokenizer.from_pretrained(qa_model_name , device_map=device )
    #Load the embedding model and LLM
    #embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
    #qa_model_name = "tiiuae/falcon-7b-instruct"  # too heavy to load in memory without offload to disk
    qa_model_name = "ibm-granite/granite-3.0-2b-instruct"  
    #"openlm-research/open_llama_3b" - to try next    
    
    
    # Initialize BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' ,  device_map=device)
    #This embedding model takes integer token IDs directly as input.
    embedding_model = BertModel.from_pretrained('bert-base-uncased',  device_map=device)  

    # Set the pad_token to be the same as the eos_token
    tokenizer.pad_token = tokenizer.eos_token
    # OR Add a custom pad token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    qa_model = AutoModelForCausalLM.from_pretrained(qa_model_name, 
    #low_cpu_mem_usage=True,
    device_map=device,  # Automatically distributes the model across devices
    #offload_folder="offload",  # Offload layers to disk to save memory
    #offload_state_dict=True,
    #torch_dtype=torch.float16,
    )
    # load_in_8bit=True
    
    # ==> Open From here ==============================
    # spaces  = {CONFLUENCE_SPACE_KEY: CONFLUENCE_SPACE_NAME}
    # print(f"spaces: {spaces}")
    
    # pickle_file_path = "confluence_chunks.pkl"
    # chunks = []
    
    # if os.path.exists(pickle_file_path):
    #     with open(pickle_file_path, "rb") as f:
    #         chunks = pickle.load(f)            
    #         print(f"=============Chunks of text after preprocessing: {len(chunks)}")
    #         for i, chunk in enumerate(chunks):
    #             if i % 20 == 0:
    #                 print(f"Chunk {i}: {chunk}")            
    # else:        
    #     for space_key,  space_name in spaces.items():            
    #         space_chunks = get_and_save_space(space_key, space_name , tokenizer)
    #         if space_chunks:
    #             chunks.extend(space_chunks)
            
    #         #Save all chunks to a pickle file
    #         if len(space_chunks) != 0 :
    #             with open(pickle_file_path, "wb") as f:
    #                 pickle.dump(chunks, f)

    #         print(f" {len(space_chunks)} text chunks successfully saved to {pickle_file_path}.")
    # ==> To here ==============================
    embeddings = load_embed_text_from_directory("./OnyxData/FE_Group" , tokenizer , embedding_model)
    print(f"=============Embeddings len after preprocessing: {len(embeddings)} , embeddings type: {type(embeddings)}")
    # for i, embedding in enumerate(embeddings):
    #     if i % 20 == 0:
    #         print(f"Chunk {i}: {embedding}")   

    
    index = build_faiss_index(embeddings)
    # Step 5: Integrate with Gradio chatbot
    def chatbot(query):      
        answer = process_query(query, embedding_model , tokenizer, qa_model, index)
        return answer

    # Gradio UI
    interface = gr.Interface(
        fn=chatbot,
        inputs="text",
        outputs="text",
        title="Confluence assistant",
        description="Ask questions about the content of the locally available Confluence spaces"
    )

    # Run the chatbot
    interface.launch()






