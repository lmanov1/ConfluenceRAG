import os
import sys
from dotenv import load_dotenv, find_dotenv
import logging
import pickle
from  confluence_reader import get_and_save_space
from config import (CONFLUENCE_SPACE_NAME, CONFLUENCE_SPACE_KEY)
from config import embeds_file_path
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer , models
from transformers import  AutoModelForCausalLM
from transformers import  AutoTokenizer
import numpy as np
import pandas as pd
from injest import  load_embed_text_from_directory , run_embedding_pipeline_on_file , print_cuda_memory
from tqdm import tqdm
import time
# Retrieve the context window size (max tokens)
def get_context_window_size(tokenizer, model):
    return model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else 2048

# Function to calculate token length using
def estimate_tokens(text , tokenizer):
    return len(tokenizer.encode(text))

# Function to find the most similar chunks and return a combined context
def find_similar_chunks(query, embeddings_df, max_tokens , model):
    """
    Finds and retrieves semantically similar text chunks for a given query using embedding similarity.
    This function embeds the query text, computes similarity scores with existing document chunks,
    and returns the most relevant chunks while respecting a maximum token limit.
    Args:
        query (str): The input query text to find similar chunks for
        embeddings_df (pd.DataFrame): DataFrame containing text chunks and their embeddings
        max_tokens (int): Maximum number of tokens allowed in the combined output
        model: The language model to use for embeddings
    Returns:
        str: Combined context string containing the query and most similar text chunks
    The function:
    1. Embeds the query text
    2. Computes cosine similarity between query and existing chunk embeddings
    3. Sorts chunks by similarity score
    4. Selects top chunks while staying within token limit
    5. Returns formatted context string with query and selected chunks
    """
    query_embeddings_df = run_embedding_pipeline_on_file(query, embedding_model, tokenizer,
                                              chunk_size=500, overlap=50)
    if query_embeddings_df is None:
        print(f"No embeddings generated for the query text: {query}")
        return ""
    else:
       # Convert 'embedding' column from Series to NumPy array in-place (if needed)
        query_embeddings_df['embedding'] = query_embeddings_df['embedding'].apply(lambda x: x if isinstance(x, np.ndarray) else np.array(x))
        embeddings_df['embedding'] = embeddings_df['embedding'].apply(lambda x: x if isinstance(x, np.ndarray) else np.array(x))      

        # Compute similarity scores using numpy
        similarities = []
        for _, query_emb in query_embeddings_df.iterrows():
            for _, chunk_emb in embeddings_df.iterrows():
                # Compute cosine similarity
                #print(f"\nComparing texts:\nQuery text: {query_emb['text_chunk']}\nChunk text: {chunk_emb['text_chunk']}")
                similarities.append(
                    np.dot(query_emb["embedding"], chunk_emb["embedding"]) /
                    (np.linalg.norm(query_emb["embedding"]) * np.linalg.norm(chunk_emb["embedding"]))
                )        

    # Add similarity scores to DataFrame
    embeddings_df["similarity"] = similarities

    # Sort by similarity in descending order
    sorted_df = embeddings_df.sort_values(by="similarity", ascending=False)

    # Select the top chunks while staying within the token limit
    top_chunks = []
    token_count = estimate_tokens(query,tokenizer)  # Start with query token length
    print(f"Initial token count (query tokens): {token_count}")
    for _, row in sorted_df.iterrows():
        chunk_tokens = estimate_tokens(row["text_chunk"],tokenizer)
        if token_count + chunk_tokens <= max_tokens:
            top_chunks.append(row["text_chunk"])
            token_count += chunk_tokens
        if len(top_chunks) >= 3:  # Limit to top 3 chunks
            break

    # Combine query and selected chunks
    combined_context = f"Query: {query}\n\n" + "Most similar data:\n" "\n".join(
        [f" {chunk}" for chunk in top_chunks]
    )

    # Display the result
    print("=============================\nGenerated Context:")
    print(f"\"{combined_context}\"\n=============================")
    return combined_context


# SEarch for relevant data chunks and generate LLM answer with relevant context
def process_query(query, embedding_model, tokenizer, model, embeddings_df ):
    """
    Process the input query by embedding it using the provided model and searching
    for the most similar chunk in the embeddings_df.
    Parameters:
    - query: The query string.
    - tokenizer: The tokenizer to use for encoding the query.
    - model: The model to use for generating embeddings.
    - embeddings_df - dataframe with embeddings and text chunks

    Returns:
    - combined_context: query plus the most relevant chunk(s) of text from the document.
    """

    # Define the max tokens budget
    #max_tokens_budget = 100  # Example token limit TBD model context window size
    max_tokens_budget = get_context_window_size(tokenizer, model)
    print(f"Model's max token budget: {max_tokens_budget}")

    # Find the top 3 most similar chunks and construct context
    context = find_similar_chunks(query, embeddings_df, max_tokens_budget , model)

    # Pass the context to the generative model to generate an answer            
    #print(f"Tokenizing {context}...")
    inputs =  tokenizer(context, return_tensors="pt", truncation=True, padding=True)
    #print(f"Getting attention mask for {context}...")
    attention_mask = inputs["attention_mask"]
    #print(f"Attention mask is {attention_mask}...")
   
    input_ids = inputs.input_ids
    #print(f"input_ids mask is {input_ids}...")
    # Generate an answer using the model
    # If both `max_new_tokens` (=70) and `max_length`(=131072) is set, `max_new_tokens` will take precedence    
    start_time = time.time()
    print("Starting answer generation...")
    output_ids = model.generate(input_ids.to(model.device), 
        num_return_sequences=1, 
        attention_mask=attention_mask, 
        max_new_tokens=150,        
        num_beams=3,          
        early_stopping=True)   
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Generation completed in {elapsed_time:.2f} seconds")

    #print("Decoding answer output_ids..{output_ids}")
    # Decode the generated answer
    generated_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated answer: {generated_answer}")
    # Extract the answer after the "Answer:" part
    generated_answer = generated_answer.split("Answer:")[-1].strip()
    return generated_answer

# #############################################################################
if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    # Env variables
    sys.path.append('../')
    load_dotenv(find_dotenv())

    print_cuda_memory("cuda")
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #device = "auto"
    device = "cpu"

    #qa_model_name = "tiiuae/falcon-7b-instruct"  # too heavy to load in memory without offload to disk
    #qa_model_name = "ibm-granite/granite-3.0-2b-instruct"
    #"openlm-research/open_llama_3b" - to try next

    #Load the embedding model and LLM
    #embedding_model_name= 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    embedding_model_name= 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_model = SentenceTransformer(embedding_model_name)
    print(f"Loaded embedding model: {embedding_model}, named {embedding_model_name}")

    #qa_model_name = "meta-llama/Llama-3.2-3B"  # LLM model
    qa_model_name = "ibm-granite/granite-3.1-2b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(qa_model_name , device_map=device)
    # Set the pad_token to be the same as the eos_token
    tokenizer.pad_token = tokenizer.eos_token

    qa_model = AutoModelForCausalLM.from_pretrained(qa_model_name,
    #low_cpu_mem_usage=True,
    device_map=device,  # Automatically distributes the model across devices
    #offload_folder="offload",  # Offload layers to disk to save memory
    #offload_state_dict=True,
    #torch_dtype=torch.float16,
    )
    print(f"Loaded LLM model: {qa_model_name}")
    #system_prompt = "You are a helpful assistant. Answer the questions clearly and concisely. Be short but comprehensive. If answer is not found - say so."
    qa_model.eval()
   
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
    if os.path.exists(embeds_file_path):
        embeddings_df = pd.read_csv(embeds_file_path)
        embeddings_df['embedding'] = embeddings_df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
        # print("\nFirst row of embeddings_df after conversion:")
        # print(embeddings_df.iloc[0])
        # print(embeddings_df.info())
        # print(embeddings_df.describe())
    else:
        # Load text chunks from directory
        print("Going to embed text from ./wireshark_dissectors_tutorial...")
        embeddings_df = load_embed_text_from_directory("./wireshark_dissectors_tutorial", tokenizer, embedding_model)

        # print("Going to embed text from ./OnyxData/FE_Grouprelevant directory...")
        # embeddings_df = load_embed_text_from_directory("./OnyxData/FE_Group", tokenizer, embedding_model)

        
        # Save to CSV file
        # Remove output file if it exists
        output_path = embeds_file_path
        if len(embeddings_df) > 0:       
            if os.path.exists(output_path):
                os.remove(output_path)
            print(f"Saving {len(embeddings_df)} embeddings to {output_path}...")
            # Save with progress bar
            tqdm.pandas()
            embeddings_df.to_csv(output_path, index=False, chunksize=1000)
            print(f"Embeddings saved to {embeddings_df}")

    print(f"=============Embeddings len after preprocessing: {embeddings_df.shape} , embeddings type: {(embeddings_df.info())}")
    # for i, embedding in enumerate(embeddings):
    #     if i % 20 == 0:
    #         print(f"Chunk {i}: {embedding}")

    # Step 5: Integrate with Gradio chatbot
    def chatbot(query):
        answer = process_query(query, embedding_model , tokenizer, qa_model, embeddings_df)
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






