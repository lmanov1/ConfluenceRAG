import re
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import nltk
from sklearn.preprocessing import normalize
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from typing import List
from pydantic import BaseModel, ValidationError
from sklearn.preprocessing import normalize
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt')
from config import embeds_file_path
def load_embed_text_from_directory(directory_path , tokenizer , embedding_model, chunk_size=500, overlap=50):

    # Initialize an empty DataFrame
    dir_embeddings = pd.DataFrame(columns=['text_chunk', 'embedding'])
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Adjust for other formats
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                print(f"Processing file: {filename}")
                embeddings = run_embedding_pipeline(file.read(), embedding_model, tokenizer, chunk_size, overlap)
                if embeddings is not None:
                    dir_embeddings = pd.concat([dir_embeddings, embeddings], ignore_index=True)

    print(f"==> load_text_from_directory: Total embeddings: {len(dir_embeddings)} type: {type(dir_embeddings)}")
    # , len of first text: {len(dir_embeddings[0])} , first text {dir_embeddings[0]}")
    return dir_embeddings


def run_embedding_pipeline(text, embedding_model, tokenizer, chunk_size=500, overlap=50):
    # Define stop words and lemmatizer
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    # Preprocess and split the text into tokenized chunks
    chunks = preprocess_and_split_text(
        text=text,
        tokenizer=tokenizer,
        stop_words=stop_words,
        lemmatizer=lemmatizer,
        chunk_size=chunk_size,
        overlap=overlap
    )

    df = None
    if chunks is not None and len(chunks) > 0:
        try:
            # Enable debug information for the encode process
            embeddings = embedding_model.encode(chunks, convert_to_tensor=False, convert_to_numpy= True, show_progress_bar=True, output_value='token_embeddings')                
        except KeyError as e:
            print(f"KeyError: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        print(f"chunks len {len(chunks)}    type {type(chunks)}; \nembeddings: len {len(embeddings)}    type {type(embeddings)}")
        #Check if embeddings are still tensors
        if isinstance(embeddings, list) :
            if isinstance(embeddings[0], torch.Tensor):
                embeddings = [e.cpu().numpy() for e in embeddings]
        else:
            if isinstance(embeddings, torch.Tensor):                    
                embeddings = embeddings.cpu().numpy()
            else:
                embeddings = np.array(embeddings)

        # Ensure embeddings are 2D
        embeddings = np.vstack(embeddings)
        normalized_embeddings = [normalize(e.reshape(1, -1)).flatten() for e in embeddings]           
        # Create a list of dictionaries with text chunks and embeddings
        records = [
            {
                "text_chunk": chunk,
                "embedding": embedding
            }
            for chunk, embedding in zip(chunks, normalized_embeddings)
        ]

        print(f"========================> run_embedding_pipeline: Total records: {len(records)} type: {type(records)}")
        # Print records in human-readable format
        print("\nHuman-readable records:")
        for i, record in enumerate(records):
            print(f"\nRecord {i}:")
            print(f"Text chunk: {record['text_chunk'][:100]}...\n")  # First 100 chars
            print(f"Embedding : {type(record['embedding'])}\n - value: {record['embedding']}")
        
        print("=========================> ")
        # Create DataFrame from records
        df = pd.DataFrame(records)
        print("Final DataFrame:")        
        print(df.describe())    
        print(df.info())
        print(df.shape)
        print(f"type of embeddings : {type(df['embedding'])}")
        print("=========================> End of run_embedding_pipeline")
    # else:
    #     print("No chunks to process")
    return df


def gpt_split_into_sections(content, chunk_size=500, overlap_len=50):
    """
    Splits the document into sections. If no recognizable headers or tags are present,
    the content is split into chunks of a given size with overlap.
    Support for markdown format (headers starting with '#') ,  HTML headers () and plain text

    Parameters:
        content (str): The input document as a single string.
        chunk_size (int): The maximum size of each chunk (in characters).
        overlap_len (int): The overlap length between consecutive chunks (in characters).

    Returns:
        list: A list of dictionaries, where each dictionary represents a section with 'subtitle' and 'text'.
    """
    sections = []

    # Detect if the content contains HTML headers
    html_header_pattern = re.compile(r"<h[1-6]>(.*?)<\/h[1-6]>", re.IGNORECASE)
    html_headers = html_header_pattern.findall(content)

    if html_headers:
        # Handle content with HTML headers
        current_title = None
        current_text = []
        for line in re.split(r"(<h[1-6]>.*?<\/h[1-6]>)", content):
            if html_header_pattern.match(line):
                # If there's an existing section, save it
                if current_title or current_text:
                    sections.append({
                        "subtitle": current_title or "Untitled",
                        "text": "\n".join(current_text).strip()
                    })
                # Extract the title from the HTML header
                current_title = html_header_pattern.search(line).group(1).strip()
                current_text = []
            else:
                current_text.append(line)

        # Add the last section
        if current_title or current_text:
            sections.append({
                "subtitle": current_title or "Untitled",
                "text": "\n".join(current_text).strip()
            })
    else:
        # Check if recognizable Markdown-style headers exist
        lines = content.splitlines()
        has_markdown_headers = any(line.startswith("#") for line in lines)

        if has_markdown_headers:
            # Handle content with Markdown-style headers
            current_title = None
            current_text = []
            for line in lines:
                if line.startswith("#"):
                    # If there's an existing section, save it
                    if current_title or current_text:
                        sections.append({
                            "subtitle": current_title or "Untitled",
                            "text": "\n".join(current_text).strip()
                        })
                    # Start a new section
                    current_title = line.lstrip("#").strip()  # Extract the title
                    current_text = []
                else:
                    current_text.append(line)

            # Add the last section
            if current_title or current_text:
                sections.append({
                    "subtitle": current_title or "Untitled",
                    "text": "\n".join(current_text).strip()
                })
        else:
            # Handle plain text without headers by splitting into chunks
            start = 0
            while start < len(content):
                end = start + chunk_size
                chunk = content[start:end]

                # Add overlap to the next chunk, if it exists
                if end < len(content):
                    chunk = content[start:end + overlap_len]

                sections.append({
                    "subtitle": "Plain Text Chunk",
                    "text": chunk.strip()
                })
                start += chunk_size  # Move to the next chunk

    return sections

def gpt_tokenize_text(text, tokenizer):
    """
    Tokenizes text using a pre-initialized Hugging Face tokenizer.
    Parameters:
        text (str): The input text to be tokenized.
        tokenizer: A pre-initialized Hugging Face tokenizer object.

    Returns:
        tuple: A tuple containing the tokenized list and the token count.
    """
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids, len(token_ids)

def gpt_split_sections_into_chunks(sections, max_tokens , tokenizer):
    """Splits sections into chunks based on a token budget.
	Parameters:
        sections (list): A list of dictionaries, each containing 'subtitle' and 'text'.
        max_tokens (int): Maximum token budget for each chunk, including the title.
        tokenizer: A pre-initialized Hugging Face tokenizer object.
	Returns:
        list: A list of strings where each string is a chunk prefixed with its section title.

	Subtitle is used to calculate its token length and include it as a prefix for each chunk of the corresponding section's text.
		text is split into smaller chunks that fit within the remaining token budget after accounting for the subtitle.
"""
    chunks = []

    for section in sections:
        title = section["subtitle"]
        content = section["text"]

        # Tokenize title and calculate token length
        title_tokens = tokenizer.encode(title, add_special_tokens=False)
        title_length = len(title_tokens)

        # Determine the available budget for the content
        available_budget = max_tokens - title_length

        # Tokenize the content
        content_tokens = tokenizer.encode(content, add_special_tokens=False)

        # Split content tokens into chunks
        for i in range(0, len(content_tokens), available_budget):
            chunk_tokens = content_tokens[i:i + available_budget]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(f"{title}\n{chunk_text}")

    return chunks



def print_sections(sections):
    """
    Prints the sections in a user-friendly format.
    Each section includes its title (subtitle) and the text.
    """
    print("=== Sections ===")
    for i, section in enumerate(sections, start=1):
        print(f"Section {i}: {section['subtitle']}")
        print(f"Content:\n{section['text']}\n")


def print_chunks(chunks):
    """
    Prints the chunks in a user-friendly format.
    Each chunk contains the section title and the chunked text.
    """
    print("=== Chunks ===")
    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i}:\n{chunk}\n")


def preprocess_text(text: str, stop_words : set, lemmatizer ) -> str:
    """
    Preprocesses the input text by:
      - Converting to lowercase
      - Removing URLs, HTML tags, and special characters
      - Removing stop words
      - Lemmatizing words
    """
    # Remove URLs
    #text = re.sub(r'http\S+', '', text)
    # Remove HTML tags
    # text = re.sub(r'<.*?>', '', text) - used to split into sections
    # Remove non-alphanumeric characters (optional)
    #text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Remove extra spaces
    text = ' '.join(text.split())
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


def preprocess_and_split_text(
    text: str,
    tokenizer,
    stop_words: set,
    lemmatizer ,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Preprocesses and tokenizes text, then splits it into chunks of tokens.
    Args:
        text (str): The input text to preprocess.
        tokenizer: A tokenizer that provides `encode` and special token IDs.
        chunk_size (int): Maximum number of tokens per chunk.
        overlap (int): Number of tokens to overlap between chunks.
        stop_words (set): A set of stop words to filter.
        lemmatizer: A lemmatizer to normalize words.

    Returns:
        List[str]: A list of text chunks.
    """
    # Preprocess the text
    print(f"Input text: {text[:200]}...")  # Show first 200 chars
    # print(f"Chunk size: {chunk_size}, Overlap: {overlap}")
    # print(f"Number of stop words: {len(stop_words)}")
    cleaned_text = preprocess_text(text, stop_words, lemmatizer)
    #print(f"Cleaned text: {cleaned_text[:200]}...")  # Show first 200 chars
    # Split content into sections
    sections = gpt_split_into_sections(cleaned_text)
    #print_sections(sections)
    chunks = gpt_split_sections_into_chunks(sections, chunk_size , tokenizer)
    print_chunks(chunks)

    return chunks


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
# =======================================================

# max_length = 512
# def preprocess_and_sentence_split(
#     text: str,
#     tokenizer,
#     stop_words: set ,
#     lemmatizer
# ) -> List[List[int]]:
#     """
#     Preprocesses text, splits it into sentences, and tokenizes each sentence.
#     Args:
#         text (str): The input text to preprocess.
#         tokenizer: A tokenizer that provides `encode`.
#         stop_words (set): A set of stop words to filter.
#         lemmatizer: A lemmatizer to normalize words.

#     Returns:
#         List[List[int]]: A list of tokenized sentences.
#     """
#     # Preprocess the text
#     cleaned_text = preprocess_text(text, stop_words, lemmatizer)
#     # Split into sentences
#     sentences = sent_tokenize(cleaned_text)
#     # Tokenize each sentence
#     tokenized_sentences = [
#         tokenizer.encode(sentence, return_tensors="pt", padding=False, truncation=False, max_length = max_length)[0].tolist()
#         for sentence in sentences
#     ]

#     return tokenized_sentences

# # sliding_window_chunks
# def preprocess_split_text(text, tokenizer, max_length=chunk_size, overlap=20):

#     cleaned_text= preprocess_text(text)
#     print(f"cleaned_text : ", cleaned_text)

#     tokenized = tokenizer.encode(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     # Remove special tokens like [CLS] and [SEP]
#     tokens_cleaned = [token for token in tokenized if token != tokenizer.cls_token_id and token != tokenizer.sep_token_id]
#     print(f"preprocess_split_text: Cleaned Tokens: {len(tokens_cleaned)}  type: {type(tokens_cleaned)}")
#     print(f"preprocess_split_text: len of first token cleaned: {len(tokens_cleaned[0])} , first token cleaned {tokens_cleaned[0]}")
#     # Split the text into chunks
#     chunks = []
#     #chunks.extend([text[i:i+chunk_size] for i in range(0, len(text), max_length)])
#     # for i in range(0, len(tokenized), max_length - overlap):
#     #     chunk = tokenized[i:i + max_length]
#     #     chunks.append(chunk)
#     # Chunking the tokens
#     max_tokens=20
#     chunks = [tokens_cleaned[i:i + max_tokens] for i in range(0, len(tokens_cleaned), max_tokens)]
#     print(f"preprocess_split_text: Total chunks: {len(chunks)} type: {type(chunks)} , len of first chunk: {len(chunks[0])} , first chunk {chunks[0]}")
#     if len(chunks) > 1:
#         print(f"preprocess_split_text: len of second chunk: {len(chunks[0])} , second chunk {chunks[1]}")
#     return chunks

# def preprocess_text(text):
#     # Convert to lowercase
#     text = text.lower()
#     # Remove URLs
#     text = re.sub(r'http\S+', '', text)
#     # Remove special characters or HTML tags
#     text = re.sub(r'<.*?>', '', text)
#     #  Remove non-alphabetic characters (optional, depending on the task)
#     text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
#     # Remove extra spaces
#     text = ' '.join(text.split())
#     # Tokenize and remove stop words
#     words = text.split()
#     words = [word for word in words if word not in stop_words]
#     # Lemmatize words
#     words = [lemmatizer.lemmatize(word) for word in words]
#     processed_text = " ".join(words)
#     #print(f" Processed piece: {processed_text}")
#     return processed_tex

# # Example usage
# text = "Hello World! Running text \n\n\/ {} preprocessing is essential for NLP tasks."
# cleaned_text = preprocess_text(text)
# print(cleaned_text)  # Output: "hello world running text preprocessing essential nlp task"


# from multiprocessing import Pool, cpu_count

# # Helper to pass multiple arguments to the function
# def preprocess_wrapper(args):
#     return preprocess_split_text(*args)

# # Function to preprocess chunks in parallel
# def preprocess_chunks_in_parallel(chunks, tokenizer , num_processes=None):
#     preprocessed_chunks = []
#     # Use all available CPUs if num_processes is not specified
#     if num_processes is None:
#         num_processes = cpu_count()
#     # Pair each chunk with the tokenizer
#     tasks = [(chunk, tokenizer) for chunk in chunks]

#     with Pool(processes=num_processes) as pool:
#         preprocessed_chunks.extend(pool.map(preprocess_wrapper, tasks))

#     return preprocessed_chunks