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
    print(f"!!!!!!!!!!!!!!!!! herfe in load_embed_text_from_directory {directory_path}!!!!")
    # Initialize an empty DataFrame
    dir_embeddings = pd.DataFrame(columns=['text_chunk', 'embedding'])
    for filename in os.listdir(directory_path):
        print(f"Processing file: {filename}")
        #if filename.endswith(".txt"):  # Adjust for other formats
        with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
            text = str(file.read())
            print(f"---------Processing file: {filename}, {text} -------------------")
            embeddings = run_embedding_pipeline_on_file(text, embedding_model, tokenizer, chunk_size, overlap)
            if embeddings is not None:
                dir_embeddings = pd.concat([dir_embeddings, embeddings], ignore_index=True)

    print(f"==> load_text_from_directory: Total embeddings: {len(dir_embeddings)} type: {type(dir_embeddings)}")
    # , len of first text: {len(dir_embeddings[0])} , first text {dir_embeddings[0]}")
    return dir_embeddings


def run_embedding_pipeline_on_file(text, embedding_model, tokenizer, chunk_size=500, overlap=50):
    # Define stop words and lemmatizer
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    # Preprocess and split the text into tokenized chunks
    #print(f"Calling preprocess_and_split_text   text: {text[:200]}...")  # Show first 200 chars
    chunks = preprocess_and_split_text(
        text=text,
        tokenizer=tokenizer,
        stop_words=stop_words,
        lemmatizer=lemmatizer,
        chunk_size=chunk_size,
        overlap=overlap
    )
    print(f"run_embedding_pipeline_on_file: Total chunks: {len(chunks)} type: {type(chunks)}")

    df = None
    if chunks is not None and len(chunks) > 0:
        try:
            # Enable debug information for the encode process
            embeddings = embedding_model.encode(chunks, convert_to_tensor=False, convert_to_numpy= True, show_progress_bar=True, output_value='token_embeddings')
        except KeyError as e:
            print(f"KeyError: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        #print(f"chunks len {len(chunks)}    type {type(chunks)}; \nembeddings: len {len(embeddings)}    type {type(embeddings)}")
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

        #print(f"========================> run_embedding_pipeline: Total records: {len(records)} type: {type(records)}")
        # Print records in human-readable format
        # print("\nHuman-readable records:")
        # for i, record in enumerate(records):
        #     print(f"\nRecord {i}:")
        #     print(f"Text chunk: {record['text_chunk'][:100]}...\n")  # First 100 chars
        #     print(f"Embedding : {type(record['embedding'])}\n - value: {record['embedding']}")

        # Create DataFrame from records
        df = pd.DataFrame(records)
        #print("=========================> ")
        # print("Final DataFrame:")
        # print(df.describe())
        # print(df.info())
        # print(df.shape)
        # print(f"type of embeddings : {type(df['embedding'])}")
        # print("=========================> End of run_embedding_pipeline")
    # else:
    #     print("No chunks to process")
    return df


def gpt_split_into_sections(content,  stop_words,
        lemmatizer, chunk_size=500, overlap_len=50 ):
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
        print(f"HTML headers found: {html_headers}")
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
                line = preprocess_text(line, stop_words, lemmatizer)
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
        print(f"===> Lines: {lines}")
        has_markdown_headers = any(line.startswith("#") for line in lines)
        print(f"======> Markdown headers found: {has_markdown_headers}")

        if has_markdown_headers:
            # Handle content with Markdown-style headers
            current_title = None
            current_text = []
            for line in lines:
                if line.startswith("#"):
                    print(f"=====> Markdown header found: {line}")
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
                    line = preprocess_text(line, stop_words, lemmatizer)
                    current_text.append(line)

            # Add the last section
            if current_title or current_text:
                sections.append({
                    "subtitle": current_title or "Untitled",
                    "text": "\n".join(current_text).strip()
                })
        else:
            # Handle plain text without headers by splitting into chunks
            content = preprocess_text(content, stop_words, lemmatizer)
            print("No headers found. Splitting text content into chunks.")
            start = 0
            while start < len(content):
                end = start + chunk_size
                chunk = content[start:end]

                # Add overlap to the next chunk, if it exists
                if end < len(content):
                    chunk = content[start:end + overlap_len]

                sections.append({
                    #"subtitle": "Plain Text Chunk",
                    "subtitle": "",
                    "text": chunk.strip()
                })
                start += chunk_size  # Move to the next chunk

    return sections

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
        #print(f"Section:\n======Subtitle:  {section['subtitle']} \n======text:   {section['text']}")
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
    #print(f"Input text: {text[:200]}...")  # Show first 200 chars
    #print(f"Chunk size: {chunk_size}, Overlap: {overlap}")
    # print(f"Number of stop words: {len(stop_words)}")
    #print(f"Text: {text[:200]}...")  # Show first 200 chars
    # Split content into sections
    sections = gpt_split_into_sections(text, stop_words, lemmatizer, chunk_size, overlap)
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
