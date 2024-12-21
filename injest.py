import re
import os
import torch

from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from typing import List
from pydantic import BaseModel, ValidationError

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt')

# stop_words = set(stopwords.words("english"))
# lemmatizer = WordNetLemmatizer()


# def load_text_from_directory(directory_path , tokenizer):
#     texts = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".txt"):  # Adjust for other formats
#             with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
#                 chunks = preprocess_and_split_text(file.read(), tokenizer)                
#                 texts.extend(chunks)

#     print(f"load_text_from_directory: Total texts: {len(texts)} type: {type(texts)} , len of first text: {len(texts[0])} , first text {texts[0]}")
#     return texts


def load_embed_text_from_directory(directory_path , tokenizer , embedding_model):
    dir_embeddings = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Adjust for other formats
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:              
                embeddings = run_embedding_pipeline(file.read(), embedding_model, tokenizer, chunk_size=128, overlap=20)
                dir_embeddings.extend(embeddings)
    print(f"load_text_from_directory: Total embeddings: {len(dir_embeddings)} type: {type(dir_embeddings)}")
    # , len of first text: {len(dir_embeddings[0])} , first text {dir_embeddings[0]}")
    return dir_embeddings



class EmbeddingInput(BaseModel):
    tokenized_chunks: List[List[int]]    
    device: str = "cpu"

def run_embedding_pipeline(text, embedding_model, tokenizer, chunk_size=128, overlap=20):    
    # Define stop words and lemmatizer
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    # Preprocess and split the text into tokenized chunks
    tokenized_chunks = preprocess_and_split_text(
        text=text,
        tokenizer=tokenizer,
        stop_words=stop_words,
        lemmatizer=lemmatizer,
        chunk_size=chunk_size,
        overlap=overlap  
    )

    # Create an instance of EmbeddingInput
    embedding_input = EmbeddingInput(
        tokenized_chunks=tokenized_chunks,
        device = "cuda" if torch.cuda.is_available() else "cpu" 
    )
    # Generate embeddings
    embeddings = get_embeddings(embedding_input, embedding_model)

    return embeddings

def preprocess_text(text: str, stop_words : set, lemmatizer ) -> str:
    """
    Preprocesses the input text by:
      - Converting to lowercase
      - Removing URLs, HTML tags, and special characters
      - Removing stop words
      - Lemmatizing words
    """
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters (optional)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Remove extra spaces
    text = ' '.join(text.split())
    # Tokenize and remove stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

max_length = 512 # for BERT tokenizer
def preprocess_and_split_text(
    text: str,
    tokenizer,
    stop_words: set,
    lemmatizer ,
    chunk_size: int = 128,
    overlap: int = 20   
) -> List[List[int]]:
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
        List[List[int]]: A list of tokenized text chunks.
    """
    # Preprocess the text
    cleaned_text = preprocess_text(text, stop_words, lemmatizer)
    # Tokenize the cleaned text
    tokenized = tokenizer.encode(
        cleaned_text, return_tensors="pt", padding=False, truncation=False , max_length = max_length
    )
    # Remove special tokens ([CLS] and [SEP])
    tokens_cleaned = [
        token.item()
        for token in tokenized[0]
        if token not in {tokenizer.cls_token_id, tokenizer.sep_token_id}
    ]
    # Chunk the tokens
    chunks = []
    for i in range(0, len(tokens_cleaned), chunk_size - overlap):
        chunk = tokens_cleaned[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def preprocess_and_sentence_split(
    text: str,
    tokenizer,
    stop_words: set ,
    lemmatizer
) -> List[List[int]]:
    """
    Preprocesses text, splits it into sentences, and tokenizes each sentence.
    Args:
        text (str): The input text to preprocess.
        tokenizer: A tokenizer that provides `encode`.
        stop_words (set): A set of stop words to filter.
        lemmatizer: A lemmatizer to normalize words.

    Returns:
        List[List[int]]: A list of tokenized sentences.
    """
    # Preprocess the text
    cleaned_text = preprocess_text(text, stop_words, lemmatizer)
    # Split into sentences
    sentences = sent_tokenize(cleaned_text)
    # Tokenize each sentence
    tokenized_sentences = [
        tokenizer.encode(sentence, return_tensors="pt", padding=False, truncation=False, max_length = max_length)[0].tolist()
        for sentence in sentences
    ]

    return tokenized_sentences


def get_embeddings(input_data: EmbeddingInput, embedding_model) -> List[torch.Tensor]:
    """
    Generates embeddings for the provided tokenized chunks.
    Args:
        input_data (EmbeddingInput): A Pydantic model containing tokenized_chunks and device information.
        embedding_model: The embedding model to use.

    Returns:
        List[torch.Tensor]: A list of embedding tensors.
    """
    try:        
        input_data = EmbeddingInput(**input_data.model_dump())
    except ValidationError as e:
        print(f"Validation error: {e}")
        return []

    tokenized_chunks = input_data.tokenized_chunks
    device = input_data.device

    embeddings = []

    for input_ids in tokenized_chunks:
        # Convert to tensor if not already
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Move tensor to the correct device
        input_ids = input_ids.to(device)

        # Ensure input_ids is 1D
        if input_ids.ndimension() > 1:
            input_ids = input_ids.squeeze()
        
        # print(f"Current device: {device}")
        # # Print final dimensions and shape of input_ids
        # print(f"input_ids dimensions: {input_ids.ndimension()}")
        # print(f"input_ids shape: {input_ids.shape}")
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)
        # print(f"attention_mask dimensions: {attention_mask.ndimension()}")
        # print(f"attention_mask shape: {attention_mask.shape}")
        # print(input_ids.device)
        # print(attention_mask.device)
        # print(embedding_model.device)

        # Pass through the model
        with torch.no_grad():
            output = embedding_model(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))

        # Extract embeddings (mean pooling assumed)
        chunk_embeddings = output.last_hidden_state.mean(dim=1)
        # Move embeddings to the specified device
        if device == "cpu":
            chunk_embeddings = chunk_embeddings.cpu()
            print(f"chunk_embeddings device: {chunk_embeddings.device}")
        embeddings.append(chunk_embeddings)

    
    return embeddings

# =======================================================

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
#     return processed_text





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