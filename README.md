
'''
cd /path/to/your/project
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
'''

## 1. Text Ingestion Pipeline     

### Step 1: Loading Text
- Goal: Load text into a format ready for preprocessing
```python
import os

def load_text_from_directory(directory_path):
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Adjust for other formats
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

text_data = load_text_from_directory("data/text_files")

```
### Step 2: Preprocessing
Goals:
- Clean the text (remove special characters, normalize, etc.).
- Split the text into chunks.
- Ensure each chunk is meaningful and within the token limit of your embedding model.

    Basic Cleaning: Remove special characters, HTML tags, and unnecessary whitespaces.
    Chunking: Use fixed-length token chunks or sentence-based splitting with overlapping windows.

```python 
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_text(text, tokenizer, max_tokens=128, overlap=20):
    # Clean text
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower()).strip()
    # Tokenize and chunk
    tokenized = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokenized), max_tokens - overlap):
        chunk = tokenized[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return chunks

preprocessed_texts = []
for text in text_data:
    preprocessed_texts.extend(preprocess_text(text, tokenizer))

```

### Step 3: Embedding       
- Use a pre-trained embedding model (e.g., sentence-transformers or Hugging Face) to convert chunks into dense vectors.
- Store these embeddings in a vector database (e.g., Pinecone, Weaviate, FAISS).

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = embedding_model.encode(preprocessed_texts, convert_to_tensor=False)

# Store in FAISS
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean distance)
index.add(np.array(embeddings))

# Save the FAISS index to disk
faiss.write_index(index, "faiss_index.bin")

```

## 2. Query Processing Pipeline    
###  Step 1: Preprocessing the Query
- The query should be cleaned and tokenized in the same way as the text chunks during ingestion. Consistency is critical to ensure that the embeddings are comparable.

### Step 2: Embed the Query   
- Generate the embedding for the preprocessed query using the same embedding model used during ingestion.
```python
query_embedding = embedding_model.encode(query, convert_to_tensor=False)
```

### Step 3: Retrieve Relevant Chunks   
- Use the query embedding to find the most relevant chunks from the vector database.
- Retrieve the top-N most similar vectors (e.g., via cosine similarity or L2 distance).

```python
# Load FAISS index
index = faiss.read_index("faiss_index.bin")
# Search for the most relevant chunks
k = 5  # Number of results to retrieve
distances, indices = index.search(np.array([query_embedding]), k)
# Get the matching text chunks
retrieved_chunks = [preprocessed_texts[i] for i in indices[0]]
print(retrieved_chunks)
```

## Step 4: Combine Retrieved Chunks into Context    
- Combine the retrieved chunks into a single context block for the LLM to use.
```python
context = "\n".join(retrieved_chunks)
print(context)
```
## Step 5: Pass the Query and Context to the LLM   
- Use a pre-trained language model (e.g., GPT or Falcon) to generate the response based on the retrieved context and query.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

qa_model_name = "tiiuae/falcon-7b-instruct"
qa_model = AutoModelForCausalLM.from_pretrained(qa_model_name, device_map="auto")
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

def generate_answer(context, query, model, tokenizer, max_new_tokens=250):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
    output = model.generate(input_ids, max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()

# Generate the final answer
answer = generate_answer(context, query, qa_model, qa_tokenizer)
print("Answer:", answer)
```

## Key Notes
### Preprocessing Consistency:
- Preprocess both your text data and queries in the same way (cleaning, tokenization).
###Embedding Consistency:
- Use the same embedding model for both text chunks and query embeddings.
### Chunk Size:
- Choose an appropriate chunk size (e.g., 128 tokens). Larger chunks may contain more information but can exceed model limits.
### Retrieval:
- Fine-tune your retrieval mechanism to get relevant chunks. The number of chunks (k) to retrieve can depend on your task.
### LLM Input:
- Combine the retrieved chunks into a context for the LLM to answer the query effectively.


## Files

- confluence_reader.py - injestion pipeline/load texts        
Based upon the onyx project - confluence connector
- injest.py - injestion pipeline/preprocessing, embedding
- app.py - query processing pipeline/Preprocessing the Query,Embed the Query,
    Retrieve Relevant Chunks,Combine Retrieved Chunks into Context,Pass the Query and Context to the LLM
- config.py - load configuration and environment variables

