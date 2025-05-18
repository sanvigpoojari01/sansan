import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 1. Load the sentence transformer for embedding generation (retrieval part)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Prepare some documents (or knowledge base) for retrieval
documents = [
    "The capital of France is Paris.",
    "The Python programming language is great for machine learning.",
    "ChatGPT is an AI language model created by OpenAI.",
    "The Earth revolves around the Sun."
]

# 3. Encode documents into embeddings
document_embeddings = embedder.encode(documents, convert_to_numpy=True)

# 4. Create a FAISS index for efficient retrieval
index = faiss.IndexFlatL2(document_embeddings.shape[1])  # Use L2 distance (Euclidean distance)
index.add(document_embeddings)

# 5. Load the generative model (e.g., T5 for text generation)
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function for retrieval
def retrieve_documents(query, top_k=1):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[idx] for idx in indices[0]]
    return retrieved_docs

# Function for response generation
def generate_response(query, retrieved_docs):
    context = " ".join(retrieved_docs)
    input_text = f"Context: {context} Question: {query}"
    
    # Tokenize and generate response
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Main loop for the chatbot
def chatbot(query):
    retrieved_docs = retrieve_documents(query)
    response = generate_response(query, retrieved_docs)
    return response

# Test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = chatbot(user_input)
    print(f"Bot: {response}")
