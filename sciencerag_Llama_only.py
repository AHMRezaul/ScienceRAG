import os
import requests
import hashlib
import warnings
import shutil
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from synonym_finder import generate_synonymous_sentences
from requests.exceptions import RequestException, ConnectionError
from colorama import Fore, Style
import torch
import psutil


# Suppress all warnings
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

embeddings = SentenceTransformerEmbeddings()
chroma_persist_dir = "chroma_db"
vector_store = Chroma(persist_directory=chroma_persist_dir, embedding_function=embeddings)

# Function to initialize LLaMA
def initialize_llama(model_path, cache_dir, token):
    """
    Initializes the LLaMA model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        token=token
    )
    tokenizer.pad_token = tokenizer.eos_token

    accelerator = Accelerator()
    device = accelerator.device

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        device_map="auto",
        offload_folder="offload",
        torch_dtype=torch.bfloat16,
        token=token
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_length": 2000
    }

    return tokenizer, model, device, generation_kwargs

# Function to call LLaMA for generating responses
def call_llama_for_combined_context(query, combined_context, tokenizer, model, device, generation_kwargs):
    """
    Calls LLaMA to generate a response based on the query and combined context.
    """
    try:
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You are a knowledgeable and helpful assistant. <|eot_id|><|start_header_id|>"
            f"user<|end_header_id|>\n\nBased on the following content, answer the query clearly and concisely.\n\n"
            f"Content:\n{combined_context}\n\nQuery: {query}\n\nAnswer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )

        query_encoding = tokenizer.encode(prompt)

        response_tensor = model.generate(
            torch.tensor(query_encoding).unsqueeze(dim=0).to(device),
            **generation_kwargs
        ).squeeze()[len(query_encoding):]

        response = tokenizer.decode(response_tensor, skip_special_tokens=True)
        return response.strip()

    except Exception as e:
        print(f"Error during LLaMA inference: {e}")
        return ""

# Helper functions for file and text processing
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            if "references" in page_text.lower():
                break
            text += page_text
    return text

def chunk_text(text, chunk_size=250, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def process_pdfs_and_store_in_chroma(directory, wikipedia_content, chroma_persist_dir="chroma_db"):
    embeddings = SentenceTransformerEmbeddings()
    vector_store = Chroma(persist_directory=chroma_persist_dir, embedding_function=embeddings)  # No embeddings for LLaMA

    for file_name in os.listdir(directory):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(directory, file_name)
            text = read_pdf(file_path)
            if not text.strip():
                continue

            chunks = chunk_text(text)
            documents = [
                Document(page_content=chunk, metadata={"source": file_name, "chunk_hash": hashlib.md5(chunk.encode()).hexdigest()})
                for chunk in chunks
            ]
            vector_store.add_documents(documents)

    if wikipedia_content.strip():
        chunks = chunk_text(wikipedia_content)
        documents = [
            Document(page_content=chunk, metadata={"source": "Wikipedia", "chunk_hash": hashlib.md5(chunk.encode()).hexdigest()})
            for chunk in chunks
        ]
        vector_store.add_documents(documents)

    vector_store.persist()
    return vector_store

def retrieve_relevant_content(query, vector_store, top_k=50):
    results = vector_store.similarity_search(query, k=top_k * 2)
    unique_results = []
    seen_hashes = set()

    for result in results:
        chunk_hash = result.metadata.get("chunk_hash")
        if chunk_hash not in seen_hashes:
            unique_results.append(result)
            seen_hashes.add(chunk_hash)
        if len(unique_results) >= top_k:
            break

    return unique_results

def force_delete_folder(folder_path):
        """
        Forcefully delete a folder, ensuring all files are unlocked and removed.
        """
        if os.path.exists(folder_path):
            try:
                # Terminate processes accessing files in the folder
                for proc in psutil.process_iter():
                    try:
                        for open_file in proc.open_files():
                            if folder_path in open_file.path:
                                proc.terminate()  # Kill the process
                                proc.wait()       # Wait for process to terminate
                    except Exception:
                        pass  # Ignore errors for processes we can't access
                
                # Try deleting the folder
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")
            except Exception as e:
                print(f"Error forcefully deleting folder {folder_path}: {e}")
        else:
            print(f"Folder {folder_path} does not exist.")

# Main function
def main():
    print(f"{Fore.GREEN}ScienceRAG: Hi! What would you like to ask today? (Type '\\exit' to end the chat.){Style.RESET_ALL}")
    context = []
    pdf_directory = "downloaded_pdfs"
    chroma_directory = "chroma_db"
    vector_store = None

    # LLaMA model setup
    model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Replace with your model
    cache_dir = "E:/Llama/cache"
    token = "your-llama-token" # Replace with your token
    tokenizer, model, device, generation_kwargs = initialize_llama(model_path, cache_dir, token)

    while True:
        query = input(f"{Fore.BLUE}{'You: '}{Style.RESET_ALL}")
        if query.strip().lower() == "\\exit":
            print(f"{Fore.GREEN}ScienceRAG: Goodbye! Have a great day!{Style.RESET_ALL}")
            break
        elif query.strip().lower() == "\\refresh":
            print(f"{Fore.YELLOW}Refreshing data...{Style.RESET_ALL}")
            force_delete_folder(pdf_directory)
            force_delete_folder(chroma_directory)
            vector_store = None  # Reset vector store
            print(f"{Fore.GREEN}All data has been refreshed!{Style.RESET_ALL}")
            continue

        context.append({"role": "user", "content": query})
        synonymous_queries = generate_synonymous_sentences(query, max_variations=3)
        queries = synonymous_queries + [query]

        # Process documents and Wikipedia content
        wikipedia_content = "Placeholder Wikipedia content"  # Replace with actual Wikipedia fetch logic
        if vector_store is None:
            vector_store = process_pdfs_and_store_in_chroma(pdf_directory, wikipedia_content, chroma_directory)

        results = retrieve_relevant_content(query, vector_store, top_k=50)
        combined_context = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in context
        )
        final_answer = call_llama_for_combined_context(
            query=query,
            combined_context=combined_context,
            tokenizer=tokenizer,
            model=model,
            device=device,
            generation_kwargs=generation_kwargs
        )
        context.append({"role": "assistant", "content": final_answer})
        print(f"{Fore.GREEN}ScienceRAG: {final_answer}{Style.RESET_ALL}")

    shutil.rmtree(pdf_directory, ignore_errors=True)
    shutil.rmtree(chroma_directory, ignore_errors=True)

if __name__ == "__main__":
    main()
