import os
import requests
import hashlib
from PyPDF2 import PdfReader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from openai import OpenAI
import warnings
from colorama import Fore, Style
# Suppress all warnings
warnings.filterwarnings("ignore")
import shutil
import os
from synonym_finder import generate_synonymous_sentences
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import os
import requests
from requests.exceptions import RequestException, ConnectionError
import psutil



# Set your OpenAI API key
client = OpenAI(
    api_key='your-open-ai-key'  
)


# Step 1: Fetch arXiv papers
import requests
import os

def fetch_arxiv_papers_api(queries, max_results=3):
    """
    Fetch relevant paper PDF links from arXiv using the API.

    Args:
        queries (list): A list of search queries.
        max_results (int): Maximum results per query.


    Returns:
        set: A set of unique PDF links.
    """
    base_url = "http://export.arxiv.org/api/query"
    pdf_links = set()

    for query in queries:
        #print(f"Searching for: {query}")
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance"
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.text

            # Extract PDF links
            entries = data.split("<entry>")
            for entry in entries[1:]:
                if "<link title=\"pdf\"" in entry:
                    start_idx = entry.find("<link title=\"pdf\" href=\"") + len("<link title=\"pdf\" href=\"")
                    end_idx = entry.find("\"", start_idx)
                    pdf_link = entry[start_idx:end_idx]
                    pdf_links.add(pdf_link)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
    return pdf_links





def search_and_download_pmc_pdfs(query, max_results=3, save_directory="downloaded_pdfs"):
    """
    Search PubMed Central (PMC) for articles based on a query, extract PMCIDs, and download their PDFs.

    Args:
        query (str): The search query.
        max_results (int): Maximum number of PMCIDs to retrieve.
        save_directory (str): Directory to save the downloaded PDFs.

    Returns:
        None
    """
    # Ensure the save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Step 1: Search PMC and extract PMCIDs
    base_search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pmc",          # Search in PubMed Central
        "term": query,        # The search query
        "retmax": max_results,  # Number of results to retrieve
        "retmode": "json"     # Return results in JSON format
    }

    #print(f"Searching PubMed Central for query: {query}")
    try:
        # Send the GET request to search for articles
        search_response = requests.get(base_search_url, params=search_params)
        search_response.raise_for_status()  # Check for HTTP errors

        # Parse the JSON response to get PMCIDs
        search_data = search_response.json()
        pmc_ids = ["PMC" + pmcid for pmcid in search_data.get("esearchresult", {}).get("idlist", [])]

        if not pmc_ids:
            #print("No articles found for the given query.")
            return

        #print(f"Retrieved PMCIDs: {pmc_ids}")

    except RequestException as e:
        #print(f"Error during PMC search: {e}")
        return

    # Step 2: Download PDFs for each PMCID
    base_pdf_url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for pmc_id in pmc_ids:
        try:
            pdf_url = f"{base_pdf_url}{pmc_id}/pdf"
            pdf_response = requests.get(pdf_url, headers=headers, stream=True)
            pdf_response.raise_for_status()  # Check for request errors

            pdf_filename = os.path.join(save_directory, f"{pmc_id}.pdf")
            with open(pdf_filename, 'wb') as pdf_file:
                for chunk in pdf_response.iter_content(chunk_size=1024):
                    if chunk:
                        pdf_file.write(chunk)

        except requests.exceptions.HTTPError as e:
            print(f"HTTPError for {pmc_id}: {e}")
            continue  # Skip to the next PMCID
        except Exception as e:
            print(f"Error for {pmc_id}: {e}")
            continue

 








def download_pdfs(pdf_links, save_dir="downloaded_pdfs"):
    """
    Download PDFs from arXiv.

    Args:
        pdf_links (set): Unique PDF links.
        save_dir (str): Directory to save PDFs.

    Returns:
        None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, link in enumerate(pdf_links, 1):
        try:
            #print(f"Downloading: {link}")
            response = requests.get(link, stream=True)
            response.raise_for_status()
            file_path = os.path.join(save_dir, f"paper_{i}.pdf")
            with open(file_path, "wb") as pdf_file:
                for chunk in response.iter_content(chunk_size=1024):
                    pdf_file.write(chunk)
            #print(f"Saved to: {file_path}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {link}: {e}")

# Step 2: Load Wikipedia content
def load_wikipedia_content(search_term, limit_sections=5):
    """
    Loads Wikipedia content for a given search term using LangChain's WikipediaLoader.

    Args:
        search_term (str): The Wikipedia page title to search for.
        limit_sections (int): Maximum number of sections to include.

    Returns:
        str: Combined content of the fetched sections or an error message if the page is not found.
    """
    try:
        # Fetch Wikipedia content
        loader = WikipediaLoader(query=search_term)
        documents = loader.load()
        
        if not documents or not documents[0].page_content.strip():
            return f"Error: Wikipedia page '{search_term}' does not exist or has no content."

        # Limit the number of sections to process
        limited_documents = documents[:limit_sections]
        content = "\n\n".join(doc.page_content for doc in limited_documents)
        
        return content

    except Exception as e:
        return f"Error: {str(e)}"

# Step 3: Process PDFs and Wikipedia content into Chroma

def read_pdf(file_path):
    """
    Reads text from a PDF file and filters non-content sections.
    
    Args:
        file_path (str): Path to the PDF file.
    
    Returns:
        str: Filtered text from the PDF.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            # Remove potential "References" sections
            if "references" in page_text.lower():
                break
            text += page_text
    return text

def chunk_text(text, chunk_size=250, chunk_overlap=100):
    """
    Splits text into manageable chunks.
    
    Args:
        text (str): Text to be chunked.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
    
    Returns:
        list: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def process_pdfs_and_store_in_chroma(directory, wikipedia_content, chroma_persist_dir="chroma_db"):
    """
    Processes PDFs in a directory, chunks the text, and stores it in Chroma Vector DB.
    
    Args:
        directory (str): Directory containing PDF files.
        chroma_persist_dir (str): Directory to store the Chroma database.
    
    Returns:
        Chroma: The initialized vector store.
    """
    embeddings = OpenAIEmbeddings(openai_api_key="your-open-ai-key")
    vector_store = Chroma(persist_directory=chroma_persist_dir, embedding_function=embeddings)

    for file_name in os.listdir(directory):
        #print(file_name)
        if file_name.endswith(".pdf"):
            file_path = os.path.join(directory, file_name)
            #print(f"Processing {file_name}...")
            
            # Extract text
            text = read_pdf(file_path)
            if not text.strip():
                #print(f"No text extracted from {file_name}. Skipping.")
                continue

            # Chunk text and add metadata
            chunks = chunk_text(text)
            documents = []
            for i, chunk in enumerate(chunks):
                # Create a unique hash for each chunk
                chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
                documents.append(
                    Document(page_content=chunk, metadata={"source": file_name, "chunk_hash": chunk_hash})
                )

            # Store in Chroma
            vector_store.add_documents(documents)
    if wikipedia_content.strip():
        #print("Processing Wikipedia content...")
        chunks = chunk_text(wikipedia_content)
        documents = [
            Document(page_content=chunk, metadata={"source": "Wikipedia", "chunk_hash": hashlib.md5(chunk.encode()).hexdigest()})
            for chunk in chunks
        ]

        # Store in Chroma
        vector_store.add_documents(documents)
    
    # Persist Chroma DB
    vector_store.persist()
    #print(f"Vector DB stored in {chroma_persist_dir}")
    return vector_store

def retrieve_relevant_content(query, vector_store, top_k=50,deduplicate=False):
    """
    Retrieves the most relevant content for a query from the Chroma Vector DB.
    Filters duplicate content by chunk hash.
    
    Args:
        query (str): The query for which to retrieve relevant content.
        vector_store (Chroma): The vector store to search in.
        top_k (int): The number of top results to return.
    
    Returns:
        list: The most relevant content chunks and their metadata.
    """
    results = vector_store.similarity_search(query, k=top_k * 2)  # Fetch more to filter duplicates
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


client = OpenAI(
  api_key="your-open-ai-key"  # Replace with your actual key
  # this is also the default, it can be omitted
)






def combine_results_into_chunks(results, chunk_size=2500, chunk_overlap=250):
    """
    Combines retrieved content into chunks of specified size with overlap.
    
    Args:
        results (list): List of retrieved results with page_content.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
    
    Returns:
        list: List of text chunks.
    """
    chunks = []
    current_chunk = ""

    for result in results:
        content = result.page_content
        if len(current_chunk) + len(content) <= chunk_size:
            current_chunk += content + " "
        else:
            chunks.append(current_chunk.strip())
            # Start a new chunk with overlap
            current_chunk = current_chunk[-chunk_overlap:] + content + " "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def combine_chunks_for_model(chunks, max_combined_length=15000):
    """
    Combines multiple chunks into a single input for the model, ensuring it fits within token limits.
    
    Args:
        chunks (list): List of individual chunks.
        max_combined_length (int): Maximum length of the combined chunks in characters.
    
    Returns:
        str: Combined content for the model.
    """
    combined = ""
    for chunk in chunks:
        if len(combined) + len(chunk) <= max_combined_length:
            combined += chunk + "\n\n"
        else:
            break
    return combined.strip()

def call_gpt_for_combined_context(query, combined_context):
    """
    Calls OpenAI GPT-4 model with a query and a combined context.
    
    Args:
        query (str): The query to be answered.
        combined_context (str): The combined content to provide context.
    
    Returns:
        str: The model's response.
    """
    try:
        messages = [
            {"role": "system", "content": "You are a knowledgeable and helpful assistant."},
            {
                "role": "user",
                "content": f"Based on the following content, answer the query clearly and concisely.\n\nContent:\n{combined_context}\n\nQuery: {query}\n\nAnswer:",
            },
        ]

        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        #print(f"Error calling GPT-4: {e}")
        return ""

def generate_answer_from_combined_chunks(query, results, chunk_size=2500, chunk_overlap=250):
    """
    Processes the Chroma results into chunks, combines them into a single context,
    and calls GPT-4 to generate an answer.
    
    Args:
        query (str): The query to be answered.
        results (list): Relevant content retrieved from Chroma.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
    
    Returns:
        str: The generated answer.
    """
    # Step 1: Chunk the results
    chunks = combine_results_into_chunks(results, chunk_size, chunk_overlap)
    #print(f"Total chunks created: {len(chunks)}")

    # Step 2: Combine chunks into a single context
    combined_context = combine_chunks_for_model(chunks)
    #print(f"Combined context length: {len(combined_context)} characters")

    # Step 3: Pass the combined context to the GPT model
    answer = call_llama_for_combined_context(query, combined_context)
    return answer

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

def cleanup_folders():
    """
    Forcefully deletes the `downloaded_pdfs` and `chroma_db` folders if they exist.
    """
    folders_to_delete = ["downloaded_pdfs", "chroma_db"]

    for folder in folders_to_delete:
        force_delete_folder(folder)


# Initialize LLaMA model and tokenizer
def initialize_llama(model_path, cache_dir, token):
    """
    Initializes the LLaMA 3.1 model and tokenizer.

    Args:
        model_path (str): Path to the model.
        cache_dir (str): Cache directory for loading the model.
        token (str): Hugging Face token for authentication.

    Returns:
        tuple: Tokenizer, model, device, and generation arguments.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        token=token
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize accelerator and device
    accelerator = Accelerator()
    device = accelerator.device

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        device_map="auto",
        offload_folder="offload",
        torch_dtype=torch.bfloat16,
        token=token
    )

    # Define generation arguments
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_length": 2000
    }

    return tokenizer, model, device, generation_kwargs

# Function to call LLaMA for combined context
def call_llama_for_combined_context(query, combined_context, tokenizer, model, device, generation_kwargs):
    """
    Calls the LLaMA 3.1 model with a query and combined context to generate a response.

    Args:
        query (str): The query to be answered.
        combined_context (str): The combined content to provide context.
        tokenizer (AutoTokenizer): Tokenizer for the LLaMA model.
        model (AutoModelForCausalLM): LLaMA model for inference.
        device (torch.device): Device for computation (CPU/GPU).
        generation_kwargs (dict): Arguments for the generation process.

    Returns:
        str: The model's response.
    """
    try:
        # Define the system prompt with context and query
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You are a knowledgeable and helpful assistant. <|eot_id|><|start_header_id|>"
            f"user<|end_header_id|>\n\nBased on the following content, answer the query clearly and concisely.\n\n"
            f"Content:\n{combined_context}\n\nQuery: {query}\n\nAnswer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )

        # Tokenize the prompt
        query_encoding = tokenizer.encode(prompt)

        # Generate the response
        response_tensor = model.generate(
            torch.tensor(query_encoding).unsqueeze(dim=0).to(device),
            **generation_kwargs
        ).squeeze()[len(query_encoding):]

        # Decode the response
        response = tokenizer.decode(response_tensor, skip_special_tokens=True)
        return response.strip()

    except Exception as e:
        print(f"Error during LLaMA inference: {e}")
        return ""



def main():
    print(f"{Fore.GREEN}ScienceRAG: Hi! What would you like to ask today? (Type '\\exit' to end the chat.){Style.RESET_ALL}")
    
    # Initialize conversation context
    context = []
    pdf_directory = "downloaded_pdfs"  # Directory containing downloaded PDFs
    chroma_directory = "chroma_db"    # Directory to store the Chroma database
    vector_store = None

    while True:
        # Get user input
        terminal_width = os.get_terminal_size().columns
        query = input(f"{Fore.BLUE}{'You: '}{Style.RESET_ALL}")
        #uery = input(input_prompt.rjust(terminal_width))
        
        # Check if the user wants to exit
        if query.strip().lower() == "\\exit":
            print(f"{Fore.GREEN}ScienceRAG: Goodbye! Have a great day!{Style.RESET_ALL}")
            break

        # Add the current query to the context
        context.append({"role": "user", "content": query})

        # Fetch relevant documents and process content for retrieval
        synonymous_queries = generate_synonymous_sentences(query, max_variations=3)
        synonymous_queries.append(query)
        #print(synonymous_queries)
        queries = synonymous_queries
        unique_pdf_links = fetch_arxiv_papers_api(queries)

        search_and_download_pmc_pdfs(query)

        # Downlaod the PDFs
        download_pdfs(unique_pdf_links)
        wikipedia_content = load_wikipedia_content(query, limit_sections=5)
        #print(wikipedia_content)
        
        # If vector_store is not already initialized, process PDFs and Wikipedia content
        if vector_store is None:
            vector_store = process_pdfs_and_store_in_chroma(pdf_directory, wikipedia_content, chroma_directory)
        
        # Retrieve relevant content based on the current query
        results = retrieve_relevant_content(query, vector_store, top_k=50)

        # Combine context with the current query for generating a response
        combined_context = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in context
        )
        model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        cache_dir = "F:\Llama\cache"
        token = "your-llama-token" # Replace with your actual token  

        tokenizer, model, device, generation_kwargs = initialize_llama(model_path, cache_dir, token)

        final_answer = call_llama_for_combined_context(
            query=query,
            combined_context=combined_context,
            tokenizer=tokenizer,
            model=model,
            device=device,
            generation_kwargs=generation_kwargs
        )

        # Add the model's response to the context
        context.append({"role": "assistant", "content": final_answer})

        # Display the chatbot's response
        print(f"{Fore.GREEN}ScienceRAG: {final_answer}{Style.RESET_ALL}")
        
    cleanup_folders()

    
   
    

if __name__ == "__main__":
    main()
