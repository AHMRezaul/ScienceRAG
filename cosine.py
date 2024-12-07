import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from synonym_finder import generate_synonymous_sentences  # Import your rewriting script functions

# Step 1: Load the dataset
def load_dataset(file_path):
    """
    Load the dataset of queries from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing the queries.
        
    Returns:
        pd.DataFrame: DataFrame containing the original queries.
    """
    try:
        df = pd.read_csv(file_path)
        if 'Query' not in df.columns:
            raise ValueError("The CSV file must contain a 'query' column.")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Step 2: Generate rewritten queries
def generate_rewritten_queries(queries, max_variations=1):
    """
    Generate rewritten queries using the synonym-based script.
    
    Args:
        queries (list): List of original queries.
        max_variations (int): Number of variations to generate per query.
        
    Returns:
        list: List of rewritten queries.
    """
    rewritten_queries = []
    for query in queries:
        try:
            # Use the synonym generator to create one rewritten query per original query
            variations = generate_synonymous_sentences(query, max_variations=max_variations)
            rewritten_queries.append(variations[0] if variations else query)
        except Exception as e:
            print(f"Error generating rewrites for query '{query}': {e}")
            rewritten_queries.append(query)  # Fallback to original query if rewriting fails
    return rewritten_queries

# Step 3: Calculate cosine similarity
def calculate_cosine_similarity(original_queries, rewritten_queries, model_name='all-MiniLM-L6-v2'):
    """
    Calculate cosine similarity between original and rewritten queries.
    
    Args:
        original_queries (list): List of original queries.
        rewritten_queries (list): List of rewritten queries.
        model_name (str): Name of the embedding model to use.
        
    Returns:
        list: List of cosine similarity scores.
    """
    try:
        model = SentenceTransformer(model_name)
        original_embeddings = model.encode(original_queries)
        rewritten_embeddings = model.encode(rewritten_queries)
        similarities = [
            cosine_similarity([orig], [rew])[0][0]
            for orig, rew in zip(original_embeddings, rewritten_embeddings)
        ]
        return similarities
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return [0.0] * len(original_queries)

# Step 4: Save results to CSV
def save_results_to_csv(df, output_path):
    """
    Save the DataFrame with results to a new CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame containing queries, rewrites, and scores.
        output_path (str): Path to save the output CSV file.
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

# Step 5: Main execution
def calculate_total_cosine_similarity(original_queries, rewritten_queries, model_name='all-MiniLM-L6-v2'):
    """
    Calculate the total average cosine similarity for the dataset.
    
    Args:
        original_queries (list): List of original queries.
        rewritten_queries (list): List of rewritten queries.
        model_name (str): Name of the embedding model to use.
        
    Returns:
        float: Average cosine similarity score for the entire dataset.
        list: List of individual cosine similarity scores.
    """
    try:
        model = SentenceTransformer(model_name)
        original_embeddings = model.encode(original_queries)
        rewritten_embeddings = model.encode(rewritten_queries)
        similarities = [
            cosine_similarity([orig], [rew])[0][0]
            for orig, rew in zip(original_embeddings, rewritten_embeddings)
        ]
        total_similarity = sum(similarities) / len(similarities)
        return total_similarity, similarities
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0.0, []

# Example integration in the main function:
def main():
    # File paths
    input_csv = "data.csv"  # Replace with your input file path
    output_csv = "query_rewrite_results.csv"

    # Load the dataset
    df = load_dataset(input_csv)
    if df is None:
        return

    # Extract original queries
    queries = df['Query'].tolist()

    # Generate rewritten queries
    print("Generating rewritten queries...")
    rewritten_queries = generate_rewritten_queries(queries, max_variations=1)

    # Calculate cosine similarity
    print("Calculating cosine similarity...")
    total_similarity, similarities = calculate_total_cosine_similarity(queries, rewritten_queries)

    # Add results to the DataFrame
    df['rewritten_query'] = rewritten_queries
    df['cosine_similarity'] = similarities

    # Display total similarity
    print(f"Total Average Cosine Similarity: {total_similarity:.2f}")

    # Save results to CSV
    save_results_to_csv(df, output_csv)


if __name__ == "__main__":
    main()
