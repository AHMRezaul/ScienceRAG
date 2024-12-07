import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from itertools import product
import random
from nltk.tag.perceptron import PerceptronTagger

# Force the PerceptronTagger to use the default "averaged_perceptron_tagger"
PerceptronTagger.lang = "default"




# Ensure NLTK data is downloaded
#nltk.data.path.append('/nltk_data')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')  # Fix here
#nltk.download('wordnet')
#nltk.download('omw-1.4')

def get_synonyms(word, pos):
    """Get synonyms for a word based on its part of speech."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        if pos.startswith('N') and syn.pos() != 'n':
            continue
        if pos.startswith('V') and syn.pos() != 'v':
            continue
        if pos.startswith('J') and syn.pos() != 'a':
            continue
        if pos.startswith('R') and syn.pos() != 'r':
            continue
        for lemma in syn.lemmas():
            if lemma.name() != word and '_' not in lemma.name():
                synonyms.add(lemma.name())
    return list(synonyms)

def pos_to_wordnet(treebank_tag):
    """Convert Penn Treebank POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return 'J'
    elif treebank_tag.startswith('V'):
        return 'V'
    elif treebank_tag.startswith('N'):
        return 'N'
    elif treebank_tag.startswith('R'):
        return 'R'
    else:
        return None

def generate_synonymous_sentences(sentence, max_variations=3):
    """Generate multiple synonymous versions of the input sentence."""
    # Tokenize and get POS tags
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)
    
    # Get synonyms for each word
    word_alternatives = []
    for word, pos in pos_tags:
        wordnet_pos = pos_to_wordnet(pos)
        if wordnet_pos:
            synonyms = get_synonyms(word.lower(), pos)
            word_alternatives.append([word] + synonyms if synonyms else [word])
        else:
            word_alternatives.append([word])
    
    # Generate combinations
    all_combinations = list(product(*word_alternatives))
    
    # Select random subset if too many combinations
    if len(all_combinations) > max_variations:
        all_combinations = random.sample(all_combinations, max_variations)
    
    # Convert combinations to sentences
    sentences = [' '.join(combo) for combo in all_combinations]
    
    # Remove duplicates and original sentence
    sentences = list(set(sentences))
    if sentence in sentences:
        sentences.remove(sentence)
    
    return sentences
def main():
    print("Welcome to the Synonym Generator!")
    print("Enter a sentence to generate synonymous versions (or type '\\exit' to quit):")
    
    
    # Get user input
    sentence = "I am a struggling PhD student"
    
    
    
        
    
    # Generate synonymous sentences
    try:
        print("\nGenerating synonymous sentences...")
        variations = generate_synonymous_sentences(sentence, max_variations=5)
        
        if variations:
            print("\nGenerated Variations:")
            for i, variation in enumerate(variations, 1):
                print(f"{i}. {variation}")
        else:
            print("No synonymous sentences could be generated.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
