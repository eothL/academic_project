# libraries
from datasets import load_dataset
import os
from typing import BinaryIO
import regex as re
import multiprocessing #to parallelize the tokenization 
import pyarrow as pa
from collections import defaultdict, Counter
import multiprocessing 


# loading data
def load_data(input_path:str, file_name :str) -> str:
    """Load text data"""
    file_path = "/".join([input_path,file_name])
    with open(file_path, "r", encoding ='utf-8') as f:
        return f.read()

# test
# text_data = load_data(input_path,file_name)
# print(f'loaded{len(text_data)} characters')

def pre_tokenize(base_pattern :str, text: str,special_tokens: list[str]) -> dict[bytes,int]:
    """
    pre tokenize the text and return a list of pre tokenize text
    """
    assert isinstance(special_tokens, list)

    # define the pattern use by regex
    # by defining special pattern, we don't add the special tokens in the list
    special_pattern = '|'.join([re.escape(token) for token in special_tokens])
    full_pattern = f'({special_pattern})|({base_pattern})'


    # DEBUG: Let's see what patterns are being created
    print(f"Special tokens: {special_tokens}")
    print(f"Escaped special pattern: {special_pattern}")
    print(f"Base pattern: {base_pattern}")
    print(f"Full pattern: {full_pattern}")

    try: 
        re.compile(full_pattern)
        print("it is working well")
    except re.error as e:
        raise ValueError(f'invalid regex patter: {e}')


    pre_tokenize_list = []
    for match in re.finditer(full_pattern,text):
        pre_tokenize_list.append(match.group(0).encode("utf-8")) # we add and convert the word into bytes 
    
    return dict(Counter(pre_tokenize_list))



def replace_pair(seq: tuple[int, ...], pair: tuple[int,int], new_id : int)->tuple[int,...]:
    """    Return an tuple of updated pair   """

    a, b = pair
    merged = []
    i = 0
    while i < len(seq) :
        if i < len(seq) - 1 and seq[i] == a and seq[i+1] == b:
            merged.append(new_id)
            i += 2
        else:
            merged.append(seq[i])
            i += 1

    return tuple(merged)



def train_bpe(
    input_path: str,
    vocab_size : int,
    special_tokens : list[str]
) -> tuple[dict[int,bytes], list[tuple[bytes,bytes]]]:
    """  train a bpe tokenizer and return the merged vocab  """
    data 
    pretoken_counts = 0
    # initialization of vocabulary
    # we keep two dictionaries to accelerate the process as it is easier to make lookups
    id_to_token: dict[int,bytes] = {}  # maps vocab id (int) to token (bytes)
    token_to_id: dict[bytes, int] = {} # maps token (bytes) to vocab id (int)
    # initialization of the vocab with numbers from 0 to 255, so we will have bytes 0 to 255
    for i in range(256):
        id_to_token[i] = bytes([i])
        token_to_id[bytes([i])] = i

    # adding the special tokens to the vocab
    next_id = 256
    for tok in special_tokens:
        b = tok.encode("utf-8")
        id_to_token[next_id] = b
        token_to_id[b] = next_id
        next_id += 1

    base_size = 256 + len(special_tokens)
    assert vocab_size >= base_size

    # --- COMMENT VERIFICATION ---
    # Your comment: 
    # "we keep two list to accelerate the process as it is easier to make lookups"
    # Correction: 
    # These are actually two dictionaries (not lists), which allow for fast lookups in both directions:
    #   - id_to_token: from integer id to bytes token
    #   - token_to_id: from bytes token to integer id
    # This is a standard approach for vocabularies in tokenization, enabling efficient mapping in both directions.
    # The rest of your comments about initializing the vocab with bytes 0-255 and adding special tokens are correct.

    merge_history : list[tuple[bytes, bytes]] = [] # to store merge history records
    num_merges = vocab_size - base_size
    
    # prepares sequences once : initialization 
    sequences: dict[tuple[int, ...], int] = {}  # updated corpus: maps token id tuples to counts
    # we are building here a dictionary whose keys are tupples of ids (current token sequences) and whose values are the counts 
    # keep in minds that BPE works over token ids, so the first thing we do is express each pre-token as a sequence of ids. 
    for token_bytes, count in pretoken_counts.items(): 
        # initialization of the updated corpus : we transfer every data from the pre tokenize list to the sequences
        if token_bytes in token_to_id: # handle special token as they are already in the vocab 
            seq = (token_to_id[token_bytes],) # for a special token, we only have one element tuple 
        else:                                 # but for other words, we have multiple byte per word, for example the could be represented by a tuple like (116,104,101)
            seq = tuple([token_to_id[bytes([b])] for b in token_bytes]) # so we have to add every bytes inside the tuple to add the sequences 
        # The .get() method for dictionaries takes two arguments:
        #   - The key to look up (here, 'seq')
        #   - A default value to return if the key is not found (here, 0)
        # It returns the value associated with 'seq' if it exists in 'sequences', otherwise it returns 0.
        sequences[seq] = sequences.get(seq, 0) + count #safeguard to use sequences.get(seq,0) + count instead of just count
        # In which scenario we need it ? : if we go through another chunk and we encounter the same tuple, we will add the count 
        # instead of rewriting it 
    
    # number of merges
    for _ in range(num_merges):
        pair_counts = Counter()  # count how often each adjacent pair occurs
        for seq, count in sequences.items():
            for j in range(len(seq) - 1):
                pair_counts[(seq[j], seq[j + 1])] += count

        if not pair_counts:
            break  # nothing left to merge

        (a, b), freq = pair_counts.most_common(1)[0]
        new_bytes = id_to_token[a] + id_to_token[b]
        if new_bytes in token_to_id:
            new_id = token_to_id[new_bytes]
        else:
            new_id = len(id_to_token)
            merge_history.append((id_to_token[a], id_to_token[b]))
            id_to_token[new_id] = new_bytes
            token_to_id[new_bytes] = new_id

        updated_sequences: defaultdict[tuple[int, ...], int] = defaultdict(int)
        for seq, count in sequences.items():
            new_seq = replace_pair(seq, (a, b), new_id)
            updated_sequences[new_seq] += count
        sequences = dict(updated_sequences)
    return id_to_token, merge_history



if __name__ == "__main__":
    # init variable
    input_path = r"C:\Users\theo-\OneDrive\Documents\VS Code project\NLP_stanford_lecture\assignment1-basics\cs336_basics\dataset"
    file_name = "TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 600
    special_tokens =  ["<|endoftext|>"]


    # test for pre_tokenize and merging 
    text_test = "some text that i'll pre-tokenize and by pre-tokenize, I speak about pre-tokenize and i'll be right every time<|endoftext|>" 

    pretoken_counts= pre_tokenize(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",text_test,special_tokens)
    print(pretoken_counts)
    id_to_token, merge_list = train_bpe(pretoken_counts, vocab_size, special_tokens)
    print("bpe merging: \n",id_to_token)
