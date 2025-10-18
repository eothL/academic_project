# libraries
import regex as re
import multiprocessing #to parallelize the tokenization 
from collections import defaultdict, Counter
import cProfile
import pstats

def get_compression_rate(token_list : list, string: str) ->float:
    """Given string that has been tokenized into indices """
    num_bytes = len(bytes(string, encoding= "utf-8"))
    num_tokens = len(token_list)
    return num_bytes/num_tokens

def pretokenize(base_pattern :str, text: str,special_tokens: list[str]) -> dict[bytes,int]:
    """
    pre tokenize the text and return a list of pre tokenize text
    """
    assert isinstance(special_tokens, list)

    # define the pattern use by regex
    # by defining special pattern, we don't add the special tokens in the list
    if special_tokens:
        special_pattern = '|'.join([re.escape(token) for token in special_tokens])
        full_pattern = f'({special_pattern})|({base_pattern})'
    else:
        full_pattern = base_pattern

    # # DEBUG: Let's see what patterns are being created
    # print(f"Special tokens: {special_tokens}")
    # print(f"Escaped special pattern: {special_pattern}")
    # print(f"Base pattern: {base_pattern}")
    # print(f"Full pattern: {full_pattern}")

    # try: 
    #     re.compile(full_pattern)
    #     print("working pattern")
    # except re.error as e:
    #     raise ValueError(f'invalid regex pattern: {e}')


    pre_tokenize_list = []
    for match in re.finditer(full_pattern,text):
        pre_tokenize_list.append(match.group(0).encode("utf-8")) # we add and convert the word into bytes 
    
    return dict(Counter(pre_tokenize_list))


def replace_pair(seq: tuple[int, ...], pair: tuple[int,int], new_id : int)->tuple[int,...]:
    """    Return an tuple of updated pair, needed if the pair_count dict is created at each merge  """

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
    print(" Starting BPE training...")
    
    # Phase 1: Reading file
    print(" Reading input file...")
    with open(input_path, "r", encoding = "utf-8") as f:
        data = f.read()

    
    # Phase 2: Pretokenization
    print(" Pretokenizing text...")
    gpt2_regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretoken_counts = pretokenize(gpt2_regex, data, special_tokens)

    print(f"    Found {len(pretoken_counts)} unique pretokens")

    # Phase 3: Vocabulary initialization
    print(" Initializing vocabulary...")
    # initialization of vocabulary
    # These are two dictionaries, which allow for fast lookups in both directions:  
    #   - id_to_token: from integer id to bytes token which represent the vocab 
    #   - token_to_id: from bytes token to integer id
    # vocab
    id_to_token: dict[int,bytes] = {}  # maps vocab id (int) to token (bytes) : {id : bytes}
    # id_to_token represent the vocab we are creating with this algorithm that will grow as we continue to merge 
    token_to_id: dict[bytes, int] = {} # maps token (bytes) to vocab id (int) : {bytes : id}


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
    print(f"   Base vocabulary size: {base_size}")

    # Phase 4: Sequence preparation
    print(" Preparing sequences...")
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

    for merge_idx in range(num_merges):
        pair_counts = Counter()  # count how often each adjacent pair occurs
        for seq, count in sequences.items():
            for j in range(len(seq) - 1):
                pair_counts[(seq[j], seq[j + 1])] += count

        if not pair_counts:
            print(f"    No more pairs to merge at iteration {merge_idx}")
            break  # nothing left to merge

        (a, b), _ = pair_counts.most_common(1)[0]

        new_bytes = id_to_token[a] + id_to_token[b]
        if new_bytes in token_to_id: # do not create a new id if the new_bytes created already exist
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
   

    

    print(f"   Final vocabulary size: {len(id_to_token)}")
    print(f"   Total merges performed: {len(merge_history)}")
    print("=" * 50)
    
    return id_to_token, merge_history



if __name__ == "__main__":
    with cProfile.Profile() as profile:
            
        print("=" * 60)
        print("  BPE TOKENIZER TRAINING WITH TIMING ANALYSIS")
        print("=" * 60)

        # init variable
        input_path_test = r"C:\Users\theo-\OneDrive\Documents\VS Code project\academic_project\NLP_stanford_lecture\assignment1\cs336_basics\dataset\testing_file.txt"
        input_path = r"C:\Users\theo-\OneDrive\Documents\VS Code project\academic_project\NLP_stanford_lecture\assignment1\cs336_basics\dataset\TinyStoriesV2-GPT4-valid.txt"
        vocab_size = 600
        special_tokens =  ["<|endoftext|>"]

        print(f" Configuration:")
        print(f"   • Test file: {input_path_test}")
        print(f"   • Main file: {input_path}")
        print(f"   • Vocabulary size: {vocab_size}")
        print(f"   • Special tokens: {special_tokens}")
        print()

        # test for pretokenize step   
        text_test = "some text that i'll pre-tokenize and by pre-tokenize, I speak about pre-tokenize and i'll be right every time<|endoftext|>" 

        # pretoken_counts= pretokenize(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",text_test,special_tokens)
        # print(pretoken_counts)
        # result
        # Special tokens: ['<|endoftext|>']
        # Escaped special pattern: <\|endoftext\|>
        # Base pattern: '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
        # Full pattern: (<\|endoftext\|>)|('(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)
        # it is working well
        # {b'some': 1, b' text': 1, b' that': 1, b' i': 2, b"'ll": 2, b' pre': 3, b'-': 3, b'tokenize': 3, b' and': 2, b' by': 1, b',': 1, b' I': 1, b' speak': 1, b' about': 1, b' be': 1, b' right': 1, b' every': 1, b' time': 1, b'<|endoftext|>': 1}

        # Training BPE tokenizer
        print(" TRAINING BPE TOKENIZER")
        print("-" * 40)
        id_to_token, merge_list = train_bpe(input_path, vocab_size, special_tokens)

        # Compression rate calculation
        print(" COMPRESSION ANALYSIS")
        print("-" * 40)

        with open(input_path, "r", encoding = "utf-8") as f:
            text = f.read()

        print(f" File size: {len(text)} characters")
        print(f" File size: {len(bytes(text, encoding='utf-8'))} bytes")

        compression_rate = get_compression_rate(id_to_token, text)
        print(f" Compression rate: {compression_rate:.2f}")
        print("=" * 60)

    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats(20)
