# libraries
import regex as re
from collections import Counter, defaultdict
import cProfile
import pstats
from pretokenization_example import find_chunk_boundaries 
git

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

    # re.compile the pattern once and saves it as a pattern object, so it can be reusable
    pattern = re.compile(full_pattern)
    counts: Counter[bytes] = Counter()
    for match in pattern.finditer(text):
        counts[match.group(0).encode("utf-8")] += 1
    return dict(counts)


def _increment_pair(
    pair_counts: Counter[tuple[int, int]],
    pair_locs: dict[tuple[int, int], set[tuple[int, int]]],
    pair: tuple[int, int],
    weight: int,
    occurrence: tuple[int, int]
) -> None:
    pair_counts[pair] += weight
    pair_locs[pair].add(occurrence)


def _decrement_pair(
    pair_counts: Counter[tuple[int, int]],
    pair_locs: dict[tuple[int, int], set[tuple[int, int]]],
    pair: tuple[int, int],
    weight: int,
    occurrence: tuple[int, int] | None = None
) -> None:
    if pair not in pair_counts:
        return
    pair_counts[pair] -= weight
    if pair_counts[pair] <= 0:
        pair_counts.pop(pair, None)
        pair_locs.pop(pair, None)
        return
    if occurrence is not None:
        locs = pair_locs.get(pair)
        if locs:
            locs.discard(occurrence)
            if not locs:
                pair_locs.pop(pair, None)


def update_occurrences(
    pair: tuple[int, int],
    new_id: int,
    corpus: list[list[int]],
    weights: list[int],
    pair_counts: Counter[tuple[int,int]],
    pair_locs: dict[tuple[int,int], set[tuple[int,int]]]
    ) -> None:

    a, b = pair
    occurrences = pair_locs.pop(pair, set())
    for seq_idx, pos in sorted(occurrences, key=lambda item: item[1], reverse=True):
        seq = corpus[seq_idx]

        if pos >= len(seq) - 1 or seq[pos] != a or seq[pos + 1] != b:
            continue

        weight = weights[seq_idx]
        left = seq[pos - 1] if pos > 0 else None
        right = seq[pos + 2] if pos + 2 < len(seq) else None

        _decrement_pair(pair_counts, pair_locs, pair, weight)
        if left is not None:
            _decrement_pair(pair_counts, pair_locs, (left, a), weight, (seq_idx, pos - 1))
        if right is not None:
            _decrement_pair(pair_counts, pair_locs, (b, right), weight, (seq_idx, pos + 1))

        seq[pos] = new_id
        del seq[pos + 1]

        if left is not None:
            _increment_pair(pair_counts, pair_locs, (left, new_id), weight, (seq_idx, pos - 1))
        if right is not None:
            _increment_pair(pair_counts, pair_locs, (new_id, right), weight, (seq_idx, pos))

    pair_counts.pop(pair, None)


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
    # corpus split into character or words or smaller units based on the regex pattern chosen and its frequencies
    corpus_counts = pretokenize(gpt2_regex, data, special_tokens) # return dict(byte:count)

    print(f"    Found {len(corpus_counts)} unique pretokens")

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
    # to look at the approach : look at the v0 version
    # --------------different approach to accelerate pair_counting : by updating only changed pair---------------------------------------
    # we are now using two list of mutable token sequences instead of dict mapping tupple[int, ...] -> count 
    # build a list of mutable token sequences 
    # one per occurence in the corpus and to avoid duplicating a string n times for counting, we are adding a weight link to the occurence

    # initialization of the corpus and its counting with the pretokenize text
    tokenized_corpus: list[list[int]] = []
    corpus_weights: list[int] = [] # associate each tokenized word/subword to its frequencies in the tokenized corpus 

    # each list in corpus is still a chunck of the original text, just expressed as ids instead of raw characters
    
    for token_bytes, count in corpus_counts.items():
        if token_bytes in token_to_id:
            # handle special tokens
            seq = [token_to_id[token_bytes]]
        else:
            seq = [token_to_id[bytes([b])] for b in token_bytes]
        tokenized_corpus.append(seq)
        corpus_weights.append(count)

    # Phase 5: BPE Merging
    print(f" Starting BPE merging ({num_merges} merges)...")
    # pair counting
    pair_counts : Counter[tuple[int,int]] = Counter()
    pair_locs : dict[tuple[int,int],set[tuple[int,int]]] = defaultdict(set)


    # starting counting 
    for seq_idx, seq in enumerate(tokenized_corpus):
            w = corpus_weights[seq_idx]
            for pos in range(len(seq) - 1):
                pair = (seq[pos], seq[pos + 1])
                pair_counts[pair] += w
                pair_locs[pair].add((seq_idx, pos))


    # pair_locs[(x, y)] = [(seq_idx, pos), ...]                                                                                             
    for _ in range (num_merges):
        if not pair_counts: # no bigram left anywhere
            break

        (a, b), _ = pair_counts.most_common(1)[0]
        new_bytes = id_to_token[a] + id_to_token[b]

        # add the new merged bytes to the sequences and create new id if it is not already the case
        if new_bytes in token_to_id:
            new_id = token_to_id[new_bytes]
        else:
            new_id = len(token_to_id)
            merge_history.append((id_to_token[a], id_to_token[b]))
            id_to_token[new_id], token_to_id[new_bytes] = new_bytes, new_id
        
        update_occurrences(pair=(a,b), new_id=new_id, corpus=tokenized_corpus,
                            weights=corpus_weights, pair_counts=pair_counts,                                                                     
                            pair_locs=pair_locs)
        

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
        input_test = r"C:\Users\theo-\OneDrive\Documents\VS Code project\academic_project\NLP_stanford_lecture\assignment1\cs336_basics\dataset\test.txt"
        vocab_size = 42250000*4
        special_tokens =  ["<|endoftext|>"]

        # test for pretokenize step   
        # text_test = "some text that i'll pre-tokenize and by pre-tokenize, I speak about pre-tokenize and i'll be right every time<|endoftext|>" 

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
        # id_test, merge_test = train_bpe(input_test,vocab_size= 300, special_tokens=special_tokens)
        # id_test_2, merge_test_2 = train_bpe(input_path_test, vocab_size= 3000, special_tokens=special_tokens)
        print(f" Configuration:")
        print(f"   • Test file: {input_path_test}")
        print(f"   • Main file: {input_path}")
        print(f"   • Vocabulary size: {vocab_size}")
        print(f"   • Special tokens: {special_tokens}\n")

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
