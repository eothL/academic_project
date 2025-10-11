def get_compression_ratio(string: str, indices: list[int] ) -> float:
    """Given string that has been tokenized into indices """
    num_bytes = len(bytes(string, encoding= "utf-8"))
    num_tokens = len(indices)
    return num_bytes/num_tokens
    
     








