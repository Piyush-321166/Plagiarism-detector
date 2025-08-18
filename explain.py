def highlight_overlap(query_tokens, doc_tokens, n=3):
    query_ngrams = set(tuple(query_tokens[i:i+n]) for i in range(len(query_tokens)-n+1))
    doc_ngrams = set(tuple(doc_tokens[i:i+n]) for i in range(len(doc_tokens)-n+1))
    overlap = query_ngrams & doc_ngrams
    doc_str = " ".join(doc_tokens)
    for ng in overlap:
        phrase = " ".join(ng)
        doc_str = doc_str.replace(phrase, f"<mark>{phrase}</mark>")
    return doc_str, len(overlap)
