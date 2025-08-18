try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    MinHash = None
    MinHashLSH = None

def minhash_candidates(query: str, docs: list, threshold=0.5) -> list:
    if MinHash is None:
        return list(range(len(docs)))
    def get_minhash(s):
        m = MinHash()
        for word in s.split():
            m.update(word.encode('utf8'))
        return m
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    doc_hashes = []
    for i, d in enumerate(docs):
        mh = get_minhash(d)
        lsh.insert(f"doc{i}", mh)
        doc_hashes.append(mh)
    query_mh = get_minhash(query)
    results = lsh.query(query_mh)
    return [int(r.replace("doc","")) for r in results]
