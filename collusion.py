from .tfidf import tfidf_cosine

def collusion_pairs(docs, min_score=0.5):
    pairs = []
    for i in range(len(docs)):
        for j in range(i+1, len(docs)):
            score = tfidf_cosine(docs[i], [docs[j]])[0]
            if score >= min_score:
                pairs.append((i, j, score))
    return pairs
