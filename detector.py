import argparse
from plagiarism import preprocess, tfidf, minhash_lsh, explain, report

def detect_plagiarism(text, reference_documents, topk, min_score):
    norm_text = preprocess.normalize(text)
    norm_refs = [preprocess.normalize(r) for r in reference_documents]
    candidates = minhash_lsh.minhash_candidates(norm_text, norm_refs)
    scored = []
    sims = tfidf.tfidf_cosine(norm_text, [norm_refs[i] for i in candidates])
    for idx, score in zip(candidates, sims):
        if score >= min_score:
            highlighted, overlap = explain.highlight_overlap(
                preprocess.tokenize(norm_text),
                preprocess.tokenize(norm_refs[idx])
            )
            scored.append({
                "ref_index": idx,
                "score": float(score),
                "overlap": overlap,
                "highlighted": highlighted
            })
    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored[:topk]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--refs", nargs='+', required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min_score", type=float, default=0.3)
    ap.add_argument("--report", default="report.html")
    args = ap.parse_args()

    text = open(args.query, encoding="utf-8").read()
    refs = [open(r, encoding="utf-8").read() for r in args.refs]

    raw = detect_plagiarism(text, refs, args.topk, args.min_score)

    results = {
        "doc_scores": {r["ref_index"]: r["score"] for r in raw},
        "matches": [
            {
                "doc": r["ref_index"],
                "para": None,  
                "score": r["score"],
                "query_text": text,
                "ref_text": refs[r["ref_index"]],
            }
            for r in raw
        ],
    }

    report.save_html_report(results, refs, args.report)
    print(f"✅ Report generated: {args.report}")
