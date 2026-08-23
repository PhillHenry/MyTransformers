import torch
from transformers import AutoModel, AutoTokenizer

jane = "Jane Smith, 12 High St., London"
janet = "Janet Smith, 14 High St., London"

def embed(model, tokenizer, records: [str]):
    """One vector per string: the mean of its token embeddings, ignoring padding."""
    inputs = tokenizer(records, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    token_vectors = outputs.last_hidden_state         # [batch, tokens, hidden]
    mask = inputs.attention_mask.unsqueeze(-1)        # [batch, tokens, 1]
    summed = (token_vectors * mask).sum(dim=1)
    return summed / mask.sum(dim=1).clamp(min=1)      # [batch, hidden]


def cosine_similarities(vectors):
    """Pair-wise cosine similarity of every row against every other row."""
    normalised = torch.nn.functional.normalize(vectors, p=2, dim=1)
    return normalised @ normalised.T


def print_similarities(data, similarities):
    """Every pair once, most similar first."""
    pairs = [(similarities[i, j].item(), i, j)
             for i in range(len(data))
             for j in range(i + 1, len(data))]
    for score, i, j in sorted(pairs, reverse=True):
        print(f"{score:.4f}\t{data[i]!r}\n\tvs\t{data[j]!r}")


def describe_mebeddings(tokenizer, records: [str]):
    inputs = tokenizer(records, return_tensors="pt", padding=True, truncation=True)
    for record, encoding in zip(records, inputs.encodings):
        print(f"{record} -> {encoding.tokens}")


def do_similarities(data: [str]):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    vectors = embed(model, tokenizer, data)
    print(f"vectors shape = {tuple(vectors.shape)}")

    similarities = cosine_similarities(vectors)
    print(f"similarities shape = {tuple(similarities.shape)}")
    print(similarities)
    print_similarities(data, similarities)
    return tokenizer


if __name__ == "__main__":
    tokenizer = do_similarities(
        [
            "John Smith, 12 High St., London",
            "J. Smith, 12 High Street, London",
            "E. J. Thribb, 24 Acacia Avenue, London",
            "Mr Smith, 102 Van Ness, San Francisco",
            jane,
            janet,  # this confuses the Jane Smith record a little
        ]
    )
    print(describe_mebeddings(tokenizer, [jane, janet]))
    do_similarities(
        [
            "Phil Henry, 30 Acacia Avenue, London",
            "Phillip Henry, 30 Acacia Ave., London",
            jane,
            janet,  # this confuses the Jane Smith record a little
        ]
    )