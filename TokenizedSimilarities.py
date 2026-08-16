from transformers import AutoModelForSequenceClassification, AutoTokenizer

def do_similarities():
    data = [
        "John Smith, 12 High St., London",
        "J. Smith, 12 High Street, London",
        "E. J. Thribb, 24 Acacia Avenue, London",
        "Mr Smith, 102 Van Ness, San Francisco",
        "Jane Smith, 12 High St., London",
    ]
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    vectors = map(lambda x: tokenizer(x, return_tensors="pt"), data)



if __name__ == "__main__":
    do_similarities()