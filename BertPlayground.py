import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline


# Taken from Mastering Pytorch, p500



def classify(model, tokenizer):
    print(model.config.id2label)
    input_text = "Shall I compare thee to a Summer's day? Thou art more fair and temperate."
    # Tokenize the input text using the tokenizer
    inputs = tokenizer(input_text, return_tensors="pt")  # inputs.attention_mask are for padding, inputs.token_type_ids for question/answer tasks
    examine_tokens(inputs, tokenizer)

    # Perform inference using the pre-trained model
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"outputs shape = {np.shape(outputs)}")
    # Access the model predictions or outputs
    logits = outputs.logits
    print(f"logits shape = {np.shape(logits)}")
    print(f"logits = {logits}")
    predicted_class = torch.argmax(logits, dim=1).item()

    print(predicted_class)  # this is garbage until we've trained fine tuned the model


def examine_tokens(inputs, tokenizer):
    print(f"inputs shape = {np.shape(inputs)}")
    print(f"inputs = {inputs}")
    word_ids = inputs.input_ids[0, 1:-1]  # CLS and SEP are always the first and last
    print("Tokens:")
    for token in tokenizer.convert_ids_to_tokens(word_ids):
        print("\t" + token)


def translate():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    input_ids = tokenizer("translate English to German: I love PyTorch.",
                          return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def next_token(model):
    unmasker = pipeline('fill-mask', model=model)
    print(unmasker("Hello I'm a [MASK] model."))


def play_with_bert():
    # a pre-trained model from HuggingFace
    model_name = "bert-base-uncased"
    next_token(model_name)
    # next_token(model_name)
    # Load the pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("model is of type ", type(model))  # transformers.models.bert.modeling_bert.BertForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classify(model, tokenizer)
    translate()


if __name__ == "__main__":
    play_with_bert()
