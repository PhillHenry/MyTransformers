from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
# Taken from Mastering Pytorch, p500



def inference(model, tokenizer):
    input_text = "I love PyTorch!"
    # Tokenize the input text using the tokenizer
    inputs = tokenizer(input_text, return_tensors="pt")
    print(f"inputs shape = {np.shape(inputs)}")

    # Perform inference using the pre-trained model
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"outputs shape = {np.shape(outputs)}")
    # Access the model predictions or outputs
    logits = outputs.logits
    print(f"logits shape = {np.shape(logits)}")
    print(f"logits = {logits}")
    predicted_class = torch.argmax(logits, dim=1).item()

    print(predicted_class)

def translate():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    input_ids = tokenizer("translate English to German: I love PyTorch.",
                          return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def play_with_bert():
    # a pre-trained model from HuggingFace
    model_name = "bert-base-uncased"
    # Load the pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    translate()
    inference(model, tokenizer)


if __name__ == "__main__":
    play_with_bert()