from transformers import BartTokenizer, BartForConditionalGeneration
import re

# Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def summarize_text(text):
    # Tokenize input text
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True, padding=True)

    # Perform summarization
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=1500,        # Adjust maximum length of generated summary
        num_beams=4,           # Adjust the number of beams for beam search
        length_penalty=2.0,    # Adjust the length penalty to encourage longer summaries
        early_stopping=True,   # Allow early stopping to generate shorter summaries
        temperature=0.9,       # Adjust temperature for sampling if needed
        top_k=50,              # Adjust top-k sampling if needed
        top_p=0.95             # Adjust top-p sampling if needed
    )

    # Decode the generated summary
    summarized_output = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
   
    return summarized_output
