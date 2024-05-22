from flask import Flask,render_template, request, redirect, url_for, jsonify
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import XLMRobertaModel
import numpy as np
from summarizer import summarize_text

app = Flask(__name__)

#--------------------------------------------------------------File Paths----------------------------------------------------------------------------
MODEL_PATH = os.path.join('virtual_env', 'model.pt')
#-----------------------------------------------------------------------------------------------------------------------------------------------
class XLMRobertaClass(torch.nn.Module):
    def __init__(self):
        super(XLMRobertaClass, self).__init__()
        self.l1 = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    

def initialize_model():
    device = torch.device('cpu')  # Use CPU for inference

    if os.path.exists(MODEL_PATH):
        model = torch.load(MODEL_PATH, map_location=device)  # Load the entire model object
        model.eval()
        print('Initialized')
        return model
    else:
        raise FileNotFoundError("Model file not found.")
    
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------JSON Files--------------------------------------------------------------------
# Load JSON data from the file
scraped_data_path = os.path.join('virtual_env', 'scraped_data.json')
refined_scraped_data_path = os.path.join('virtual_env', 'refined_scraped_data.json')


def get_refine_data(scraped_data_path, refined_scraped_data_path):
    with open(scraped_data_path, 'r') as f:
        data = json.load(f)

    # Extract the "body" field from each object, skipping entries without "body" field
    bodies = []
    for entry in data:
        if entry.get("dataType") == "post":
            continue
        bodies.append(entry['body'])

    # Save the extracted bodies into another JSON file
    with open(refined_scraped_data_path, 'w') as f:
        json.dump(bodies, f, indent=4)
    print('done.')

def load_sentences_from_json(json_file):
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    else:
        return []

#------------------------------------------------------------------------------------------------------------------------------------------
# Detect emotion for each sentence
def detect_emotion_for_sentences(sentences, model):
    emotions = []
    for sentence in sentences:
        emotion = detect_emotion_with_model(sentence, model)
        emotions.append(emotion)
    return emotions

# Detect emotion for a single sentence
def detect_emotion_with_model(sentence, model):
    model_name = 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    type_rep = {0: 'Neutral', 1: 'Positive', 2: 'Negative', 3: 'Irrelevant'}
    logits = output[0]  # Accessing logits directly from the output tensor
    predicted_label = type_rep[int(torch.argmax(logits))] 
    return predicted_label

# Count emotions
def count_emotions(emotions):
    counts = {'Neutral': 0, 'Positive': 0, 'Negative': 0, 'Irrelevant': 0}
    for emotion in emotions:
        counts[emotion] += 1
    return counts

# Route to get emotion counts
@app.route('/emotion-counts')
def emotion_counts():
    get_refine_data(scraped_data_path, refined_scraped_data_path)
    sentences = load_sentences_from_json(refined_scraped_data_path )
    if not sentences:
        return jsonify({'error': 'No sentences found in the JSON file'})
    model = initialize_model()
    emotions = detect_emotion_for_sentences(sentences, model)
    counts = count_emotions(emotions)
    return counts


def get_text(path):

    with open(path, 'r') as f:
        data = json.load(f)

    # Extract the "body" field from each object, skipping entries without "body" field
    for entry in data:
        if entry.get("dataType") == "post":  
            text = entry['body']
        else:
            break
    return text

def get_title():
    with open(scraped_data_path, 'r') as f:
        data = json.load(f)

    # Extract the "body" field from each object, skipping entries without "body" field
    for entry in data:
        if entry.get("dataType") == "post":  
            title = entry['title']
            return title 

def get_userId():
    with open(scraped_data_path, 'r') as f:
        data = json.load(f)

    # Extract the "body" field from each object, skipping entries without "body" field
    for entry in data:
        if entry.get("dataType") == "post":  
            userId = entry['userId']
            return userId
        
def get_upvote():
    with open(scraped_data_path, 'r') as f:
        data = json.load(f)

    # Extract the "body" field from each object, skipping entries without "body" field
    for entry in data:
        if entry.get("dataType") == "post":  
            upvotes = entry['upVotes']
            return upvotes

def get_createdAt():
    with open(scraped_data_path, 'r') as f:
        data = json.load(f)

    # Extract the "body" field from each object, skipping entries without "body" field
    for entry in data:
        if entry.get("dataType") == "post":  
            createdAt = entry['createdAt']
            return createdAt
        
def get_full_article():
    with open(scraped_data_path, 'r') as f:
        data = json.load(f)

    # Extract the "body" field from each object, skipping entries without "body" field
    for entry in data:
        if entry.get("dataType") == "post":  
            full_article = entry['body']
            return full_article


    

@app.route('/summarize', methods=['GET','POST'])
def summarize():

    content =get_text(scraped_data_path)
    # Call function to summarize text
    try:
        summary = summarize_text(content)
        print(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Internal server error

    # Render HTML template with the summary and button
    return summary


if __name__ == '__main__':
    app.run(debug=True)
