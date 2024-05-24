import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask import Flask, request, jsonify, render_template_string
import re
import logging
from functools import lru_cache

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {device}")

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

def preprocess_input(input_text):
    input_text = input_text.strip()
    return input_text

def postprocess_output(response_text):
    # Basic filtering for inappropriate content
    response_text = re.sub(r'\b(idiot|stupid|dumb|fool)\b', 'person', response_text, flags=re.IGNORECASE)
    # Split sentences and return the first meaningful sentence
    sentences = response_text.split('.')
    unique_sentences = []
    for sentence in sentences:
        if sentence.strip() and sentence.strip() not in unique_sentences:
            unique_sentences.append(sentence.strip())
    response_text = '. '.join(unique_sentences[:2])  # Return the first two unique sentences
    return response_text

@lru_cache(maxsize=100)
def rule_based_response(input_text):
    input_text = input_text.lower()
    if "what is" in input_text and ("times" in input_text or "multiplied by" in input_text):
        try:
            parts = re.split(r'\s+', input_text)
            num1 = int(parts[2])
            num2 = int(parts[4])
            return str(num1 * num2)
        except:
            return "I couldn't parse the numbers in the math question."
    if "what is" in input_text and ("plus" in input_text or "added to" in input_text):
        try:
            parts = re.split(r'\s+', input_text)
            num1 = int(parts[2])
            num2 = int(parts[4])
            return str(num1 + num2)
        except:
            return "I couldn't parse the numbers in the math question."
    if "what is" in input_text and ("minus" in input_text or "subtracted from" in input_text):
        try:
            parts = re.split(r'\s+', input_text)
            num1 = int(parts[2])
            num2 = int(parts[4])
            return str(num1 - num2)
        except:
            return "I couldn't parse the numbers in the math question."
    if "what is" in input_text and ("divided by" in input_text):
        try:
            parts = re.split(r'\s+', input_text)
            num1 = int(parts[2])
            num2 = int(parts[4])
            return str(num1 / num2)
        except:
            return "I couldn't parse the numbers in the math question."
    return None

def generate_response(input_text):
    # Preprocess the input
    input_text = preprocess_input(input_text)
    
    # Rule-based responses for specific questions
    rule_response = rule_based_response(input_text)
    if rule_response:
        return rule_response
    
    # Encode the input text
    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    # Generate the output text with adjusted parameters
    outputs = model.generate(
        inputs, 
        max_length=100,           # Increased max length for more detailed responses
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,           # Enable sampling
        temperature=0.7,          # Slightly higher temperature for balanced randomness
        top_k=50,                 # Top-K sampling
        top_p=0.9,                # Top-P (nucleus) sampling
        repetition_penalty=1.5    # Lowered repetition penalty for better response flow
    )
    
    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Postprocess the output
    response = postprocess_output(response)
    
    return response

# Create a Flask app
app = Flask(__name__)

# Route to serve the HTML file
@app.route('/')
def home():
    try:
        with open('index.html') as f:
            return render_template_string(f.read())
    except Exception as e:
        logging.error(f"Error serving home page: {e}")
        return "Error serving home page", 500

@app.route('/chat', methods=['POST'])
def chat():
    if request.is_json:
        try:
            user_input = request.json.get('message')
            logging.info(f"Received user input: {user_input}")
            response = generate_response(user_input)
            logging.info(f"Generated response: {response}")
            return jsonify({'response': response})
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return jsonify({'error': 'An error occurred processing your request'}), 500
    return jsonify({'error': 'Invalid request'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
