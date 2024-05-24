import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask import Flask, request, jsonify, render_template_string
import re

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

def preprocess_input(input_text):
    input_text = input_text.strip()
    if not input_text.endswith('?'):
        input_text += '?'
    return input_text

def postprocess_output(response_text):
    # Basic filtering for inappropriate content
    response_text = re.sub(r'\b(idiot|stupid|dumb|fool)\b', 'person', response_text, flags=re.IGNORECASE)
    # Split sentences and return the first unique sentence
    sentences = response_text.split('.')
    unique_sentences = []
    for sentence in sentences:
        if sentence.strip() and sentence.strip() not in unique_sentences:
            unique_sentences.append(sentence.strip())
    response_text = '. '.join(unique_sentences)
    return response_text

def generate_response(input_text):
    # Preprocess the input
    input_text = preprocess_input(input_text)
    
    # Encode the input text
    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    # Generate the output text with adjusted parameters
    outputs = model.generate(
        inputs, 
        max_length=150,           # Increased max length for more detailed responses
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
    return render_template_string(open('index.html').read())

@app.route('/chat', methods=['POST'])
def chat():
    if request.is_json:
        user_input = request.json.get('message')
        response = generate_response(user_input)
        return jsonify({'response': response})
    return "Invalid request", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
