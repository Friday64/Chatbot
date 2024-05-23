import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask import Flask, request, jsonify
import re

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

def generate_response(input_text):
    # Encode the input text
    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    # Generate the output text with adjusted parameters
    outputs = model.generate(
        inputs, 
        max_length=100, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,    # Enable sampling
        temperature=0.6,   # Lower temperature to reduce randomness
        top_k=40,          # Top-K sampling
        top_p=0.85,        # Top-P (nucleus) sampling
        repetition_penalty=2.5  # Penalize repetition
    )
    
    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Basic filtering for inappropriate content
    response = re.sub(r'\b(idiot|stupid|dumb|fool)\b', 'person', response, flags=re.IGNORECASE)
    response = response.split('\n')[0]  # Take only the first sentence
    
    return response

# Create a Flask app
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    if request.is_json:
        user_input = request.json.get('message')
        response = generate_response(user_input)
        return jsonify({'response': response})
    return "Invalid request", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
