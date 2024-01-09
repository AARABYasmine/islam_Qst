from flask import Flask, request, jsonify
import torch
from preprocess_data import preprocess_data  # Import your preprocessing function

app = Flask(__name__)

# Load your trained RNN model
model = torch.load('RNNModel.pth')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive data from POST request
        data = request.get_json()

        # Extract question_ids and answers from the request
        question_ids = data.get('question_ids', [])
        answers = data.get('answers', [])

        if not answers:
            return jsonify({'error': 'No answers provided'})

        # Preprocess the input data
        processed_data = [preprocess_data(answer) for answer in answers]

        # Convert processed data to tensor (if required)
        # Ensure the tensors are in the appropriate format for your model

        # Make prediction using the model
        with torch.no_grad():
            outputs = model(processed_data)
            predictions = [1 if output.item() > 0.5 else 0 for output in outputs]  # Adjust based on your output format

        # Combine question_ids with predictions
        result = {qid: pred for qid, pred in zip(question_ids, predictions)}

        # Return predictions as JSON response
        return jsonify({'predictions': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
