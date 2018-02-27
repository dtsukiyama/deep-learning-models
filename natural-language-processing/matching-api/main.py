import io
import logging

from flask import Flask, current_app, request, jsonify
from model import build

pr = build()

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    """
    Send document, receive back entity with designation
    Use: 
    curl -i -H "Content-Type: application/json" -X POST -d '{"data":"my printer does not work. Also my iPhone screen is cracked"}' http://0.0.0.0:8080/predict

    """
    
    data = {}
    try:
        data = request.json['data']
    except Exception:
        return jsonify(status_code='400', msg='Bad Request'), 400

    predictions = pr.spanMatcher(data)
    current_app.logger.info('Predictions: %s', predictions)
    return jsonify(predictions=predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

