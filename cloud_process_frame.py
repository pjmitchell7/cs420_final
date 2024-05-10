from flask import Flask, request, jsonify
import numpy as np
import joblib
from scipy.spatial import cKDTree
import os
import zlib
import json  # Import JSON for parsing JSON data

app = Flask(__name__)

# Load SVM models and other necessary components
model_directory = "/home/ubuntu/models"
svm_models = {name: joblib.load(os.path.join(model_directory, f'{name}_svm_model.pkl'))
              for name in ['left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'full_nose', 'mouth']}

kd_tree = joblib.load(os.path.join(model_directory, 'kd_tree.pkl'))
user_ids = joblib.load(os.path.join(model_directory, 'user_ids.pkl'))
component_weights = joblib.load(os.path.join(model_directory, 'component_weights.pkl'))
svm_confidence_threshold = 0.7

def combine_component_outputs(combined_confidences):
    total_score = sum(confidence * component_weights.get(component, 0) 
                      for component, confidence in combined_confidences.items() 
                      if confidence > svm_confidence_threshold)
    max_possible_score = sum(component_weights.values())
    return 'authorized' if total_score > 0.7 * max_possible_score else 'unauthorized'

def find_closest_profile(embeddings):
    profile_votes = {}
    for component, embedding in embeddings.items():
        _, index = kd_tree.query(np.array(embedding), k=1)
        profile_id = user_ids[index]
        profile_votes[profile_id] = profile_votes.get(profile_id, 0) + 1
    max_votes = max(profile_votes.values())
    winning_profile = max(profile_votes, key=profile_votes.get)
    return winning_profile if profile_votes[winning_profile] == max_votes else "Unrecognized"

@app.route('/process', methods=['POST'])
def process_request():
    try:
        data = zlib.decompress(request.get_data())  # Decompress the data
        embeddings = json.loads(data.decode('utf-8'))  # Decode bytes to string and parse JSON
        combined_confidences = {}
        for component, embedding in embeddings.items():
            confidence = svm_models[component].decision_function([np.array(embedding)])
            combined_confidences[component] = confidence[0]
        status = combine_component_outputs(combined_confidences)
        profile_name = find_closest_profile(embeddings) if status == 'authorized' else "Unrecognized"
        return jsonify({'status': status, 'profile_name': profile_name})
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'status': 'error', 'profile_name': 'Error processing request'}), 500
        embeddings = json.loads(data.decode('utf-8'))  # Decode bytes to string and parse JSON
        combined_confidences = {}
        for component, embedding in embeddings.items():
            confidence = svm_models[component].decision_function([np.array(embedding)])
            combined_confidences[component] = confidence[0]
        status = combine_component_outputs(combined_confidences)
        profile_name = find_closest_profile(embeddings) if status == 'authorized' else "Unrecognized"
        return jsonify({'status': status, 'profile_name': profile_name})
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'status': 'error', 'profile_name': 'Error processing request'}), 500

if __name__ == '__main__':
    from gunicorn.app.base import BaseApplication

    class Application(BaseApplication):
        def init(self, parser, opts, args):
            pass

        def load_config(self):
            self.cfg.set('bind', '0.0.0.0:5000')
            self.cfg.set('workers', 4)

        def load(self):
            return app

    Application().run()