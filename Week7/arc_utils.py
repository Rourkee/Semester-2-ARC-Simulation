##### arc_utils.py

import json
import numpy as np

def load_arc_task(json_path):
    with open(json_path, 'r') as f:
        task = json.load(f)
    
    train_inputs = [np.array(pair['input']) for pair in task.get('train', [])]
    train_outputs = [np.array(pair['output']) for pair in task.get('train', [])]
    test_inputs = [np.array(pair['input']) for pair in task.get('test', [])]
    test_outputs = [np.array(pair['output']) for pair in task.get('test', [])]  # might be empty
    
    return train_inputs, train_outputs, test_inputs, test_outputs
