import json
import os
import glob

input_dir = 'data/file_json/'
output_dir = 'data/file_json_new/'
for file in glob.glob(input_dir + '*.json'):
    filename = os.path.basename(file)
    
    with open(file, 'r') as fin:
        label = json.load(fin)
    label['imageData'] = None

    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as fout:
        json.dump(label, fout, indent=2)
    
