import os
import pickle
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn

from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename

from classifier import ResNet50
from model_trainer import ModelTrainer
from utils.seed_everything import seed_everything
from utils.load_data import load_data
from utils.transform_image import transform_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dir = os.getcwd()
car_data = pd.read_csv('./car_specs.csv', header=0, index_col=0)

model_path_50 = './classification_model_50.pth'
model_path_101 = './classification_model_101.pth'
dict_path = './class_to_idx_2023.pkl'

with open(dict_path, 'rb') as f:
    class_to_idx = pickle.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_car_model(image_path, model_path):
   
   map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   checkpoint = torch.load(model_path, map_location)
   model = ResNet50(hidden_1=1024, hidden_2=512, num_target_classes=175).to(device)
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()

   image = Image.open(image_path).convert('RGB')
   transform = transform_image().to(device)
   image = transform(image).to(device)
   image = image.unsqueeze(0).to(device)

   with torch.inference_mode():
       output = model(image)
       probabilities, predicted_idxs = torch.topk(softmax(output, dim=1), 6)
       
       return probabilities, predicted_idxs

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(dir,'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

@app.route('/')
def index():
  return render_template('index.html', message='No file uploaded', specs='Upload an image to see the car specifications')

@app.route('/img/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
  
  image_path = None
  if 'file' not in request.files:
        return render_template('index.html', message='No file part', specs='')

  file = request.files['file']

  if file.filename == '':
     return render_template('index.html', message='No selected file',specs='')

  if file and allowed_file(file.filename):
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    image_path = url_for('uploaded_file', filename=filename)

    probabilities, predicted_idxs = predict_car_model(file_path, model_path)
    best_match = idx_to_class[predicted_idxs[0][0].item()]

    top_result = f"{idx_to_class[predicted_idxs[0][0].item()]}: {probabilities[0][0].item():.1%} confidence"
    other_results = "<br>".join([f"{idx_to_class[label.item()]}: {prob.item():.1%} confidence"
                                 for prob, label in zip(probabilities[0][1:], predicted_idxs[0][1:])])

    message = "<br>".join(['',top_result, '','Other Top Results: ',other_results])
    
    try:
      car_spec = car_data.loc[idx_to_class[predicted_idxs[0][0].item()]]
      spec_message = car_spec
      
    except KeyError:
      spec_message = 'No specifications available'
      

    return render_template('index.html', message=message,
                            specs=car_message, image_path=image_path)

app.run(debug = True)
