import os
import pickle
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn

from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename

from classifier import ResNet50, ResNet101
from model_trainer import ModelTrainer
from utils.seed_everything import seed_everything
from utils.load_data import load_data
from utils.transform_image import transform_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

car_data = pd.read_csv('car_specs.csv', header=0, index_col=0)

model_path_50 = 'classification_model_50.pth'
model_path_101 = 'classification_model_101.pth'
dict_path = 'class_to_idx_2023.pkl'

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
    pred_idx = torch.argmax(torch.softmax(output, dim=1), dim=1).item()

    return pred_idx
   
with open(dict_path, 'rb') as f:
  class_to_idx = pickle.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

@app.route('/')
def index():
  return render_template('index.html', message='No file uploaded', specs='')

@app.route('/predict', methods=['POST'])
def predict():

  def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
  
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
    #image_path = url_for('uploaded_file', filename=filename)

    predicted_class = idx_to_class[predict_car_model(file_path, model_path_50)]

    try:
      car_spec = car_data.loc[predicted_class]
      spec_message = car_spec
      
    except KeyError:
      spec_message = 'No specifications available'
      

    return render_template('index.html', message=predicted_class,
                            specs=car_spec, image_path=file_path)

app.run(debug = True)