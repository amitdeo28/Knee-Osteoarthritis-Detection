from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')  # Absolute path for stability
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)  # Adjust for 3 classes
    try:
        model.load_state_dict(torch.load('best_model_params.pt', map_location=torch.device('cpu')))
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    return model

model = load_model()

# Preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

CLASS_LABELS = {
    0: "No Osteoarthritis",
    1: "Moderate Osteoarthritis",
    2: "Severe Osteoarthritis"
}

# Prediction function
def predict(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return CLASS_LABELS[predicted.item()]  # Return label instead of number


# Allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for home page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file found. Please upload an image.")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No file selected. Please choose an image.")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            prediction = predict(filepath)

            # Generate image URL for Flask static serving
            image_url = url_for('static', filename=f'uploads/{filename}')

            return render_template('index.html', prediction=prediction, image_url=image_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)  # Allow external access

