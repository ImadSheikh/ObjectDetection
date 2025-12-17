import os
from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Setup folders
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the 'Medium' model for better accuracy than the 'Nano' model
# The first time you run this, it will download the weight file (~50MB)
model = YOLO('yolov8m.pt') 

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was actually uploaded
        if 'image' not in request.files:
            return "No file part"
        
        file = request.files['image']
        
        if file.filename == '':
            return "No selected file"

        if file:
            # Save the uploaded file
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(input_path)

            # --- Image Processing / Object Detection ---
            # conf=0.5 means: only show objects the AI is 50% sure about
            results = model.predict(source=input_path, conf=0.5)
            
            # Draw the boxes and labels on the image
            res_plotted = results[0].plot()
            
            # Save the processed image
            output_filename = 'processed_' + file.filename
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, res_plotted)

            # Pass the paths to the HTML template
            return render_template('index.html', 
                                   original=input_path, 
                                   processed=output_path)
    
    return render_template('index.html')

if __name__ == '__main__':
    # Start the Flask server
    app.run(debug=True)