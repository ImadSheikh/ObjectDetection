import os
from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- Configuration ---
# Render has a temporary filesystem. This will work, but 
# remember images disappear when the server restarts.
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model. YOLO will automatically download 'yolov8m.pt' 
# to the root directory on the first run on Render.
# Change 'yolov8m.pt' (Medium) to 'yolov8n.pt' (Nano)
model = YOLO('yolov8n.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part"
        
        file = request.files['image']
        if file.filename == '':
            return "No selected file"

        if file:
            # secure_filename prevents hackers from using names like "../../system_file"
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            # Object Detection (Confidence set to 0.5 to prevent "bear/cat" errors)
            results = model.predict(source=input_path, conf=0.5)
            
            # Create the result image
            res_plotted = results[0].plot()
            
            output_filename = 'proc_' + filename
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            # Save the result
            cv2.imwrite(output_path, res_plotted)

            return render_template('index.html', 
                                   original=input_path, 
                                   processed=output_path)
    
    return render_template('index.html')

if __name__ == '__main__':
    # Render assigns a port via an environment variable. 
    # We must listen on 0.0.0.0 to be accessible externally.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)