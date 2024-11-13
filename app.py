import os
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from skimage import io, color
from skimage.util import img_as_ubyte
from sklearn.decomposition import PCA
from datetime import datetime

app = Flask(__name__)

# Set up folder for uploads and compressed images
UPLOAD_FOLDER = 'uploads'
COMPRESSED_FOLDER = 'compressed_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['COMPRESSED_FOLDER'] = COMPRESSED_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Function to check valid image extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# PCA Compression function
def compress_image(img_path, quality):
    # Read the image
    img = io.imread(img_path)
    
    # Convert to grayscale
    gray_image = color.rgb2gray(img)
    
    # Flatten the image
    flattened_img = gray_image.reshape(gray_image.shape[0], -1)
    
    # Apply PCA
    pca = PCA(n_components=quality)
    compressed_data = pca.fit_transform(flattened_img)
    
    # Reconstruct the image
    reconstructed_image = pca.inverse_transform(compressed_data)
    
    # Normalize and convert to 8-bit image
    normalized_image = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())
    compressed_img = img_as_ubyte(normalized_image)
    
    # Generate unique filename
    compressed_filename = f"compressed_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    compressed_filepath = os.path.join(COMPRESSED_FOLDER, compressed_filename)
    
    # Save the compressed image
    io.imsave(compressed_filepath, compressed_img)
    
    return compressed_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400
    
    image = request.files['image']
    accuracy = float(request.form['accuracy'])
    
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if image and allowed_file(image.filename):
        # Save the uploaded image
        filename = secure_filename(image.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(file_path)
        
        # Compress the image
        compressed_filename = compress_image(file_path, accuracy)
        
        # Return the path of the compressed image
        compressed_image_url = f"/download/{compressed_filename}"
        
        return jsonify({'compressedImagePath': compressed_image_url})

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['COMPRESSED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
