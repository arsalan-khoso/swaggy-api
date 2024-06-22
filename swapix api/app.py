import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from io import BytesIO
from PIL import Image
import insightface
from insightface.app import FaceAnalysis

app = Flask(__name__)

# Initialize the FaceAnalysis app
app_analysis = FaceAnalysis(name='buffalo_l')
app_analysis.prepare(ctx_id=0, det_size=(640, 640))

# Load the inswapper model
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=False)

@app.route('/face_swap', methods=['POST'])
def face_swap():
    if 'source' not in request.files or 'target' not in request.files:
        return jsonify({'error': 'Please provide both source and target images.'}), 400

    source_file = request.files['source']
    target_file = request.files['target']

    # Read the images
    # source_img = np.array(Image.open(source_file))
    # target_img = np.array(Image.open(target_file))
    
    # Read the images
    source_img = np.array(Image.open(source_file))
    target_img = np.array(Image.open(target_file))

    # Convert images from RGB to BGR (for OpenCV)
    # source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
    # target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)

    # Detect faces in the images
    source_faces = app_analysis.get(source_img)
    target_faces = app_analysis.get(target_img)

    if not source_faces or not target_faces:
        return jsonify({'error': 'No faces detected in one or both images.'}), 400

    # Assume we are swapping the first detected face
    source_face = source_faces[0]
    target_face = target_faces[0]

    # Perform face swap
    result_img = swapper.get(source_img, source_face, target_face, paste_back=True)

    # Convert result image from BGR to RGB
    # result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image for sending response
    result_pil = Image.fromarray(result_img)
    buf = BytesIO()
    result_pil.save(buf, format='png', quality=1000)
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    if 'image' not in request.files:
        return jsonify({'error': 'Please provide an image.'}), 400

    image_file = request.files['image']

    # Read the image
    image = np.array(Image.open(image_file))

    # Detect faces in the image
    faces = app_analysis.get(image)

    if not faces:
        return jsonify({'error': 'No faces detected in the image.'}), 400

    # Prepare response with bounding boxes
    face_data = []
    for face in faces:
        bbox = face.bbox.astype(int).tolist()
        face_data.append(bbox)

    return jsonify({'faces': face_data})


@app.route('/multi_face_swap', methods=['POST'])
def multi_face_swap():
    if 'source' not in request.files or 'target' not in request.files:
        return jsonify({'error': 'Please provide both source and target images.'}), 400

    if 'index' not in request.form:
        return jsonify({'error': 'Please provide index.'}), 400

    index = int(request.form['index'])


    source_file = request.files['source']
    target_file = request.files['target']

    # Read the images
    source_img = np.array(Image.open(source_file))
    target_img = np.array(Image.open(target_file))

    # Detect faces in the images
    source_faces = app_analysis.get(source_img)
    target_faces = app_analysis.get(target_img)

    if not source_faces or not target_faces:
        return jsonify({'error': 'No faces detected in one or both images.'}), 400

    if index >= len(source_faces)  :
        return jsonify({'error': 'Invalid source_index or target_index.'}), 400

    # Get the specified faces
    source_face = source_faces[index]
    target_face = target_faces[0]

    # Perform face swap
    result_img = swapper.get(source_img, source_face, target_face, paste_back=True)

    # Convert to PIL Image for sending response
    result_pil = Image.fromarray(result_img)
    buf = BytesIO()
    result_pil.save(buf, format='png', quality=100)
    buf.seek(0)

    return send_file(buf, mimetype='image/png')


if __name__ == '__main__':
    app.run()
