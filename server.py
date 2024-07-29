from flask import Flask, request, render_template, Response

import image_detect
import cam_detect

app = Flask(__name__)

img_formats = ['jpg', 'JPG', 'jpeg', 'png']

def is_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in img_formats

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/image', methods=['POST'])
def image():
    file = request.files['uploaded_file']
    filepath = './data/' + file.filename
    file.save(filepath)
    try:
        if is_image(file.filename):
            return Response(image_detect.image_detector(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')
    except():
        return "<h3> File of unsupported format is uploaded, please use a valid format </h3>"

@app.route('/webcam', methods=['POST'])
def webcam():
    return Response(cam_detect.cam_detector(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
