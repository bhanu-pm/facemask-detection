import torch
import cv2
from PIL import Image

import utils

model = utils.CNN()
model.load_state_dict(torch.load('facemask_ext.pt'))
model.to(utils.device)

# Prediction of detected face
def predict_on_face(frame, xcor, ycor, width, height):
    image = frame[xcor:xcor + width, ycor:ycor + height]
    image = cv2.resize(image, (32, 32))
    img = Image.fromarray(image)

    img = utils.transformations(img)
    img = img.view(1, 3, 32, 32)  # View in tensor

    with torch.no_grad():
        model.eval()  # Set eval mode
        img = img.to(utils.device)
        output = model(img)
        predicted = torch.argmax(output, 1)
        predicted = predicted.to('cpu')
        result = utils.classifier(predicted)
        return result
