
# Facemask Detection

Detecting a person's face, then predicting whether they are wearing a mask(properly) or not, using Deep Learning and Computer Vision


## Tech Stack

**Languages:** Python, HTML, CSS

**Packages:** Pytorch(torch, torchvision), OpenCV, MediaPipe, Flask

**Model Creation and Testing:** Convolutional Neural Networks, Aritficial Neural Networks, ReLU Activation function, Binary Cross Entropy loss, Adam Optimizer, Cuda GPU Toolkit


## Installation

Download and Install python from the official website.
Go to cmd or any other terminal and run the commands below.

```bash
  py -m pip install --user virtualenv
  py -m venv FacemaskDetection
  FacemaskDetection\scripts\activate
```
Then open the FacemaskDetection directory(in the same terminal) 
cloned from Github, then run the command below.

```bash
  py -m pip install -r requirements.txt
```

## Deployment

To deploy this project run the command in the same terminal

```bash
  python server.py
```


## Demo

Demo Video : https://youtu.be/ItF9suBvv68,

The Application works perfectly locally, but, in the 'Free Tier' of Heroku app it keeps on crashing due to limited computing resources.
The above claim can be verified below, where it is deployed. You can observe that the website is loading in correctly and buttons functioning properly but the only problem is the limited computing resources available, so it keeps crashing.
Deployed on Cloud : https://facemask--detection.herokuapp.com
