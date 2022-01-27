import onnxruntime
from PIL import Image
import os
import argparse
import numpy as np
import time

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model.tflite path',
                    required=True)
parser.add_argument('--image', help='Path of image to be inferenced',
                    required=True)
parser.add_argument('--img_size', help='Image size for model',
                    nargs=2, default=[224,224])

args = parser.parse_args()

MODEL_PATH = args.model
IMAGE_PATH = args.image
IMG_SIZE= args.img_size #not working xD
W=int(IMG_SIZE[0])
H=int(IMG_SIZE[1])

img= Image.open(IMAGE_PATH)
img= img.resize((W,H))
# img= np.array(img)
#img= np.expand_dims(np.array(img),axis=0)#.shape
img= np.expand_dims(np.array(img),axis=0).astype('float32')#.shape



# onnx_model = onnx.load(MODEL_PATH)
# onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(MODEL_PATH)
#---
# start_time = time.time()
start_time = time.perf_counter()#windows 10
inputs= {ort_session.get_inputs()[0].name: img}
outputs = ort_session.run(None, inputs)
# elapsed_ms = (time.time() - start_time) * 1000
elapsed_ms = (time.perf_counter() - start_time) * 1000 #windows10
print(f'Inference Time: {np.round_(elapsed_ms,2)} ms')
#---

print(outputs)
