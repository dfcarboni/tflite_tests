"""
2021 Eirene Solutions
Computer vision team
"""
# Import packages
import argparse
import numpy as np
import time
from PIL import Image

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model.tflite path',
                    required=True)
parser.add_argument('--image', help='Path of image to be inferenced',
                    required=True)
parser.add_argument('--img_size', help='Image size for model',
                    nargs=2, default= (224,224))
parser.add_argument('--grayscale', help='convert img to grayscale',
                    action='store_true')

args = parser.parse_args()

MODEL = args.model
IMAGE_PATH = args.image
IMG_SIZE= args.img_size
grayscale= args.grayscale

W= int(IMG_SIZE[0])
H= int(IMG_SIZE[1])
# print(IMG_SIZE)

#------------------------------------------------------------------------------
#Load TFlite Model  
import tflite_runtime.interpreter as tflite

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
model= interpreter

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Inference
if grayscale:    
    input_data = Image.open(IMAGE_PATH).convert('L') 
    input_data= input_data.resize((W,H), Image.ANTIALIAS)
    input_data= np.array(input_data, dtype=np.float32).reshape(-1,W,H,1)
else: 
    input_data = Image.open(IMAGE_PATH) 
    input_data= input_data.resize((W,H), Image.ANTIALIAS)
    input_data= np.array(input_data, dtype=np.float32).reshape(-1,W,H,3)


interpreter.set_tensor(input_details[0]['index'], input_data)

start_time = time.perf_counter()
interpreter.invoke()
elapsed_ms = (time.perf_counter() - start_time) * 1000

print('Inference Time: {:.2f} ms'.format(elapsed_ms))

output_data = interpreter.get_tensor(output_details[0]['index'])

# print("Sem daninha" if int(output_data) > 127 else "Com daninha")
print(output_data)
#---------------------------------------------------------------

#functions

#Example
#python3 tflite_detect_script.py --model weeds.tflite --image img_1597293167.46.png