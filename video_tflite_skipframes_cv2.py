# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:42:30 2022

@author: DiegoPC
"""

import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
import os
import time

#Video
path= os.getcwd()
video_name= 'input.h264'
video_path= os.path.join(path,video_name)
print(f'File path: {video_path}')
#Detection model
MODEL = os.path.join(path,'mobilenetv3','model_q_075_noconv_v2.tflite')

# Write some Text
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (80,470)
fontScale              = 3
fontColor              = (0,0,255)
thickness              = 5
lineType               = 3

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
model= interpreter

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_inference(image):
    # print(image) #should be an array
    input_data = cv2.resize(image, (224,224), interpolation= cv2.INTER_NEAREST)
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2HSV)
    input_data= np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


#----------------------------------------------------
#Open video
cap = cv2.VideoCapture(video_path)

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")


seconds = 1
fps = cap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
# multiplier = fps * seconds
multiplier= 3

#----------------------------------------------------
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  start_time = time.perf_counter()
  ret, frame = cap.read()
  count = 0
  if ret == True:
      frameId = int(round(cap.get(1)))
      ret, frame = cap.read()

      if frameId % multiplier == 0:
          frame= cv2.flip(frame, 0)
          inference_result= np.squeeze(run_inference(frame))
          print(inference_result)
          if inference_result <= 120:
              frame= cv2.putText(frame,'Detectado!', 
                  bottomLeftCornerOfText, 
                  font, 
                  fontScale,
                  fontColor,
                  thickness,
                  lineType)
          elapsed_ms = (time.perf_counter() - start_time) * 1000   
          print('Time: {:.2f} ms'.format(elapsed_ms))
          cv2.imshow('Frame',frame)
      
  # Press Q on keyboard to  exit
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  # Break the loop
  else:
      break
# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()


# start_time = time.perf_counter()
# interpreter.invoke()
# elapsed_ms = (time.perf_counter() - start_time) * 1000

# print('Inference Time: {:.2f} ms'.format(elapsed_ms))