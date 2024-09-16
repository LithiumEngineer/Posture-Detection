import tensorflow as tf
import numpy as np
from matplotlib import pyplot
import cv2
from constants import EDGES
import subprocess



def main(): 
    
    interpreter = tf.lite.Interpreter(model_path='3.tflite')
    interpreter.allocate_tensors()
    


if __name__ == "__main__":
    main()
