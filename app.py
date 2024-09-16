import tensorflow as tf
import numpy as np
from matplotlib import pyplot
import cv2
from constants import EDGES
import subprocess


def draw_points(frame, points, conf): 
    y, x, c = frame.shape
    shaped = np.multiply(points, [y, x, 1])
    
    for point in shaped[0][0]: 
        py, px, pc = point
        if pc > conf: 
            cv2.circle(frame, (int(px), int(py)), 5, (255, 0, 0), -1)

def draw_connections(frame, edges, points, conf): 
    y, x, c = frame.shape
    shaped = np.multiply(points, [y, x, 1])
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[0][0][p1]
        y2, x2, c2 = shaped[0][0][p2]
        if (c1 > conf) and (c2 > conf): 
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

def send_notification(title, message):
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(['osascript', '-e', script])

def main(): 
    send_notification('Notification Title', 'This is the notification message.')
    interpreter = tf.lite.Interpreter(model_path='3.tflite')
    interpreter.allocate_tensors()
    video = cv2.VideoCapture(0)
    while video.isOpened(): 
        ret, frame = video.read()
        
        # Preprocessing image, reshaping into 192x192x3
        img = frame.copy()
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize_with_pad(img, 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)

        # Retrieving points
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()

        points = interpreter.get_tensor(output_details[0]['index'])
        
        # drawing keypoints
        draw_connections(frame, EDGES, points, 0.4)
        draw_points(frame, points, 0.4)
        

        cv2.imshow('Window', frame)
    
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
