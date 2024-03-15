import tensorflow_hub as hub
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Apply image detector on a batch of image.
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
width = 1028
height = 1028

# Load image using OpenCV
img = cv2.imread('Image.jpg.jpeg')

# Resize the image to match the input shape
inp = cv2.resize(img, (width , height ))

# Convert the image to RGB
rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

# Convert the RGB image to a TensorFlow tensor
rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

# Add a batch dimension to the tensor
rgb_tensor = tf.expand_dims(rgb_tensor , 0)

# Perform object detection on the image
boxes, scores, classes, num_detections = detector(rgb_tensor)

# Load object labels from a CSV file
labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

# Extract predicted labels, boxes, and scores
pred_labels = classes.numpy().astype('int')[0]
pred_labels = [labels[i] for i in pred_labels]
pred_boxes = boxes.numpy()[0].astype('int')
pred_scores = scores.numpy()[0]

# Draw bounding boxes and labels on the image
for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
    if score < 0.5:
        continue
        
    score_txt = f'{100 * round(score)}%'
    rgb = cv2.rectangle(rgb,(xmin, ymax),(xmax, ymin),(0,255,0),2)      
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(rgb, label, (xmin, ymax-10), font, 1.5, (255,0,0), 2, cv2.LINE_AA)
    cv2.putText(rgb, score_txt, (xmax, ymax-10), font, 1.5, (255,0,0), 2, cv2.LINE_AA)

# Plot the image with bounding boxes and labels
plt.figure(figsize=(10,10))
plt.imshow(rgb)

# Save the plotted image
plt.savefig('image_pred.jpg', transparent=True)

# Show the plot
plt.show()
