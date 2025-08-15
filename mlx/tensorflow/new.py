#!/usr/bin/env python3
#author:rangapv@yahoo.com
#11-08-25

import coremltools as ct
import ssl

from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context

# Download class labels (from a separate file)
import urllib
import urllib.request
label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
class_labels = urllib.request.urlopen(label_url).read().splitlines()
class_labels = class_labels[1:] # remove the first class which is background
assert len(class_labels) == 1000

# make sure entries of class_labels are strings
for i, label in enumerate(class_labels):
  if isinstance(label, bytes):
    class_labels[i] = label.decode("utf8")

# Define the input type as image, 
# set pre-processing parameters to normalize the image 
# to have its values in the interval [-1,1] 
# as expected by the mobilenet model
image_input = ct.ImageType(shape=(1, 224, 224, 3,),
                           bias=[-1,-1,-1], scale=1/127)

# set class labels
classifier_config = ct.ClassifierConfig(class_labels)

#keras_model = model2.keras

#model = ct.convert('model2.keras',convert_to="mlprogram", 
#                   compute_precision=ct.precision.FLOAT32)

# Convert the model using the Unified Conversion API to an ML Program
model = ct.convert(
   'model2.keras',
    source='tensorflow',
    inputs=[image_input],
    classifier_config=classifier_config,
)

# Print a message showing the model was converted.
print('Model converted to an ML Program')


# Use PIL to load and resize the image to expected size
example_image = Image.open("./beach.jpg").resize((224, 224))

# Make a prediction using Core ML
out_dict = model.predict({"input_1": example_image})

# Print out top-1 prediction
print(out_dict["classLabel"])
