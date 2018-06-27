import os
import json
from train_unet import convert_labels
import matplotlib.pyplot as plt

data_path = os.path.join(os.getcwd(), '..', 'data', 'raw', 'labels')

label_1 = os.listdir(data_path)[0]
print(label_1)

with open(os.path.join(data_path, label_1)) as f:
    test = convert_labels(f)

test = test.reshape(plt.imread(img).shape[:2])
