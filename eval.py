from pathlib import Path
import pprint
import os
import shutil
import io
import sys

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score

def read_labels(file_path):
    print(file_path)
    f_labels = open(file_path,'r')
    labels = {}
    for line in f_labels:
        parts = line.strip().split(' ')
        file_name = parts[0]
        label = (parts[1])
        labels[file_name] = label
    f_labels.close()
    return labels

classes = ['course', 'faculty', 'project', 'student']

true_labels_path = sys.argv[1]
predicted_labels_path = sys.argv[2]

true_labels_map = read_labels(true_labels_path)
predicted_labels_map = read_labels(predicted_labels_path)

true_labels = []
predicted_labels = []

for file in true_labels_map.keys():
    true_labels.append(true_labels_map[file])
    predicted_labels.append(predicted_labels_map[file])

cm = confusion_matrix(true_labels, predicted_labels)
pprint.pprint(cm)

class_report = classification_report(true_labels, predicted_labels)
print(class_report)

macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
print(f"\nMacro F1-Score: {macro_f1:.4f}")

