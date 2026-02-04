'''
this script opens a folder containing outputs from the yolo model
each folder will be labeled with the class id
each file in the folder will have a name that starts with the confidence score
eg: 0.99_48545556_YOLO_debug.jpg
and creates a histogram plot of the confidence scores for each class
and saves it as `class_conf_histogram.png` in the dataset root.
It uses matplotlib to create the histogram.
'''
import argparse
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import re
from pathlib import Path

FOLDER = "/Volumes/OWC52/segment_testing_flowers/test_output"


def plot_histogram_class_conf(output_folder):
    class_confidences = defaultdict(list)

    # Iterate over class directories
    for class_dir in os.listdir(output_folder):
        class_path = os.path.join(output_folder, class_dir)
        if os.path.isdir(class_path):
            # Iterate over files in class directory
            for file_name in os.listdir(class_path):
                match = re.match(r'([0-9.]+)_', file_name)
                if match:
                    confidence = float(match.group(1))
                    class_confidences[class_dir].append(confidence)

    # Plot histograms
    plt.figure(figsize=(12, 8))
    for class_id, confidences in class_confidences.items():
        plt.hist(confidences, bins=20, alpha=0.5, label=f'Class {class_id}')


    plt.title('Histogram of Class Confidence Scores')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'class_conf_histogram.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot histogram of class confidence scores from YOLO outputs.')
    parser.add_argument('--output_folder', type=str, default=FOLDER,
                        help='Path to the folder containing YOLO output class directories.')
    args = parser.parse_args()

    plot_histogram_class_conf(args.output_folder)