import os
import json

folder = '/Users/michaelmandiberg/Library/CloudStorage/Dropbox/YOLO_Training_Data/new_images/money_mix/labels'

for filename in os.listdir(folder):
    # rename only .txt files
    if filename.endswith('.txt'):
        print(f'Processing file: {filename}')
        # split the filename on "-" and take the second part
        parts = filename.split('-', 1)
        if len(parts) == 2:
            new_filename = parts[1]
            src = os.path.join(folder, filename)
            dst = os.path.join(folder, new_filename)
            os.rename(src, dst)
            print(f'Renamed: {filename} -> {new_filename}')