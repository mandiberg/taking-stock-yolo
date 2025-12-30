
import os


FOLDER = '/Volumes/OWC52/YOLO_Training_Data/decoys/images'
CLASS_ID = None # if None, will create empty labels for all classes

# open FOLDER and create empty label files for each image. 
# #put the label files in a folder called "labels_empty" inside FOLDER
for filename in os.listdir(FOLDER):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_filepath = os.path.join(FOLDER, "labels_empty", label_filename)
        os.makedirs(os.path.dirname(label_filepath), exist_ok=True)
        with open(label_filepath, 'w') as f:
            if CLASS_ID is not None:
                f.write(f"{CLASS_ID} 0.5 0.5 1.0 1.0\n")  # bbox covering entire image
            else:
                pass  # create empty file