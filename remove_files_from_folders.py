import os

'''
Use this to delete images that are in the "remove" folder
'''

ROOT = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/labeling_round2/done_integrate/Project_6_Final"
REMOVE = "remove"
REMOVE_FILES_IN_FOLDER = os.path.join(ROOT, REMOVE)
IMAGES = os.path.join(ROOT, "images")
LABELS = os.path.join(ROOT, "labels")

# open the remove folder and add each filename to remove list
to_remove = []
for filename in os.listdir(REMOVE_FILES_IN_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        to_remove.append(filename)

all_images = os.listdir(IMAGES)
all_labels = os.listdir(LABELS)

for filename in to_remove:
    # construct full paths
    image_path = os.path.join(IMAGES, filename)
    label_filename = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(LABELS, label_filename)
    file_base_name, file_ext = os.path.splitext(filename)

    # remove image file
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Removed image: {image_path}")
    else:
        # check to see if the filename existst in the images folder
        for img in all_images:
            img_name, img_ext = os.path.splitext(img)
            # print(f"Checking image: {img_name} against {file_base_name}")
            if file_base_name in img_name or img_name in file_base_name:
                image_path = os.path.join(IMAGES, img)
                os.remove(image_path)
                print(f" ✅ Removed image: {image_path}")
                break
            print(f"Image not found, could not remove: {filename}")

    # remove label file
    if os.path.exists(label_path):
        os.remove(label_path)
        print(f"Removed label: {label_path}")
    else:
        for lbl in all_labels:
            lbl_name, lbl_ext = os.path.splitext(lbl)
            # print(f"Checking label: {lbl_name} against {label_filename.split('.')[0]}")
            if label_filename.split(".")[0] in lbl_name or lbl_name in label_filename.split(".")[0]:
                label_path = os.path.join(LABELS, lbl)
                os.remove(label_path)
                print(f" ✅ Removed label: {label_path}")
                break
            print(f"Label not found, could not remove: {filename}")