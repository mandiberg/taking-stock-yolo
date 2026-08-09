
from ultralytics import YOLOE
# import supervision as sv
import cv2
import numpy as np

FOLDER = "/Users/michaelmandiberg/Documents/YOLO_Training_Data/yoga_mats_for_testing" 
LABELS = "labels"
IMAGES = "images"
model = YOLOE("yoloe-26x-seg.pt")           # newest YOLO26 backbone
 
# Multi-class long-tail prompts on a busy NYC street
NAMES = ["yoga mat"]
model.set_classes(NAMES)
iou_threshold = 0.7
SCORE_CUTOFF = .7

# load all images from the folder
import os
image_files = [os.path.join(FOLDER, IMAGES, f) for f in os.listdir(os.path.join(FOLDER, IMAGES)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]    


def compute_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2] (top-left, bottom-right)
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    # Calculate area of the overlap (if any)
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate area of both boxes
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Calculate union area by removing double-counted overlap
    union_area = area_a + area_b - intersection_area

    # Prevent division by zero
    if union_area == 0:
        return 0.0

    return intersection_area / union_area

for image_file in image_files:
    image = cv2.imread(image_file)
    results = model.predict(image, conf=0.2, verbose=False)
    print(f"Detection results for {image_file}: {len(results)}")
    if len(results) > 1:
        # check to see if the bboxes overlap, using IOU
        # if they do, pick the one with the highest confidence score

        for i in range(len(results[0].boxes)):
            boxA = results[0].boxes.xyxy[i].cpu().numpy()
            scoreA = results[0].boxes.conf[i].cpu().numpy()
            for j in range(i + 1, len(results[0].boxes)):
                boxB = results[0].boxes.xyxy[j].cpu().numpy()
                scoreB = results[0].boxes.conf[j].cpu().numpy()
                iou = compute_iou(boxA, boxB)
                print(f"Comparing box {i} and box {j}: IoU = {iou:.2f}, scoreA = {scoreA:.2f}, scoreB = {scoreB:.2f}")
                if iou > iou_threshold:
                    # If boxes overlap, keep the one with the higher confidence score
                    if scoreA >= scoreB:
                        results[0].boxes.xyxy[j] = torch.tensor([0, 0, 0, 0])  # effectively remove boxB
                    else:
                        results[0].boxes.xyxy[i] = torch.tensor([0, 0, 0, 0])  # effectively remove boxA


        # # compare each pair of boxes
        # boxes = results[0].boxes.xyxy.cpu().numpy()  # get the bounding boxes in xyxy format
        # scores = results[0].boxes.conf.cpu().numpy()


        # Example usage
        box_gt = [0, 0, 10, 10]  # Ground truth
        box_pred = [2, 2, 12, 12] # Prediction
        iou = compute_iou(box_gt, box_pred)
        print(f"IoU: {iou:.2f}")


    # save the bbox as a YOLO format text file
    # YOLO is center and percentage based, so we need to convert the bbox coordinates
    # to YOLO format (x_center, y_center, width, height) in percentage
    for result in results:
        boxes_for_labels = []
        boxes = result.boxes.xyxy.cpu().numpy()  # get the bounding boxes in xyxy format
        # remove any boxes with confidence score less than 0.2
        # scores = result.boxes.conf.cpu().numpy()
        # for i in range(len(boxes)):
        #     if scores[i] < SCORE_CUTOFF:
        #         boxes[i] = np.array([0, 0, 0, 0])  # effectively remove the box

        # if len(boxes) > 1:
        #     for i in range(len(boxes)):
        #         boxA = boxes[i]
        #         scoreA = result.boxes.conf[i].cpu().numpy()
        #         for j in range(i + 1, len(boxes)):
        #             boxB = boxes[j]
        #             scoreB = result.boxes.conf[j].cpu().numpy()
        #             iou = compute_iou(boxA, boxB)
        #             print(f"Comparing box {i} and box {j}: IoU = {iou:.2f}, scoreA = {scoreA:.2f}, scoreB = {scoreB:.2f}")
        #             if iou > iou_threshold:
        #                 # If boxes overlap, keep the one with the higher confidence score
        #                 if scoreA >= scoreB:
        #                     boxes[j] = np.array([0, 0, 0, 0])  # effectively remove boxB
        #                 else:
        #                     boxes[i] = np.array([0, 0, 0, 0])  # effectively remove boxA
        # boxes = [box for box in boxes if not np.array_equal(box, [0, 0, 0, 0])]  # remove boxes that were set to [0, 0, 0, 0]
        # print(f"Final boxes for {image_file}: {boxes}")
        for box in boxes:
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2 / image.shape[1]
            y_center = (y1 + y2) / 2 / image.shape[0]
            width = (x2 - x1) / image.shape[1]
            height = (y2 - y1) / image.shape[0]
            class_id = 151  # since we have only one class "yoga mat"
            boxes_for_labels.append((class_id, x_center, y_center, width, height))
    # only take unique boxes, in case the model detected the same object multiple times
    boxes_for_labels = list(set(boxes_for_labels))
    print(f"Detected {len(boxes_for_labels)} unique yoga mats in {image_file}")

            
    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(FOLDER, LABELS, os.path.basename(label_file))
    with open(label_path, "w") as f:
        for class_id, x_center, y_center, width, height in boxes_for_labels:
            print(f"Writing to label file: {label_path}, class_id: {class_id}, x_center: {x_center:.6f}, y_center: {y_center:.6f}, width: {width:.6f}, height: {height:.6f}")
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    

    # # print("Detection results:", results)

    # # display the results on the image using OpenCV
    # # 3. Plot the detection results directly onto the image
    # # This automatically draws bounding boxes, labels, and confidence scores
    # annotated_img = results[0].plot()

    # # 4. Display the image in a window using OpenCV
    # cv2.imshow("YOLO Detection Results", annotated_img)

    # # Keep the window open until a key is pressed
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# image = cv2.imread("/Users/michaelmandiberg/Downloads/yoga_test.jpg")
# results = model.predict(image, conf=0.2, verbose=False)

# # print("Detection results:", results)

# # display the results on the image using OpenCV
# # 3. Plot the detection results directly onto the image
# # This automatically draws bounding boxes, labels, and confidence scores
# annotated_img = results[0].plot()

# # 4. Display the image in a window using OpenCV
# cv2.imshow("YOLO Detection Results", annotated_img)

# # Keep the window open until a key is pressed
# cv2.waitKey(0)
# cv2.destroyAllWindows()