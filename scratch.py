# from ultralytics import YOLOE
# from pathlib import Path

# # Load the YOLOE-26x zero-shot model from the repo root regardless of cwd.
# repo_root = Path(__file__).resolve().parent
# candidate_weights = ["yoloe-26x.pt", "yolo26x.pt"]
# weights_path = next((repo_root / name for name in candidate_weights if (repo_root / name).exists()), None)

# if weights_path is None:
# 	raise FileNotFoundError(
# 		"Could not find YOLOE weights. Expected one of: "
# 		+ ", ".join(str(repo_root / name) for name in candidate_weights)
# 	)

# model = YOLOE(str(weights_path))

# # Perform text-prompted auto-detection on a video
# results = model.predict("/Users/michaelmandiberg/Downloads/yoga_test.jpg", prompt="person", save=True)

# print("Detection results:", results)



from ultralytics import YOLOE
# import supervision as sv
import cv2
 
model = YOLOE("yoloe-26l-seg.pt")           # newest YOLO26 backbone
 
# Multi-class long-tail prompts on a busy NYC street
NAMES = ["person", "yoga mat"]
model.set_classes(NAMES)
 
image = cv2.imread("/Users/michaelmandiberg/Downloads/yoga_test.jpg")
results = model.predict(image, conf=0.2, verbose=False)

# print("Detection results:", results)

# display the results on the image using OpenCV
# 3. Plot the detection results directly onto the image
# This automatically draws bounding boxes, labels, and confidence scores
annotated_img = results[0].plot()

# 4. Display the image in a window using OpenCV
cv2.imshow("YOLO Detection Results", annotated_img)

# Keep the window open until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()



# detections = sv.Detections.from_ultralytics(results[0])
 
# annotated = image.copy()
# # Mask layer
 
# annotated = sv.MaskAnnotator(opacity=0.4).annotate(scene=annotated, detections=detections)
# # Box layer
 
# annotated = sv.BoxAnnotator(thickness=2).annotate(scene=annotated, detections=detections)
# labels = [f"{NAMES[int(c)]} {p:.2f}" for c, p in zip(detections.class_id, detections.confidence)]
# # Label layer
 
# annotated = sv.LabelAnnotator(text_scale=0.5).annotate(scene=annotated, detections=detections, labels=labels)
 
# cv2.imwrite("output_city.jpg", annotated)
# print(f"Detected {len(detections)} object(s) for prompts {NAMES}")