from ultralytics import YOLO
import os

# Load pretrained model
model = YOLO('yolov8x.pt')  # Options: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26x-objv1-150.pt
# model2 = YOLO('yolov8x.pt')  # Options: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26x-objv1-150.pt
# model2 = YOLO('yolo26x-objv1-150.pt')  # Options: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26x-objv1-150.pt
# DATA_FOLDER = "/Users/michael.mandiberg/Documents/takingstock_production/labeled_images_nov19"
    
# Source - https://stackoverflow.com/a
# Posted by the_artemi8
# Retrieved 2025-12-08, License - CC BY-SA 4.0

def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 10
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
    print(f"{num_freeze} layers are freezed.")

model.add_callback("on_train_start", freeze_layer)
# model2.add_callback("on_train_start", freeze_layer)
# model.train(data="./dataset.yaml")

# Train
results = model.train(
    data="/Users/michaelmandiberg/Documents/GitHub/taking-stock-yolo/yolo_dataset/data.yaml",  # absolute path
    epochs=140,
    imgsz=640,
    batch=16,        # Fixed small batch for low-memory debugging on 32GB RAM
    name='takingstock_test_c45_yolov8x',  # Experiment name
    patience=20,    # Early stopping
    device='mps',       # mps for Mac with M1/M2/M3 chips, else 'cuda' or 'cpu'
    workers=2,        # Lower host RAM pressure while debugging MPS crashes
    project='/Users/michaelmandiberg/Documents/GitHub/taking-stock-yolo/runs',  # Save to project directory
    exist_ok=True,  # Overwrite existing experiment with same name
    augment=True,
    cache=False    # Avoid large RAM spikes during debugging
)

# results2 = model2.train(
#     data="/Users/michaelmandiberg/Documents/GitHub/taking-stock-yolo/yolo_dataset2/data.yaml",  # absolute path
#     epochs=100,
#     imgsz=640,
#     batch=16,       # Reduce if you get memory errors
#     name='takingstock_glassescards_v0_yolov8x',  # Experiment name
#     patience=20,    # Early stopping
#     device='mps',       # mps for Mac with M1/M2/M3 chips, else 'cuda' or 'cpu'
#     workers=8,        # ✅ CPU workers, not GPU cores
#     project='/Users/michael.mandiberg/Documents/GitHub/taking-stock-yolo/runs',  # Save to project directory
#     exist_ok=True,  # Overwrite existing experiment with same name
#     augment=True,
#     cache='ram'   # ⭐ Enable RAM caching
# )

# Source - https://stackoverflow.com/a
# Posted by the_artemi8
# Retrieved 2025-12-08, License - CC BY-SA 4.0

