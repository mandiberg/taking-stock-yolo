from datasets import load_dataset

# Load dataset from Hugging Face
print("Loading dataset from Hugging Face...")
ds = load_dataset("visual-layer/oxford-flowers-vl-enriched")

# Inspect dataset structure
print("\nDataset splits:", list(ds.keys()))
if 'train' in ds:
    print("\nTrain split size:", len(ds['train']))
    print("\nFirst item keys:", list(ds['train'][0].keys()))
    print("\nFirst item:")
    for key, value in ds['train'][0].items():
        if key == 'image' or key == 'img':
            print(f"  {key}: <Image object>")
        else:
            print(f"  {key}: {value}")
    print("\nDataset features:")
    for key, feature in ds['train'].features.items():
        print(f"  {key}: {feature}")
