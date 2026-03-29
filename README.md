# taking-stock-yolo
Training custom YOLO model for objects in stock photographs

## Open-vocab bootstrap labels

Use `bootstrap_open_vocab_labels.py` to generate first-pass pseudo-labels from text prompts (OWL-ViT via Hugging Face), plus a confidence CSV for review queues.

Install deps:

```bash
pip install -r requirements_open_vocab.txt
```

Run:

```bash
python bootstrap_open_vocab_labels.py \
	--images-dir /path/to/unlabeled_images \
	--output-dir /path/to/bootstrap_output \
	--classes "lipstick,electric drill,flag,wallet,poker card"
```

Optional prompt synonyms file (`--prompts-json`):

```json
{
	"electric drill": ["electric drill", "power drill", "cordless drill"],
	"lipstick": ["lipstick", "lip stick tube"]
}
```

Output includes:

- `labels/*.txt` YOLO labels (`accept` + `review` detections)
- `confidence_report.csv` review queue metadata with scores and boxes
- `classes.txt` class order used for YOLO IDs
