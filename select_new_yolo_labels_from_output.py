import os

ROOT = os.path.join("/Volumes/OWC5/segment_images_90_stethoscope", "test_output/sort")
# SPLITS = ["train", "val"]
move_list_folder = os.path.join(ROOT, "move_these")
relabel_list_folder = os.path.join(ROOT, "relabel_these")
files_to_move_folder = os.path.join(ROOT, "all_yolo_labels")
decoy_list_folder =  os.path.join(ROOT, "decoys")

def ensure_dir(path: str):
	os.makedirs(path, exist_ok=True)


def find_existing_image(base_dir: str, stem: str):
	# Try common image extensions
	for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
		p = os.path.join(base_dir, stem + ext)
		if os.path.isfile(p):
			return p
	return None

def move_files(files_to_move_folder: str, move_list_folder: str, save_labels: bool = True):
	images_out = os.path.join(move_list_folder, "images")
	ensure_dir(images_out)
	if save_labels:
		labels_out = os.path.join(move_list_folder, "labels")
		ensure_dir(labels_out)

	# Build set of basenames (without extension) from files present in move_list_folder
	move_names = [f for f in os.listdir(move_list_folder) if os.path.isfile(os.path.join(move_list_folder, f))]
	if bool(not move_names):
		print(f"No files found in move list folder: {move_list_folder}. Nothing to do.")
		return 0, 0, 0, 0
	stems = set()
	for name in move_names:
		root, ext = os.path.splitext(name)
		if ext.lower() in (".jpg", ".jpeg", ".png"):
			stems.add(root)

	if not stems:
		print("No image names found in move list folder. Nothing to do.")
		return

	moved_images = 0
	moved_labels = 0
	missing_images = 0
	missing_labels = 0

	for stem in sorted(stems):
		# Locate source image in files_to_move_folder
		src_img = find_existing_image(files_to_move_folder, stem)
		if src_img is None:
			print(f"[WARN] Image not found for '{stem}' in {files_to_move_folder}")
			missing_images += 1
		else:
			dst_img = os.path.join(images_out, os.path.basename(src_img))
			if os.path.exists(dst_img):
				print(f"[SKIP] Image already exists at destination: {dst_img}")
			else:
				os.rename(src_img, dst_img)
				moved_images += 1
		if save_labels:
			# Locate source label (.txt)
			src_lbl = os.path.join(files_to_move_folder, stem + ".txt")
			if not os.path.isfile(src_lbl):
				print(f"[WARN] Label not found for '{stem}' in {files_to_move_folder}")
				missing_labels += 1
			else:
				dst_lbl = os.path.join(labels_out, os.path.basename(src_lbl))
				if os.path.exists(dst_lbl):
					print(f"[SKIP] Label already exists at destination: {dst_lbl}")
				else:
					os.rename(src_lbl, dst_lbl)
					moved_labels += 1
	return moved_images, moved_labels, missing_images, missing_labels

def main():
	if not os.path.isdir(move_list_folder):
		print(f"Move list folder not found: {move_list_folder}")
		return
	ensure_dir(files_to_move_folder)
	ensure_dir(relabel_list_folder)
	ensure_dir(decoy_list_folder)

	moved_images, moved_labels, missing_images, missing_labels = move_files(files_to_move_folder, move_list_folder)

	print(f"\nDone. Data for: {move_list_folder}")
	print(f"Moved images: {moved_images}")
	print(f"Moved labels: {moved_labels}")
	print(f"Missing images: {missing_images}")
	print(f"Missing labels: {missing_labels}")
	print(f"Data saved to: {files_to_move_folder}")

	moved_images, moved_labels, missing_images, missing_labels = move_files(files_to_move_folder, relabel_list_folder)

	print(f"\nDone. Data for: {relabel_list_folder}")
	print(f"Moved images: {moved_images}")
	print(f"Moved labels: {moved_labels}")
	print(f"Missing images: {missing_images}")
	print(f"Missing labels: {missing_labels}")
	print(f"Data saved to: {files_to_move_folder}")

	moved_images, moved_labels, missing_images, missing_labels = move_files(files_to_move_folder, decoy_list_folder, save_labels=False)

	print(f"\nDone. Data for: {decoy_list_folder}")
	print(f"Moved images: {moved_images}")
	print(f"Moved labels: {moved_labels}")
	print(f"Missing images: {missing_images}")
	print(f"Missing labels: {missing_labels}")
	print(f"Data saved to: {files_to_move_folder}")


if __name__ == "__main__":
	main()