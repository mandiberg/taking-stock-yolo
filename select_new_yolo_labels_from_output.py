import os

ROOT = os.path.join("/Volumes/OWC5/segment_images_92_headphones", "test_output/sort")
# ROOT = os.path.join("/Users/michael.mandiberg/Documents/YOLO_Training_Data/sorted_images_tempexcluded/misc_val_headphones_manual", "test_output/sort")
# ROOT = os.path.join("/Users/michaelmandiberg/Documents/yolo", "gun_sort_forV3/sort")
# SPLITS = ["train", "val"]
move_list_folder = os.path.join(ROOT, "move_these")
relabel_list_folder = os.path.join(ROOT, "relabel_these")
files_to_move_folder = os.path.join(ROOT, "all_yolo_labels")
decoy_list_folder =  os.path.join(ROOT, "decoys")

def ensure_dir(path: str):
	os.makedirs(path, exist_ok=True)


def extract_uid(filename: str):
	root, _ = os.path.splitext(filename)
	if "_YOLO_debug" in root:
		parts = root.split("_YOLO_debug")
		sub_parts = parts[0].split("_")
		UID = sub_parts[-1]
		return UID
	elif "_" not in root:
		parts = root.split(".")
		return parts[0]
	else:
		parts = root.split("_")
		if len(parts) < 2:
			return None
		return parts[1]


def index_files_by_uid(base_dir: str):
	image_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
	images_by_uid = {}
	labels_by_uid = {}
	for name in os.listdir(base_dir):
		path = os.path.join(base_dir, name)
		if not os.path.isfile(path):
			continue
		uid = extract_uid(name)
		if uid is None:
			continue
		root, ext = os.path.splitext(name)
		if ext in image_exts:
			if uid in images_by_uid:
				print(f"[WARN] Duplicate image UID '{uid}' in {base_dir}; keeping first: {os.path.basename(images_by_uid[uid])}")
				continue
			images_by_uid[uid] = path
		elif ext.lower() == ".txt":
			if uid in labels_by_uid:
				print(f"[WARN] Duplicate label UID '{uid}' in {base_dir}; keeping first: {os.path.basename(labels_by_uid[uid])}")
				continue
			labels_by_uid[uid] = path
	return images_by_uid, labels_by_uid

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

	images_by_uid, labels_by_uid = index_files_by_uid(files_to_move_folder)

	# Build set of UIDs from files present in move_list_folder (or move_list_folder/images)
	image_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
	move_search_dirs = [move_list_folder]
	images_subdir = os.path.join(move_list_folder, "images")
	if os.path.isdir(images_subdir):
		move_search_dirs.append(images_subdir)

	uids = set()
	seen_files = 0
	for search_dir in move_search_dirs:
		for name in os.listdir(search_dir):
			path = os.path.join(search_dir, name)
			if not os.path.isfile(path):
				continue
			seen_files += 1
			root, ext = os.path.splitext(name)
			if ext in image_exts or ext == "":
				uid = extract_uid(name)
				if uid is not None:
					uids.add(uid)

	if seen_files == 0:
		print(f"No files found in move list folder: {move_list_folder}. Nothing to do.")
		return 0, 0, 0, 0

	if not uids:
		print("No image names found in move list folder. Nothing to do.")
		return 0, 0, 0, 0

	moved_images = 0
	moved_labels = 0
	missing_images = 0
	missing_labels = 0

	for uid in sorted(uids):
		# Locate source image in files_to_move_folder by UID
		src_img = images_by_uid.get(uid)
		if src_img is None:
			print(f"[WARN] Image not found for UID '{uid}' in {files_to_move_folder}")
			missing_images += 1
		else:
			dst_img = os.path.join(images_out, os.path.basename(src_img))
			if os.path.exists(dst_img):
				print(f"[SKIP] Image already exists at destination: {dst_img}")
			else:
				os.rename(src_img, dst_img)
				moved_images += 1
		if save_labels:
			# Locate source label (.txt) by UID
			src_lbl = labels_by_uid.get(uid)
			if src_lbl is None:
				print(f"[WARN] Label not found for UID '{uid}' in {files_to_move_folder}")
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