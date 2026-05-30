import os
import shutil


SOURCE_ROOT = os.path.join("/Volumes/LaCie", "test_output", "sort")
SORTED_DEST_ROOT = os.path.join("/Volumes/LaCie", "test_output", "sorted")
LABEL_STUDIO_DEST_ROOT = os.path.join("/Volumes/LaCie", "test_output", "label_studio_ready")
YOLO_DEST_ROOT = os.path.join("/Volumes/LaCie", "test_output", "YOLO_ready")


def ensure_dir(path: str):
	os.makedirs(path, exist_ok=True)


def is_hidden_name(name: str):
	return name.startswith(".")


def remove_empty_parents(start_dir: str, stop_dir: str):
	current = start_dir
	stop_dir = os.path.abspath(stop_dir)
	while os.path.abspath(current).startswith(stop_dir) and os.path.abspath(current) != stop_dir:
		if not os.path.isdir(current):
			current = os.path.dirname(current)
			continue
		if os.listdir(current):
			break
		os.rmdir(current)
		current = os.path.dirname(current)


def move_folder_contents(src_dir: str, dst_dir: str):
	ensure_dir(dst_dir)
	moved_count = 0
	for name in os.listdir(src_dir):
		if is_hidden_name(name):
			continue
		src_path = os.path.join(src_dir, name)
		dst_path = os.path.join(dst_dir, name)
		if os.path.exists(dst_path):
			print(f"[SKIP] Destination already exists: {dst_path}")
			continue
		shutil.move(src_path, dst_path)
		moved_count += 1
	return moved_count


def get_destination_dir(current_root: str, source_root: str):
	relative_path = os.path.relpath(current_root, source_root)
	path_parts = relative_path.split(os.sep)

	if len(path_parts) >= 3 and path_parts[-1] == "sorted_jpgs":
		return os.path.join(SORTED_DEST_ROOT, relative_path), "sorted_jpgs"

	if len(path_parts) >= 3 and path_parts[-2] == "relabel_these" and path_parts[-1] in {"images", "labels"}:
		return os.path.join(LABEL_STUDIO_DEST_ROOT, path_parts[0], path_parts[-1]), f"relabel_{path_parts[-1]}"

	if len(path_parts) >= 3 and path_parts[-2] == "move_these" and path_parts[-1] in {"images", "labels"}:
		return os.path.join(YOLO_DEST_ROOT, path_parts[0], path_parts[-1]), f"move_{path_parts[-1]}"

	return None, None


def move_matching_folders(source_root: str):
	if not os.path.isdir(source_root):
		print(f"Source root not found: {source_root}")
		return

	folders_processed = 0
	files_moved = 0

	for current_root, dirnames, _ in os.walk(source_root, topdown=True):
		dirnames[:] = [name for name in dirnames if not is_hidden_name(name)]

		destination_dir, move_type = get_destination_dir(current_root, source_root)
		if destination_dir is None:
			continue

		ensure_dir(os.path.dirname(destination_dir))

		if not os.path.exists(destination_dir):
			shutil.move(current_root, destination_dir)
			moved_here = 0
			for _, _, filenames in os.walk(destination_dir):
				moved_here += len([name for name in filenames if not is_hidden_name(name)])
			files_moved += moved_here
			folders_processed += 1
			print(f"[MOVED:{move_type}] {current_root} -> {destination_dir}")
		else:
			moved_here = move_folder_contents(current_root, destination_dir)
			if moved_here:
				files_moved += moved_here
				folders_processed += 1
				print(f"[MERGED:{move_type}] {current_root} -> {destination_dir}")
			else:
				print(f"[SKIP:{move_type}] No files moved from {current_root}")

			if os.path.isdir(current_root) and not os.listdir(current_root):
				os.rmdir(current_root)

		remove_empty_parents(os.path.dirname(current_root), source_root)
		dirnames[:] = []

	print(f"\nDone.")
	print(f"Folders processed: {folders_processed}")
	print(f"Files moved: {files_moved}")
	print(f"Source root: {source_root}")
	print(f"Sorted destination root: {SORTED_DEST_ROOT}")
	print(f"Label Studio destination root: {LABEL_STUDIO_DEST_ROOT}")
	print(f"YOLO destination root: {YOLO_DEST_ROOT}")


def main():
	move_matching_folders(SOURCE_ROOT)


if __name__ == "__main__":
	main()