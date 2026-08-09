import os
import re
import shutil

FOLDER = "/Users/michaelmandiberg/Downloads/dupe_sorting"
TESTING = False


def get_folder_info(folder_name):
    """
    Split folder name on the LAST underscore.

    Example:
        clustercc0_p1_t0_om1_1785793247.482856
        ->
        base = clustercc0_p1_t0_om1
        timestamp = 1785793247.482856
    """
    match = re.match(r"^(.+)_(\d+\.\d+)$", folder_name)

    if not match:
        return None

    base = match.group(1)
    timestamp = float(match.group(2))

    return base, timestamp


def get_folder_size(folder_path):
    """
    Return total size of all files contained in the folder.
    """
    total = 0

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)

            try:
                total += os.path.getsize(filepath)
            except OSError as e:
                print(f"[WARNING] Could not get size: {filepath}")
                print(f"          {e}")

    return total


# ---------------------------------------------------------
# Find candidate folders
# ---------------------------------------------------------

folders = []

for name in os.listdir(FOLDER):
    path = os.path.join(FOLDER, name)

    if not os.path.isdir(path):
        continue

    info = get_folder_info(name)

    if info is None:
        continue

    base, timestamp = info

    folders.append({
        "name": name,
        "path": path,
        "base": base,
        "timestamp": timestamp,
    })


# ---------------------------------------------------------
# Group by base name
# ---------------------------------------------------------

groups = {}

for item in folders:
    groups.setdefault(item["base"], []).append(item)


# ---------------------------------------------------------
# Process duplicate groups
# ---------------------------------------------------------

for base, items in groups.items():

    if len(items) < 2:
        continue

    # Sort oldest -> newest
    items.sort(key=lambda x: x["timestamp"])

    print()
    print("=" * 80)
    print(f"DUPLICATE GROUP: {base}")
    print("=" * 80)

    # Calculate sizes
    for item in items:
        item["size"] = get_folder_size(item["path"])

        print(
            f"{item['name']}\n"
            f"    timestamp: {item['timestamp']}\n"
            f"    size:      {item['size']:,} bytes"
        )

    # Compare every folder against the newest folder.
    #
    # If sizes are identical, older folders can be deleted.
    # If sizes differ, keep both.
    newest = items[-1]

    for item in items[:-1]:

        if item["size"] == newest["size"]:

            if TESTING:
                print(
                    f"[TESTING] WOULD DELETE:\n"
                    f"    {item['path']}\n"
                    f"    because it is older than:\n"
                    f"    {newest['path']}\n"
                    f"    and both folders are "
                    f"{item['size']:,} bytes"
                )
            else:
                print(
                    f"[DELETE] {item['path']}\n"
                    f"         older duplicate of {newest['path']}"
                )

                shutil.rmtree(item["path"])

        else:
            print(
                f"[KEEP] Different sizes -- keeping both:\n"
                f"    {item['name']} = {item['size']:,} bytes\n"
                f"    {newest['name']} = {newest['size']:,} bytes"
            )

print()
print("Done.")