"""Rebuild the large data files that are stored in this repo as split zip parts.

Files bigger than 10 MB were uploaded as <name>.zip.001, .002, ... next to where the
original file belongs. Run this once after cloning (Python 3, no extra packages):

    python restore_large_files.py

Add --delete-parts to remove the .zip.NNN parts after a successful restore.
"""
import glob, io, os, sys, zipfile

root = os.path.dirname(os.path.abspath(__file__))
groups = {}
for p in glob.glob(os.path.join(root, "**", "*.zip.[0-9][0-9][0-9]"), recursive=True):
    groups.setdefault(p[:-4], []).append(p)

for base, parts in sorted(groups.items()):
    parts.sort()
    data = b"".join(open(p, "rb").read() for p in parts)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(os.path.dirname(base))
        names = zf.namelist()
    print("restored", os.path.relpath(os.path.join(os.path.dirname(base), names[0]), root))
    if "--delete-parts" in sys.argv:
        for p in parts:
            os.remove(p)
print(f"Done: {len(groups)} file(s) restored.")
