import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open(sys.argv[1], encoding="utf-8") as f:
    nb = json.load(f)
print("nbformat", nb["nbformat"])
for i, c in enumerate(nb["cells"]):
    print(f"\n===== cell {i} ({c['cell_type']}) =====")
    print(c["source"])