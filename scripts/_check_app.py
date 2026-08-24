import sys

sys.path.insert(0, "src")
from hr_rag.api.main import app

print("APP OK", app.title)
for r in app.routes:
    if hasattr(r, "path"):
        print(r.name, r.path)
print("VERSION", app.version)