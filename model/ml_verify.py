"""
Simple verification script to import key ML libraries and print their versions.
Run this with the project's venv python to verify installations.
"""
import sys
from importlib import import_module

packages = [
    "sys",
    "numpy",
    "cv2",
    "tensorflow",
    "torch",
    "torchvision",
    "torchaudio",
    "PIL",
    "sklearn",
    "matplotlib",
]

results = {}
for pkg in packages:
    try:
        if pkg == "sys":
            results[pkg] = sys.version
        else:
            m = import_module(pkg)
            # heuristics for extracting version attribute
            ver = getattr(m, "__version__", None) or getattr(m, "version", None)
            results[pkg] = str(ver)
    except Exception as e:
        results[pkg] = f"ERROR: {e!r}"

print("Verification results:")
for k, v in results.items():
    print(f"- {k}: {v}")

# Exit non-zero if any critical ML libs failed
critical = ["numpy", "cv2", "tensorflow", "torch"]
failed = [p for p in critical if results.get(p, "").startswith("ERROR")]
if failed:
    print("\nOne or more critical packages failed to import:", failed)
    sys.exit(2)
else:
    print("\nAll critical packages imported successfully.")
