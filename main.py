"""
main.py  —  Dataset QA Tool V2
Entry point: auto-installs dependencies, then launches the GUI.

Changes from V1:
    • Version string updated to V2 in the print message.
    • No other changes required.
"""

import subprocess
import sys


def _install(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Auto-install Pillow if missing
try:
    import PIL  # noqa: F401
except ImportError:
    print("Installing Pillow...")
    _install("pillow")

# Auto-install Matplotlib if missing
try:
    import matplotlib  # noqa: F401
except ImportError:
    print("Installing Matplotlib...")
    _install("matplotlib")

# Launch GUI
from gui import DatasetQATool  # noqa: E402

if __name__ == "__main__":
    print("Starting Dataset QA Tool V2...")
    app = DatasetQATool()
    app.mainloop()