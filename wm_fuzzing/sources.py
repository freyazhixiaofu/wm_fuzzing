"""Location of the bundled upstream code, independent of the working directory."""

from pathlib import Path

UPSTREAM = Path(__file__).resolve().with_name("upstream")
