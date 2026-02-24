"""
Thin wrapper to keep `python simulate.py` working.
"""
import runpy


if __name__ == "__main__":
    runpy.run_path("scripts/simulate.py", run_name="__main__")
