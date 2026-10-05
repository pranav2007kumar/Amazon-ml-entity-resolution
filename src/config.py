"""Paths shared by every stage. Override with environment variables if needed."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "ER_DATA_DIR",
    os.path.join(_HERE, "..", "..", "..", "6ab10eb3b23ba_student_resource", "student_resource", "dataset"),
)
WORK_DIR = os.environ.get("ER_WORK_DIR", os.path.join(_HERE, "..", "..", "..", "work"))
OUT_DIR = os.environ.get("ER_OUT_DIR", os.path.join(_HERE, "..", "..", "..", "output"))
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
