# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

cases = [
    ("عبد الرحمن ابوزيد على",   "abd split — should join next word"),
    ("عبدالرحمن ابوزيد على",    "abd already joined — normal split"),
    ("محمد حمدى على الجوهرى",   "normal name"),
    ("عبدالله",                  "single word starting with abd"),
    ("عبد",                      "only abd, nothing after"),
    ("عبد الله",                 "abd + الله"),
]

for raw_last, label in cases:
    words = raw_last.strip().split()
    if not words:
        mid, last = None, None
    elif len(words) == 1:
        mid, last = None, words[0]
    elif words[0] == "عبد" and len(words) >= 2:
        mid  = words[0] + words[1]
        last = " ".join(words[2:]) or None
    else:
        mid  = words[0]
        last = " ".join(words[1:]) or None

    print(f"[{label}]")
    print(f"  Input : {raw_last}")
    print(f"  Middle: {mid}")
    print(f"  Last  : {last}")
    print()
