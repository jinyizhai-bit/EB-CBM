"""Fetch public ISIC Archive metadata (incl. diagnosis) for the ISIC 2018 Task 1-2 collections (anonymous API).

    python scripts/fetch_isic_metadata.py [--out data/metadata/isic2018_task12]

Writes isic_archive_api_metadata_task1-2_{training,validation,test}.csv, which cbm.data.manifests.build_isic2018_task12
reads to derive the 3-class diagnosis label (nevus / melanoma / seborrheic keratosis) from ``diagnosis_3``.
The repository ships the snapshot used for the paper; re-fetching may differ if the archive metadata was revised.
"""
import argparse
import csv
import json
import os
import urllib.request
from collections import Counter

COLLECTIONS = {"63": "training", "62": "validation", "64": "test"}   # ISIC Archive collection ids (Task 1-2)
CLINICAL = ["diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4", "diagnosis_5", "diagnosis_confirm_type",
            "melanocytic", "age_approx", "sex", "anatom_site_general", "anatom_site_1"]


def fetch(cid: str) -> list[dict]:
    url = f"https://api.isic-archive.com/api/v2/images/search/?collections={cid}&limit=100"
    rows = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        for it in d["results"]:
            c = it["metadata"].get("clinical", {})
            a = it["metadata"].get("acquisition", {})
            rows.append({"isic_id": it["isic_id"], "license": it.get("copyright_license"),
                         "attribution": it.get("attribution"), **{k: c.get(k) for k in CLINICAL},
                         "image_type": a.get("image_type"), "dermoscopic_type": a.get("dermoscopic_type"),
                         "full_url": it["files"]["full"]["url"]})
        url = d.get("next")
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(os.environ.get("CBM_DATA_ROOT", os.path.join(os.path.dirname(here), "data")),
                               "metadata", "isic2018_task12")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for cid, name in COLLECTIONS.items():
        rows = fetch(cid)
        out = os.path.join(a.out, f"isic_archive_api_metadata_task1-2_{name}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(out, len(rows), Counter(r["diagnosis_3"] for r in rows).most_common(6))


if __name__ == "__main__":
    main()
