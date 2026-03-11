"""Find one high-confidence example per damage class from building_lookup.csv."""
import csv

LOOKUP = "../../../Benchmark-Model-xView2/cnn/data/building_lookup.csv"
LABELS = ["no-damage", "minor-damage", "major-damage", "destroyed"]

best = {}
with open(LOOKUP) as f:
    for row in csv.DictReader(f):
        label = row["pred_label"]
        conf  = float(row["confidence"])
        if label not in best or conf > float(best[label]["confidence"]):
            best[label] = row

for label in LABELS:
    r = best[label]
    print(f"{label:20s}  conf={float(r['confidence']):.2f}  tile={r['tile_id']}  uid={r['uid'][:8]}")
