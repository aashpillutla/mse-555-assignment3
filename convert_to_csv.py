"""Convert scored_notes.json to scored_notes.csv (client_id, session, score)."""
import json
import csv

input_path = "/Users/aashrita/assignment3/output/q1/scored_notes.json"
output_path = "/Users/aashrita/assignment3/output/q1/scored_notes.csv"

with open(input_path, "r") as f:
    data = json.load(f)

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["client_id", "session", "score"])
    for record in data:
        client_id = record["client_id"]
        scores = record["estimated_trajectory_vector"]
        for i, score in enumerate(scores, start=2):  # transitions: 1→2, 2→3, ...
            writer.writerow([client_id, i, score])

print(f"Saved: {output_path}")
print(f"Total rows: {sum(len(r['estimated_trajectory_vector']) for r in data)}")
