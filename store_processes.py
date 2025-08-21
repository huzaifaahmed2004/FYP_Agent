import json
from upstash_redis import Redis
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path

# Connect to Redis (keep existing connection settings)
redis = Redis(url="https://pet-crab-9639.upstash.io", token="ASWnAAIjcDFjYzg0ZjQ4YjE3MjE0OTFiODdmMmJmYjAyZDMzZTFiNXAxMA")

# Load embedding model
model = SentenceTransformer("all-mpnet-base-v2")


# We expect the Excel file to already include a description column. No generation
# of description text will be performed — the script will use the existing
# description value from the sheet and skip rows that lack it.


# Read Excel file `hospital_processes.xlsx` from the workspace root
xlsx_path = Path(__file__).parent / "hospital_processes.xlsx"
if not xlsx_path.exists():
    raise FileNotFoundError(f"Expected file not found: {xlsx_path} - place your Excel file in the script directory")

# Read sheet into DataFrame. Accept columns: id, name, description (case-insensitive)
df = pd.read_excel(xlsx_path, engine="openpyxl")
if df.empty:
    raise ValueError("Excel file is empty or no rows to process")

# Normalize columns

cols = {c.lower(): c for c in df.columns}
# Prefer explicit 'process_*' column names but fall back to generic ones
id_col = cols.get("process_id") or cols.get("id")
name_col = cols.get("process_name") or cols.get("name")
desc_col = cols.get("process_description") or cols.get("description") or cols.get("desc") or cols.get("details")

if name_col is None:
    raise ValueError("Excel file must contain a 'process_name' (or 'name') column")
if desc_col is None:
    raise ValueError("Excel file must contain a 'process_description' (or 'description') column")

# Build processes list from rows, assign integer ids if missing
processes = []
next_id = 1
for _, row in df.iterrows():
    # determine id
    if id_col and pd.notna(row[id_col]):
        # try convert supplied id to int, otherwise fallback to sequential id
        try:
            pid = int(row[id_col])
        except Exception:
            pid = next_id
            next_id += 1
    else:
        pid = next_id
        next_id += 1

    name = str(row[name_col]).strip()
    if not name:
        # skip rows without a name
        continue

    raw_desc = None
    if desc_col and pd.notna(row[desc_col]):
        raw_desc = str(row[desc_col]).strip()

    if not raw_desc:
        print(f"Skipping row id={pid} name={name} - missing description")
        continue

    description = raw_desc

    processes.append({"id": pid, "name": name, "description": description})


# Store each process as a single Redis item with id (int), name, description (~250 words) and vector.
for proc in processes:
    # Embed both the process name and the description together for better matching
    text_for_embedding = f"{proc['name']}. {proc['description']}"
    vector = model.encode(text_for_embedding).tolist()
    enriched = {
        "id": proc["id"],
        "name": proc["name"],
        "description": proc["description"],
        "vector": vector
    }
    redis.set(f"process:{proc['id']}", json.dumps(enriched))

print(f"Stored {len(processes)} processes from {xlsx_path.name} with integer id, name, description (~250 words) and vector embeddings.")
