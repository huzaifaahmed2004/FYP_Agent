import os
import json
import time
import numpy as np
from dotenv import load_dotenv
from upstash_redis import Redis
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import google.generativeai as genai
import re
import hashlib
import threading
import random

# Load environment variables
load_dotenv()

# Configure Google Generative AI only if an API key is provided.
_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if _GOOGLE_API_KEY:
    genai.configure(api_key=_GOOGLE_API_KEY)
_gemini_model = None  # lazy init when generation functions are called

# Simple in-process caches to reduce repeat Gemini calls for identical inputs
_NAME_CACHE = {}
_TASKS_LIST_CACHE = {}
_BATCH_DESC_CACHE = {}
_JOBS_CACHE = {}
_FUNCS_CACHE = {}

# Lightweight in-process rate limit and retry to avoid 429s
_RATE_LOCK = threading.Lock()
_LAST_CALL_TS = 0.0
_MIN_INTERVAL = float(os.getenv("GEMINI_MIN_INTERVAL_SEC", "0.5"))  # seconds between calls
_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "4"))
_BACKOFF_BASE = float(os.getenv("GEMINI_BACKOFF_BASE", "1.7"))

def _gen_content(prompt: str):
    """Call Gemini with basic rate limiting and exponential backoff on 429-like errors."""
    global _LAST_CALL_TS
    model = _get_gemini()
    retries = 0
    while True:
        # Simple QPS gate
        with _RATE_LOCK:
            now = time.time()
            wait = _MIN_INTERVAL - (now - _LAST_CALL_TS)
            if wait > 0:
                time.sleep(wait)
            _LAST_CALL_TS = time.time()
        try:
            return model.generate_content(prompt)
        except Exception as e:
            msg = str(e)
            transient = ("429" in msg) or ("ResourceExhausted" in msg) or ("rate" in msg.lower())
            if not transient or retries >= _MAX_RETRIES:
                raise
            # Exponential backoff with jitter
            sleep_s = (_BACKOFF_BASE ** retries) + random.uniform(0, 0.3)
            time.sleep(sleep_s)
            retries += 1

# Connect to Redis (Mumbai region)
start_conn = time.time()
redis = Redis(url="https://pet-crab-9639.upstash.io", token="ASWnAAIjcDFjYzg0ZjQ4YjE3MjE0OTFiODdmMmJmYjAyZDMzZTFiNXAxMA")

print(f"Connected to Redis in {time.time() - start_conn:.2f} seconds")

# Load models only once
print("Loading embedding model...")
model = SentenceTransformer("all-mpnet-base-v2")
print("Embedding model loaded.")

def _get_gemini():
    """Lazy initializer for the Gemini model. Raises if no API key is configured."""
    global _gemini_model
    if _gemini_model is None:
        if not _GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set; generation features are disabled.")
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    return _gemini_model

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Retrieve processes from Redis
def retrieve_processes(query, top_k=3):
    """Return top_k candidate processes from Redis by cosine similarity.

    Each candidate is a dict: { 'sim','id','name','description','vector' }
    """
    query_vec = model.encode(query)
    candidates = []

    keys = redis.keys("process:*")
    for key in keys:
        raw = redis.get(key)
        if not raw:
            continue
        entry = json.loads(raw)
        # guard: ensure vector is a numpy array
        vec = np.array(entry.get("vector", []), dtype=float)
        if vec.size == 0:
            continue
        sim = float(cosine_similarity(query_vec, vec))
        candidates.append({
            'sim': sim,
            'id': entry.get('id'),
            'name': entry.get('name'),
            'description': entry.get('description'),
            'vector': vec
        })

    candidates.sort(key=lambda x: x['sim'], reverse=True)
    return candidates[:top_k]



# Generate a new process using Gemini
def generate_new_process(query):
    prompt = f"""
Generate a concise organizational process description for the following request:
"{query}"

Important: describe WHAT the process is — its purpose, scope, key outcomes, and who it's for. Do NOT provide step-by-step instructions or implementation steps.
Keep the response focused and high-level.
"""
    response = _gen_content(prompt)
    return response.text


def get_next_process_id():
    # find max existing integer id under keys 'process:<id>'
    keys = redis.keys("process:*")
    max_id = 0
    for k in keys:
        try:
            sk = k.split(b":" if isinstance(k, bytes) else ":")[-1]
            sid = int(sk)
            if sid > max_id:
                max_id = sid
        except Exception:
            continue
    return max_id + 1


def generate_process_description(query, approx_words: int = 400):
    prompt = f"""
Generate a clear, professional organizational process description for the following request.

Request: "{query}"

Requirements:
- Produce exactly {approx_words} words.
- Single paragraph only.
- No headings, subheadings, bullets, or numbered lists.
- No line breaks or blank lines; output must be one continuous paragraph.
- Do NOT include implementation instructions or step-by-step procedures; describe purpose, scope, expected outcomes, stakeholders, and constraints.
- Return only the plain text description (no quotes or surrounding punctuation).
"""
    resp = _gen_content(prompt)
    return resp.text


def generate_task_descriptions_batch(task_names: list[str], process_description: str, min_words: int = 600) -> list[str]:
    """Generate task descriptions for a list of task names in a single Gemini call.

    Returns a list of strings the same length as task_names. If parsing fails,
    falls back to best-effort splitting and pads with empty strings to match length.
    """
    if not task_names:
        return []
    # Cache key uses names tuple and description hash
    names_tuple = tuple(task_names)
    desc_norm = re.sub(r"\s+", " ", process_description or "").strip()
    hk = hashlib.sha256(desc_norm.encode("utf-8")).hexdigest()[:16]
    cache_key = (names_tuple, hk, int(min_words))
    cached = _BATCH_DESC_CACHE.get(cache_key)
    if cached is not None:
        return cached[:]
    numbered = "\n".join(f"{i+1}. {name}" for i, name in enumerate(task_names))
    prompt = f"""
You are given a business process context and a numbered list of task names.
Write a professional single-paragraph description for each task, in order.

Context:\n{process_description}\n
Rules for each task's description:
- Minimum {min_words} words (longer is acceptable).
- Single paragraph only.
- No headings, subheadings, bullets, or numbered lists.
- No line breaks or blank lines; output must be one continuous paragraph.
- Return only JSON: an array of strings in the same order as the tasks.

Tasks:\n{numbered}

Return strictly valid JSON array, no comments, no keys, no additional text.
"""
    resp = _gen_content(prompt)
    text = (resp.text or "").strip()
    # Try parse as JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            items = [str(x) for x in data]
        else:
            items = []
    except Exception:
        # Try extract JSON array substring
        try:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                sub = text[start:end+1]
                data = json.loads(sub)
                items = [str(x) for x in data] if isinstance(data, list) else []
            else:
                items = []
        except Exception:
            items = []
    # Fallback: naive split by two newlines
    if not items:
        parts = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        items = parts
    # Ensure alignment and length
    if len(items) < len(task_names):
        items = items + [""] * (len(task_names) - len(items))
    elif len(items) > len(task_names):
        items = items[:len(task_names)]
    _BATCH_DESC_CACHE[cache_key] = items
    if len(_BATCH_DESC_CACHE) > 128:
        _BATCH_DESC_CACHE.clear()
    return items


def generate_jobs_for_tasks_batch(process_name: str, process_description: str, tasks: list[dict], approx_words: int = 350) -> dict:
    """Group tasks into jobs and produce ~350-word job descriptions in a single Gemini call.

    Inputs:
      - process_name: str
      - process_description: str
      - tasks: list of {"name": str, "description": str}
      - approx_words: target words for each job description (~350)

    Returns dict with keys:
      {
        "jobs": [ {"name": str, "description": str, "task_indices": [int, ...]} , ...],
        "task_to_job": [str, ...]  # length == len(tasks), each element is the assigned job name
      }
    """
    # Normalize and build cache key
    pname = re.sub(r"\s+", " ", (process_name or "")).strip()
    pdesc = re.sub(r"\s+", " ", (process_description or "")).strip()
    tasks_norm = [
        {
            "name": re.sub(r"\s+", " ", (t.get("name", "") or "")).strip(),
            "description": re.sub(r"\s+", " ", (t.get("description", "") or "")).strip(),
        }
        for t in (tasks or [])
    ]
    key_src = json.dumps({"pn": pname, "pd": pdesc, "ts": tasks_norm, "w": int(approx_words)}, sort_keys=True)
    hk = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:24]
    cached = _JOBS_CACHE.get(hk)
    if cached is not None:
        # shallow copy to avoid external mutation
        return json.loads(json.dumps(cached))

    # Build a compact JSON of tasks for the prompt with indices
    tasks_for_prompt = [
        {"index": i, "name": t.get("name", ""), "description": t.get("description", "")}
        for i, t in enumerate(tasks_norm)
    ]
    tasks_json = json.dumps(tasks_for_prompt, ensure_ascii=False)

    prompt = f"""
You are given a business process and a list of operational tasks (each with a name and description).
Group tasks into JOBS, where a job is a concrete human role (a person/position) that can own one or more tasks.

Process Name: {pname}
Process Description: {pdesc}

Tasks (JSON with zero-based indices):
{tasks_json}

Naming Requirements for jobs:
- The job name MUST be a human role title (e.g., Coordinator, Specialist, Analyst, Manager, Supervisor, Technician, Administrator, Officer, Lead, Director).
- DO NOT output functional areas or processes as names (e.g., "Insurance Verification", "Billing and Claims Management", "Financial Reporting and Analysis").
- Prefer Title Case role names like: "Insurance Authorization Specialist", "Billing Specialist", "Claims Processor", "Revenue Cycle Manager", "Medical Coder", "Financial Analyst".
- Non-examples: "Insurance Verification and Pre-authorization", "Billing and Claims Management", "Financial Reporting", "Medical Coding".

Rules:
- Assign every task to exactly one job.
- A single task MUST have exactly one job, but a job MAY cover multiple tasks.
- Create the minimum sensible number of jobs while keeping responsibilities coherent.
- For each job, write a professional single-paragraph description of approximately {approx_words} words (acceptable range 330–370).
- No headings, bullets, numbered lists, or line breaks; one continuous paragraph per job.
- Use concise, neutral language; avoid placeholders or repetitive filler.
- Return STRICT JSON only, no extra commentary.

Output JSON schema:
{{
  "jobs": [
    {{"name": "<human role title>", "description": "<~{approx_words}-word paragraph>", "task_indices": [<int>, ...]}} , ...
  ],
  "task_to_job": ["<job name for task 0>", "<job name for task 1>", ...]  # length must equal number of tasks
}}
"""

    resp = _gen_content(prompt)
    text = (resp.text or "").strip()
    data = None
    # Try direct JSON parse
    try:
        data = json.loads(text)
    except Exception:
        # Try extract JSON block
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                sub = text[start:end+1]
                data = json.loads(sub)
        except Exception:
            data = None

    # Fallback empty structure
    if not isinstance(data, dict):
        data = {"jobs": [], "task_to_job": []}

    jobs = data.get("jobs") or []
    t2j = data.get("task_to_job") or []

    # Normalize job names to human role titles when the model returns functional areas.
    role_nouns = [
        "coordinator", "specialist", "analyst", "manager", "supervisor", "technician",
        "administrator", "officer", "lead", "director", "engineer", "consultant",
        "assistant", "associate"
    ]

    def _title_case(x: str) -> str:
        parts = re.split(r"\s+", (x or "").strip())
        return " ".join(p.capitalize() for p in parts if p)

    def _roleize(name: str) -> str:
        base = (name or "").strip()
        if not base:
            return "Specialist"
        low = base.lower()
        if any(n in low for n in role_nouns):
            return _title_case(base)
        # Heuristic mappings by keywords
        if re.search(r"\b(billing|claims?)\b", low):
            return "Billing Specialist"
        if re.search(r"\binsurance|authorization|authorisation|verification\b", low):
            return "Insurance Authorization Specialist"
        if re.search(r"\bfinancial|reporting|analysis|analytics|revenue cycle\b", low):
            return "Financial Analyst"
        if re.search(r"\bcoding|icd|cpt\b", low):
            return "Medical Coder"
        if re.search(r"\bmanagement\b", low):
            return "Operations Manager"
        # Fallback
        return _title_case(base + " Specialist")

    # Build renaming map and apply to jobs and task_to_job strings
    rename_map = {}
    for j in jobs:
        old = str(j.get("name", "")).strip()
        new = _roleize(old)
        j["name"] = new
        if old and old != new:
            rename_map[old] = new
    if t2j:
        t2j = [rename_map.get(str(x).strip(), str(x).strip()) for x in t2j]

    # Basic repair: ensure task_to_job covers all tasks
    if len(t2j) != len(tasks_norm):
        # derive from jobs mapping if available
        mapping = [None] * len(tasks_norm)
        for j in jobs:
            jname = str(j.get("name", "")).strip()
            for idx in j.get("task_indices", []) or []:
                if isinstance(idx, int) and 0 <= idx < len(mapping):
                    mapping[idx] = jname
        # fill any gaps with unique job per task
        for i in range(len(mapping)):
            if not mapping[i]:
                jname = f"{tasks_norm[i].get('name') or 'Task'} Job"
                jobs.append({"name": jname, "description": "", "task_indices": [i]})
                mapping[i] = jname
        t2j = mapping

    # Ensure each job has task_indices within bounds and unique
    cleaned_jobs = []
    for j in jobs:
        jname = str(j.get("name", "")).strip() or "Job"
        jdesc = str(j.get("description", "")).strip()
        idxs = []
        for idx in (j.get("task_indices", []) or []):
            if isinstance(idx, int) and 0 <= idx < len(tasks_norm):
                idxs.append(idx)
        # If empty, attach any tasks that map to this job via t2j
        if not idxs:
            idxs = [i for i, nm in enumerate(t2j) if nm == jname]
        # Deduplicate and sort
        idxs = sorted(set(idxs))
        cleaned_jobs.append({"name": jname, "description": jdesc, "task_indices": idxs})

    out = {"jobs": cleaned_jobs, "task_to_job": t2j}
    _JOBS_CACHE[hk] = out
    if len(_JOBS_CACHE) > 64:
        _JOBS_CACHE.clear()
    # return deep copy to isolate callers
    return json.loads(json.dumps(out))


def generate_functions_for_jobs_batch(
    process_name: str,
    process_description: str,
    tasks: list[dict],
    jobs: list[dict],
    approx_words: int = 350,
) -> dict:
    """Group jobs (roles) into FUNCTIONS/DEPARTMENTS with ~350-word descriptions in one Gemini call.

    Inputs:
      - process_name/description: provide business context
      - tasks: list of {"name", "description"} used as context (optional but recommended)
      - jobs: list of {"name", "description", "task_indices": [int,...]}
      - approx_words: target words per function description (~350)

    Returns dict:
      {
        "functions": [ {"name": str, "description": str, "job_indices": [int,...]} , ...],
        "job_to_function": [str, ...]  # length == len(jobs)
      }
    """
    pname = re.sub(r"\s+", " ", (process_name or "")).strip()
    pdesc = re.sub(r"\s+", " ", (process_description or "")).strip()
    tasks_norm = [
        {"name": re.sub(r"\s+", " ", (t.get("name", "") or "")).strip(),
         "description": re.sub(r"\s+", " ", (t.get("description", "") or "")).strip()}
        for t in (tasks or [])
    ]
    jobs_norm = []
    for j in (jobs or []):
        nm = re.sub(r"\s+", " ", (j.get("name", "") or "")).strip()
        desc = re.sub(r"\s+", " ", (j.get("description", "") or "")).strip()
        idxs = []
        for idx in (j.get("task_indices", []) or []):
            if isinstance(idx, int):
                idxs.append(idx)
        jobs_norm.append({"name": nm, "description": desc, "job_indices": [], "task_indices": idxs})

    key_src = json.dumps({"pn": pname, "pd": pdesc, "ts": tasks_norm, "js": jobs_norm, "w": int(approx_words)}, sort_keys=True)
    hk = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:24]
    cached = _FUNCS_CACHE.get(hk)
    if cached is not None:
        return json.loads(json.dumps(cached))

    tasks_json = json.dumps(
        [{"index": i, "name": t.get("name", ""), "description": t.get("description", "")} for i, t in enumerate(tasks_norm)],
        ensure_ascii=False,
    )
    jobs_json = json.dumps(
        [{"index": i, "name": j.get("name", ""), "description": j.get("description", ""), "task_indices": j.get("task_indices", [])} for i, j in enumerate(jobs_norm)],
        ensure_ascii=False,
    )

    prompt = f"""
You are given a business process with tasks and jobs (roles). Your task is to group jobs into FUNCTIONS/DEPARTMENTS.

Process Name: {pname}
Process Description: {pdesc}

Tasks (indices):
{tasks_json}

Jobs (indices):
{jobs_json}

Requirements for function names:
- Function names MUST be departmental or functional area names (e.g., "Insurance Authorization", "Billing and Claims", "Revenue Cycle Management", "Health Information Management", "Finance", "Patient Access").
- DO NOT use human role titles (e.g., Specialist, Analyst, Manager, Coordinator, Technician, Officer, Supervisor).
- Prefer concise Title Case names. Avoid slashes and conjunction chains.

Rules:
- Assign every job to exactly one function. A function may include multiple jobs.
- Create the minimum sensible number of functions with coherent responsibilities.
- For each function, write one professional paragraph of approximately {approx_words} words (acceptable range 330–370), no headings or lists.
- Return STRICT JSON only, no commentary.

Output JSON schema:
{{
  "functions": [
    {{"name": "<function/department name>", "description": "<~{approx_words}-word paragraph>", "job_indices": [<int>, ...]}} , ...
  ],
  "job_to_function": ["<function name for job 0>", "<function name for job 1>", ...]
}}
"""

    resp = _gen_content(prompt)
    text = (resp.text or "").strip()
    data = None
    try:
        data = json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(text[start:end+1])
        except Exception:
            data = None

    if not isinstance(data, dict):
        data = {"functions": [], "job_to_function": []}

    funcs = data.get("functions") or []
    j2f = data.get("job_to_function") or []

    # Normalize function names away from role titles if leaked
    role_words = r"coordinator|specialist|analyst|manager|supervisor|technician|administrator|officer|lead|director|engineer|consultant|assistant|associate"

    def _title_case(x: str) -> str:
        parts = re.split(r"\s+", (x or "").strip())
        return " ".join(p.capitalize() for p in parts if p)

    def _functionalize(name: str) -> str:
        base = (name or "").strip()
        low = base.lower()
        if re.search(role_words, low):
            # Heuristics from keywords
            if re.search(r"\bbilling|claim", low):
                return "Billing and Claims"
            if re.search(r"\binsurance|authorization|verification", low):
                return "Insurance Authorization"
            if re.search(r"\brevenue cycle|ar |accounts receivable", low):
                return "Revenue Cycle Management"
            if re.search(r"\bcoding|him|medical records|documentation", low):
                return "Health Information Management"
            if re.search(r"\bfinance|financial|report", low):
                return "Finance"
            if re.search(r"\bscheduling|registration|admission|patient access", low):
                return "Patient Access"
            return "Operations"
        # if already functional, just title case
        return _title_case(base)

    rename = {}
    for f in funcs:
        old = str(f.get("name", "")).strip()
        new = _functionalize(old)
        f["name"] = new
        if old and old != new:
            rename[old] = new
    if j2f:
        j2f = [rename.get(str(x).strip(), str(x).strip()) for x in j2f]

    # Repair mapping if needed using job_indices from functions
    if len(j2f) != len(jobs_norm):
        mapping = [None] * len(jobs_norm)
        for f in funcs:
            fname = str(f.get("name", "")).strip()
            for idx in f.get("job_indices", []) or []:
                if isinstance(idx, int) and 0 <= idx < len(mapping):
                    mapping[idx] = fname
        for i in range(len(mapping)):
            if not mapping[i]:
                # create function per orphaned job
                fname = _functionalize(jobs_norm[i].get("name") or "Operations")
                funcs.append({"name": fname, "description": "", "job_indices": [i]})
                mapping[i] = fname
        j2f = mapping

    # Clean job_indices sets
    cleaned_funcs = []
    for f in funcs:
        fname = str(f.get("name", "")).strip() or "Operations"
        fdesc = str(f.get("description", "")).strip()
        idxs = []
        for idx in (f.get("job_indices", []) or []):
            if isinstance(idx, int) and 0 <= idx < len(jobs_norm):
                idxs.append(idx)
        idxs = sorted(set(idxs))
        cleaned_funcs.append({"name": fname, "description": fdesc, "job_indices": idxs})

    out = {"functions": cleaned_funcs, "job_to_function": j2f}
    _FUNCS_CACHE[hk] = out
    if len(_FUNCS_CACHE) > 64:
        _FUNCS_CACHE.clear()
    return json.loads(json.dumps(out))

def generate_process_name(query: str) -> str:
    """Generate a concise, professional process name from a user query using Gemini.

    Returns a cleaned single-line title. Falls back to a short form of the query if generation fails or returns empty.
    """
    prompt = f"""
Given the user's request below, produce a concise, professional process name (3–6 words) that best describes the recurring operational process being requested.

Rules:
- Use Title Case.
- Do not include quotes or surrounding punctuation.
- Avoid adding words like 'Process' or 'Workflow' unless essential.
- Return only the name text.

Request: "{query}"
"""
    resp = _gen_content(prompt)
    name = (resp.text or "").strip()
    # Clean quotes/newlines/punctuation at ends and collapse whitespace
    name = name.strip().strip('"').strip("'")
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[.。]+$", "", name).strip()
    if not name:
        name = (query[:40] + "...") if len(query) > 40 else query
    return name

def generate_tasks_list(process_description, max_tasks: int = 10):
    # Cache key based on description hash and max_tasks
    desc_norm = re.sub(r"\s+", " ", process_description or "").strip()
    hk = hashlib.sha256(desc_norm.encode("utf-8")).hexdigest()[:16]
    cache_key = (hk, int(max_tasks))
    cached = _TASKS_LIST_CACHE.get(cache_key)
    if cached is not None:
        kept_c, removed_c = cached
        # return shallow copies to avoid external mutation
        return kept_c[:], removed_c[:]
    prompt = f"""
Read the following process description and produce a numbered list of the operational tasks that MUST be performed each time this process is run.
Do NOT include one-time activities such as designing, developing, setting up systems, or strategic planning. Focus on repeatable, operational tasks (for example: 'Collect documents', 'Approve request', 'Notify stakeholders').
Ensure the tasks collectively represent the full set required to complete the process; when all tasks are completed, the process is complete.
Return each task on its own line, prefixed with its number. Produce up to {max_tasks} tasks.

Process description:
{process_description}
"""
    resp = _gen_content(prompt)
    text = resp.text.strip()
    # parse lines that start with a number
    tasks = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # remove numbering if present
        parts = line.split(".", 1)
        if parts[0].isdigit() and len(parts) > 1:
            task = parts[1].strip()
        else:
            # fallback take the whole line
            task = line
        tasks.append(task)
    # filter likely one-time/design/setup tasks using simple heuristics
    one_time_phrases = [
        'design', 'build', 'implement', 'set up', 'setup', 'create system', 'establish', 'initial', 'once', 'one-time',
        'develop', 'deploy', 'configure', 'plan', 'strategy', 'architect', 'installation'
    ]
    kept = []
    removed = []
    for t in tasks:
        tl = t.lower()
        if any(p in tl for p in one_time_phrases):
            removed.append(t)
        else:
            kept.append(t)

    if removed:
        print("\nFiltered out tasks that appear to be one-time/setup or design activities (these are removed by default):")
        for r in removed:
            print(f" - {r}")
        print("You can re-include them during task review if desired.")

    # store in cache (bound size)
    _TASKS_LIST_CACHE[cache_key] = (kept, removed)
    if len(_TASKS_LIST_CACHE) > 256:
        _TASKS_LIST_CACHE.clear()
    return kept, removed


def generate_task_name(task_line: str) -> str:
    """Generate a short 3–4 word task name from a task line."""
    key = re.sub(r"\s+", " ", task_line or "").strip().lower()
    cached = _NAME_CACHE.get(key)
    if cached is not None:
        return cached
    prompt = f"""
From the following task description, produce a concise task name of 3 to 4 words in Title Case that clearly expresses what the task is. Do not include quotes or punctuation. Return only the name text.

Task: "{task_line}"
"""
    resp = _gen_content(prompt)
    name = (resp.text or "").strip().strip('"').strip("'")
    name = re.sub(r"\s+", " ", name)
    # Remove trailing periods or punctuation
    name = re.sub(r"[.。!?]+$", "", name).strip()
    _NAME_CACHE[key] = name
    if len(_NAME_CACHE) > 512:
        _NAME_CACHE.clear()
    return name


def generate_task_description(task_name: str, process_description: str, min_words: int = 600):
    """Generate a detailed single-paragraph description for a task (min words, no headings or line breaks)."""
    prompt = f"""
Write a professional description for the task "{task_name}" within the following business process context. Describe what the task is, its purpose, inputs and outputs conceptually, dependencies, stakeholders involved, quality considerations, and how it contributes to completing the process. Do not provide step-by-step instructions.

Process context:
{process_description}

Requirements:
- Minimum {min_words} words (longer is acceptable).
- Single paragraph only.
- No headings, subheadings, bullets, or numbered lists.
- No line breaks or blank lines; output must be one continuous paragraph.
- Return only the plain text description (no quotes or surrounding punctuation).
"""
    gemini = _get_gemini()
    resp = gemini.generate_content(prompt)
    return resp.text


def generate_task_detail(task_name, process_description):
    prompt = f"""
    Create a detailed description for the task titled: "{task_name}" which belongs to the following process:
    {process_description}

    Requirements (output only):
    - Produce approximately 600 words total as plain text.
    - Within the output include a subsection headed exactly 'Detailed description:' containing approximately 250 words that describe the task in detail (this subsection counts toward the 600 words).
    - After the detailed description, include plain, unformatted headings (do not bold) for:
        Inputs:
        Sub-tasks:
        Outputs:
        Under each heading provide a short bullet list (use '-' for bullets) of the relevant items.
    - Do not include implementation instructions beyond the detailed description subsection; do not include code blocks or extra commentary.

    Return only the resulting text.

Return the full ~600-word paragraph as plain text.
"""
    gemini = _get_gemini()
    resp = gemini.generate_content(prompt)
    return resp.text


def suggest_job_for_task(task_name, task_detail):
    prompt = f"""
Suggest an appropriate job title (one short job name) that would be responsible for this task.
Task: {task_name}
Task detail (short): {task_detail[:400]}
Return only a short job title.
"""
    resp = _gen_content(prompt)
    return resp.text.strip().splitlines()[0]


def suggest_function_for_job(job_name, task_list_for_job):
    prompt = f"""
For the job titled '{job_name}', suggest a single Python function name and a one-line description of what that function would do to perform or trigger the job's responsibilities for these tasks:
{chr(10).join(['- '+t for t in task_list_for_job])}

Return in the format: function_name: short description
Use snake_case for function_name.
"""
    gemini = _get_gemini()
    resp = gemini.generate_content(prompt)
    return resp.text.strip().splitlines()[0]

# Main interaction loop
def main():
    while True:
        query = input("\nEnter your process query: ").strip()
        if not query:
            continue

        t0 = time.time()
        candidates = retrieve_processes(query, top_k=3)
        t1 = time.time()

        # Select the best candidate by cosine similarity
        best = candidates[0] if candidates else None

        # threshold on similarity to decide reuse vs generate
        threshold = 0.33
        if best and best.get('sim', 0.0) >= threshold:
            pid = best.get('id')
            pname = best.get('name')
            pdesc = best.get('description')
            sim = best.get('sim', 0.0)
            print(f"\nIdentified Process -> ID: {pid} | Name: {pname} (sim={sim:.2f} | Fetched in {t1 - t0:.2f}s)\n")
        else:
            print(f"\nNo similar process found. (Checked in {t1 - t0:.2f}s)")
            choice = input("Generate new process with Gemini? (yes/no): ").strip().lower()
            if choice == "yes":
                print("Generating process description (≈350 words)...")
                proc_desc = generate_process_description(query, approx_words=350)
                print("\nGenerated Process Description:\n")
                print(proc_desc)
                confirm = input("\nApprove this process description? (yes/no): ").strip().lower()
                if confirm != "yes":
                    print("Process discarded.")
                else:
                    # proceed to generate tasks
                    print("Generating task list (operational tasks only)...")
                    tasks_kept, tasks_removed = generate_tasks_list(proc_desc, max_tasks=12)
                    tasks = tasks_kept[:]  # start with kept tasks
                    print(f"\nGenerated {len(tasks)} operational tasks:\n")
                    for i, t in enumerate(tasks, start=1):
                        print(f"{i}. {t}")

                    if tasks_removed:
                        print("\nThere were additional candidate tasks identified as one-time/setup/design (not included):")
                        for r in tasks_removed:
                            print(f" - {r}")
                        include = input("\nInclude any of these in the task list? Enter numbers separated by commas corresponding to the list above, or press enter to skip: ").strip()
                        if include:
                            # allow user to enter indices relative to removed list
                            chosen = {int(x.strip()) for x in include.split(',') if x.strip().isdigit()}
                            for c in sorted(chosen):
                                if 1 <= c <= len(tasks_removed):
                                    tasks.append(tasks_removed[c-1])
                        print("\nFinal task list:")
                        for i, t in enumerate(tasks, start=1):
                            print(f"{i}. {t}")

                    # ask user to approve tasks selection and optionally remove/rename
                    edit = input("\nEdit tasks before generating details? (yes/no): ").strip().lower()
                    if edit == "yes":
                        print("Enter task numbers to remove separated by commas (or press enter to keep):")
                        rem = input().strip()
                        if rem:
                            to_remove = {int(x.strip()) for x in rem.split(',') if x.strip().isdigit()}
                            tasks = [t for i, t in enumerate(tasks, start=1) if i not in to_remove]
                        # allow renaming
                        print("If you want to rename a task, enter lines in format 'n:New Task Name' or press enter to skip:")
                        while True:
                            line = input().strip()
                            if not line:
                                break
                            if ':' in line:
                                a, b = line.split(':', 1)
                                if a.strip().isdigit():
                                    idx = int(a.strip()) - 1
                                    if 0 <= idx < len(tasks):
                                        tasks[idx] = b.strip()
                        print("Updated tasks:")
                        for i, t in enumerate(tasks, start=1):
                            print(f"{i}. {t}")

                    # for each task generate detail (600 words) and ask approval
                    approved_tasks = []
                    for t in tasks:
                        print(f"\nGenerating detail for task: {t} (≈600 words)...")
                        detail = generate_task_detail(t, proc_desc)
                        print(f"\nTask detail for '{t}':\n")
                        print(detail[:4000])
                        ok = input("\nApprove this task detail? (yes/no): ").strip().lower()
                        if ok == 'yes':
                            approved_tasks.append({'task': t, 'detail': detail})
                        else:
                            print(f"Task '{t}' skipped by user.")

                    if not approved_tasks:
                        print("No tasks approved. Process creation aborted.")
                    else:
                        # map tasks to jobs
                        job_map = {}
                        for t in approved_tasks:
                            suggested = suggest_job_for_task(t['task'], t['detail'])
                            print(f"\nSuggested job for task '{t['task']}': {suggested}")
                            job = input(f"Enter job to assign (or press enter to accept '{suggested}'): ").strip()
                            if not job:
                                job = suggested
                            job_map.setdefault(job, []).append(t)

                        # map jobs to functions
                        function_map = {}
                        for job, tasks_for_job in job_map.items():
                            sugg = suggest_function_for_job(job, [t['task'] for t in tasks_for_job])
                            print(f"\nSuggested function mapping for job '{job}': {sugg}")
                            fn = input(f"Enter function mapping (or press enter to accept): ").strip()
                            if not fn:
                                fn = sugg
                            function_map[job] = fn

                        # final confirmation and store
                        proc_name = input("\nEnter a name for this process (or press enter to use a short form of the query): ").strip()
                        if not proc_name:
                            proc_name = (query[:40] + '...') if len(query) > 40 else query

                        pid = get_next_process_id()
                        # store process and tasks as one Redis object under process:<id>
                        store_obj = {
                            'id': pid,
                            'name': proc_name,
                            'description': proc_desc,
                            'tasks': approved_tasks,
                            'job_map': job_map,
                            'function_map': function_map
                        }
                        # compute vector for process description
                        # Embed the process using both its name and description for better retrieval
                        text_for_embedding = f"{proc_name}. {proc_desc}"
                        vec = model.encode(text_for_embedding).tolist()
                        store_obj['vector'] = vec
                        redis.set(f"process:{pid}", json.dumps(store_obj))
                        print(f"\nProcess created and stored with id={pid} name='{proc_name}'.")

        again = input("\nSearch another process? (yes/no): ").strip().lower()
        if again != "yes":
            print("\nExiting. Goodbye!")
            break

# Run the program
if __name__ == "__main__":
    main()
