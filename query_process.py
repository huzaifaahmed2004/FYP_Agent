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

# Load environment variables
load_dotenv()

# Configure Google Generative AI only if an API key is provided.
_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if _GOOGLE_API_KEY:
    genai.configure(api_key=_GOOGLE_API_KEY)
_gemini_model = None  # lazy init when generation functions are called

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
    gemini = _get_gemini()
    response = gemini.generate_content(prompt)
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


def generate_process_description(query, approx_words: int = 350):
    prompt = f"""
Generate a clear, professional organizational process description for the following request.
Goal: produce approximately {approx_words} words (aim for close to the target).

Request: "{query}"

Important: the description should state WHAT the process is — its purpose, scope, expected outcomes, stakeholders, and constraints. Do NOT include instructions or procedural steps about how to carry out the process.

Return only the description text.
"""
    gemini = _get_gemini()
    resp = gemini.generate_content(prompt)
    return resp.text


def generate_tasks_list(process_description, max_tasks: int = 10):
    prompt = f"""
Read the following process description and produce a numbered list of the operational tasks that MUST be performed each time this process is run.
Do NOT include one-time activities such as designing, developing, setting up systems, or strategic planning. Focus on repeatable, operational tasks (for example: 'Collect documents', 'Approve request', 'Notify stakeholders').
Return each task on its own line, prefixed with its number. Produce up to {max_tasks} tasks.

Process description:
{process_description}
"""
    gemini = _get_gemini()
    resp = gemini.generate_content(prompt)
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

    return kept, removed


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
    gemini = _get_gemini()
    resp = gemini.generate_content(prompt)
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
