from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, List, Dict
import re

# Import the retrieval function and ensure heavy resources are available once per process
from query_process import (
    retrieve_processes,
    generate_process_description,
    generate_tasks_list,
    generate_task_name,
    generate_task_description,
    generate_task_descriptions_batch,
    generate_jobs_for_tasks_batch,
    generate_functions_for_jobs_batch,
)

app = FastAPI(title="Process Identification API", version="1.0.0")

# No in-memory sessions; stateless simple generation

class IdentifyRequest(BaseModel):
    query: str

class Candidate(BaseModel):
    id: Optional[int]
    name: Optional[str]
    description: Optional[str]
    sim: float

class IdentifyResponse(BaseModel):
    best: Optional[Candidate]
    message: Optional[str]  # Added to provide feedback like "No process found"

def is_valid_query(query: str) -> bool:
    """Check if the query is meaningful (not too short or generic)."""
    query = query.strip()
    if len(query) < 3:  # Reject very short queries
        return False
    # Reject common meaningless words/phrases
    meaningless = {"hello", "hi", "test", "hey", "abc", "123"}
    if query.lower() in meaningless:
        return False
    # Basic check for non-alphanumeric spam or overly simplistic input
    if not re.search(r"[a-zA-Z0-9].*[a-zA-Z0-9]", query):
        return False
    return True

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/identify", response_model=IdentifyResponse)
def identify_process(payload: IdentifyRequest) -> Any:
    q = (payload.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    
    if not is_valid_query(q):
        raise HTTPException(status_code=400, detail="Query is too short or meaningless")

    try:
        # Retrieve only the top candidate
        cands = retrieve_processes(q, top_k=1)
        # Convert numpy types to plain Python types
        serializable = [
            {
                "id": c.get("id"),
                "name": c.get("name"),
                "description": c.get("description"),
                "sim": float(c.get("sim", 0.0)),
            }
            for c in cands
        ]
        best = serializable[0] if serializable else None
        
        # Apply similarity threshold
        SIMILARITY_THRESHOLD = 0.33
        if best and best.get("sim", 0.0) >= SIMILARITY_THRESHOLD:
            return {"best": best, "message": None}
        else:
            return {"best": None, "message": "No process found with sufficient similarity"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identification failed: {e}")


# ---------------------- Simple Process Generation ----------------------
class CreateSimpleRequest(BaseModel):
    query: str
    name: Optional[str] = None
    description: Optional[str] = None


class CreateSimpleResponse(BaseModel):
    name: str
    description: str
    tasks: Optional[List[str]] = None


class ProcessCreationRequest(BaseModel):
    query: str


class ProcessCreationResponse(BaseModel):
    name: str
    description: str


@app.post("/processcreation", response_model=ProcessCreationResponse)
def create_process_simple(payload: ProcessCreationRequest) -> Any:
    q = (payload.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="'query' is required")
    if not is_valid_query(q):
        raise HTTPException(status_code=400, detail="Query is too short or meaningless")

    # Derive a concise name from the query (strip filler like "I want", "process of/for", etc.)
    def _derive_name(text: str) -> str:
        s = text.strip()
        lower = s.lower()
        # Remove leading polite/request patterns
        patterns = [
            r"^(please\s+)?(give|provide)\s+(me\s+)?(a\s+|the\s+)?",
            r"^(i|we)\s+(want|need|would\s+like|require|am\s+looking\s+for)\s+(a\s+|the\s+)?",
            r"^(create|make|build|generate|draft|design|prepare)\s+(a\s+|the\s+)?(process|workflow)\s+(for|of|to)\s+",
            r"^(create|make|build|generate|draft|design|prepare)\s+(a\s+|the\s+)?",
            r"^(how\s+to\s+)",
            r"^(process|workflow)\s+(for|of|to|on|about)\s+",
        ]
        tmp = lower
        for pat in patterns:
            tmp = re.sub(pat, "", tmp, flags=re.IGNORECASE)
        # Trim common trailing filler
        tmp = re.sub(r"\s+(please|thanks|thank\s+you)\s*$", "", tmp, flags=re.IGNORECASE)
        # Tokenize and remove leading stopwords
        tokens = re.findall(r"[A-Za-z0-9&/-]+", tmp)
        stop = {"the", "a", "an", "of", "for", "to", "in", "on", "about"}
        while tokens and tokens[0].lower() in stop:
            tokens.pop(0)
        if not tokens:
            tokens = re.findall(r"[A-Za-z0-9&/-]+", lower)[:4]
        # Keep first 3–6 tokens, title-case smartly
        keep = tokens[:5] if len(tokens) > 3 else tokens
        def _cap(w: str) -> str:
            return w if w.isupper() and len(w) <= 5 else w.capitalize()
        title = " ".join(_cap(w) for w in keep)
        # Ensure it ends with "Process" if it looks too generic
        if title and not re.search(r"\b(process|workflow)\b", title, re.IGNORECASE):
            return title
        return title or "Untitled Process"

    # Normalize and validation helpers
    def _normalize_ws(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    def _is_provided(val: Optional[str], *, min_len: int = 1, treat_string_placeholder: bool = True) -> bool:
        if val is None:
            return False
        norm = _normalize_ws(val)
        if not norm:
            return False
        if treat_string_placeholder and norm.lower() == "string":
            return False
        return len(norm) >= min_len

    # Pick name solely from query for this endpoint
    pname = _derive_name(q)

    # Generate description using existing generator and steer towards ~350 words
    def _word_count(s: str) -> int:
        return len(re.findall(r"\b\w+\b", s or ""))

    def _trim_to_words(s: str, target: int) -> str:
        words = re.findall(r"\S+", s or "")
        if len(words) <= target:
            return " ".join(words)
        out = " ".join(words[:target])
        # ensure sentence ending
        return out.rstrip(" ,;:-") + "."

    def _strip_headings(s: str) -> str:
        # Remove common heading labels like 'Purpose:', 'Scope:', etc. anywhere they appear
        labels = [
            "Purpose", "Scope", "Objectives", "Stakeholders", "Constraints",
            "Overview", "Outcomes", "Benefits", "Risks", "Process Description",
            "Introduction", "Summary", "Goals", "Deliverables"
        ]
        pattern = r"\b(?:" + "|".join(re.escape(l) for l in labels) + r")\s*[:\-]\s*"
        return re.sub(pattern, "", s)

    try:
        # Always generate description from query for this endpoint
        target = 400
        desc = generate_process_description(q, approx_words=target)
        # Normalize whitespace and strip heading labels
        desc = _strip_headings(_normalize_ws(desc))
        wc = _word_count(desc)

        # Minimal retries: only 1 retry and only if initial text is very short (<350 words)
        attempts = 0
        initial_short = wc < 350
        while wc < 400 and attempts < 1 and initial_short:
            target += 40
            alt = generate_process_description(q, approx_words=target)
            alt = _strip_headings(_normalize_ws(alt))
            if _word_count(alt) > wc:
                desc = alt
                wc = _word_count(desc)
            attempts += 1

        # If still short, conservatively pad with generic neutral sentence until >=400, then trim
        if wc < 400:
            filler = " This description clarifies the process and maintains a professional, neutral, and operational tone."
            while wc < 400:
                desc += filler
                desc = _strip_headings(_normalize_ws(desc))
                wc = _word_count(desc)

        # Trim to exactly 400 words
        if wc >= 400:
            desc = _trim_to_words(desc, 400)
    except RuntimeError as e:
        # likely missing GOOGLE_API_KEY
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate description: {e}")

    return {"name": _normalize_ws(pname), "description": _normalize_ws(desc)}

# ---------------------- Tasks Generation from Provided Process ----------------------
class ProcessTasksRequest(BaseModel):
    name: str
    description: str


class TaskItem(BaseModel):
    name: str
    description: str


class ProcessTasksResponse(BaseModel):
    task_names: List[str]
    tasks: List[TaskItem]


@app.post("/processtasks", response_model=ProcessTasksResponse)
def create_process_tasks(payload: ProcessTasksRequest) -> Any:
    # Helpers (local to keep module tidy)
    def _normalize_ws(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    def _is_provided(val: Optional[str], *, min_len: int = 1) -> bool:
        if val is None:
            return False
        norm = _normalize_ws(val)
        if not norm or norm.lower() == "string":
            return False
        return len(norm) >= min_len

    def _strip_headings(s: str) -> str:
        labels = [
            "Purpose", "Scope", "Objectives", "Stakeholders", "Constraints",
            "Overview", "Outcomes", "Benefits", "Risks", "Process Description",
            "Introduction", "Summary", "Goals", "Deliverables"
        ]
        pattern = r"\b(?:" + "|".join(re.escape(l) for l in labels) + r")\s*[:\-]\s*"
        return re.sub(pattern, "", s)

    def _word_count(s: str) -> int:
        return len(re.findall(r"\b\w+\b", s or ""))

    def _trim_to_words(s: str, target: int) -> str:
        words = re.findall(r"\S+", s or "")
        if len(words) <= target:
            return " ".join(words)
        out = " ".join(words[:target])
        return out.rstrip(" ,;:-") + "."

    def _ensure_task_name_3_4_words(src: str) -> str:
        # Force Title Case and 3–4 words
        words = re.findall(r"[A-Za-z0-9&/-]+", src)
        if not words:
            return "Task Name"
        # Title-case smartly
        titled = [w if (w.isupper() and len(w) <= 5) else w.capitalize() for w in words]
        if len(titled) >= 4:
            keep = titled[:4]
        elif len(titled) == 3:
            keep = titled
        else:
            # pad from remaining source tokens if available
            pad_src = titled + ["Task", "Step"]
            keep = pad_src[:3]
        return " ".join(keep)

    def _sanitize_text(s: str) -> str:
        # Allow only letters, numbers, spaces, and basic punctuation: period and comma
        s = re.sub(r"[^A-Za-z0-9 ,.]", " ", s or "")
        return _normalize_ws(s)

    def _dedupe_sentences(paragraph: str) -> str:
        # Split by sentence endings, dedupe case-insensitively, and rejoin with single spaces
        text = paragraph or ""
        # Ensure standard periods, replace multiple punctuation with period
        text = re.sub(r"[!?]+", ".", text)
        parts = re.split(r"(?<=[.])\s+", text)
        seen = set()
        out = []
        for p in parts:
            sent = p.strip()
            if not sent:
                continue
            key = re.sub(r"\s+", " ", sent.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(sent)
        result = " ".join(out)
        # Normalize spaces and ensure ends with a period
        result = _normalize_ws(result)
        if result and not re.search(r"[.]$", result):
            result += "."
        return result

    # Validate inputs
    if not _is_provided(payload.name, min_len=2):
        raise HTTPException(status_code=400, detail="'name' is required")
    if not _is_provided(payload.description, min_len=50):
        raise HTTPException(status_code=400, detail="'description' is required and must be substantive")

    pname = _normalize_ws(payload.name)
    pdesc = _strip_headings(_normalize_ws(payload.description))

    # Generate candidate tasks from provided description
    try:
        kept, _removed = generate_tasks_list(pdesc, max_tasks=12)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate tasks: {e}")

    # Post-filter for occasional/exceptional items
    occasional_terms = [
        "return", "refund", "replacement", "recall", "chargeback",
        "incident", "escalation", "breach", "outage", "dispute",
        "exception", "non-routine", "one-off", "ad hoc", "adhoc",
        "complaint handling", "investigation", "root cause", "rca",
        "warranty claim", "damage claim"
    ]
    filtered_lines: List[str] = []
    for t in kept:
        tl = t.lower()
        if any(term in tl for term in occasional_terms):
            continue
        filtered_lines.append(t)

    if not filtered_lines:
        # Retry once with a slightly higher cap
        try:
            retry_kept, _ = generate_tasks_list(pdesc, max_tasks=15)
            for t in retry_kept:
                tl = t.lower()
                if any(term in tl for term in occasional_terms):
                    continue
                filtered_lines.append(t)
        except Exception:
            pass

    # Limit to at most 6 tasks
    task_lines = filtered_lines[:6]
    if not task_lines:
        raise HTTPException(status_code=422, detail="No essential operational tasks could be generated; please refine the process description.")

    # Generate 3–4 word names first, then batch-generate descriptions
    results: List[Dict[str, str]] = []
    task_names: List[str] = []
    for line in task_lines:
        try:
            gen_name = generate_task_name(line)
        except Exception:
            gen_name = ""
        gen_name = _ensure_task_name_3_4_words(gen_name or line)
        gen_name = _sanitize_text(gen_name)
        task_names.append(gen_name)

    # Batch generate descriptions for all tasks in one Gemini call
    try:
        batch_descs = generate_task_descriptions_batch(task_names, pdesc, min_words=650)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate task descriptions: {e}")

    # First pass: sanitize, dedupe, and detect very short items (<500 words)
    processed: List[str] = []
    word_counts: List[int] = []
    short_idxs: List[int] = []
    for i, raw in enumerate(batch_descs):
        tdesc = _strip_headings(_normalize_ws(raw or ""))
        tdesc = _sanitize_text(tdesc)
        tdesc = _dedupe_sentences(tdesc)
        wc = _word_count(tdesc)
        processed.append(tdesc)
        word_counts.append(wc)
        if wc < 500:
            short_idxs.append(i)

    # Optional single retry batch only for very short ones
    if short_idxs:
        try:
            retry_names = [task_names[i] for i in short_idxs]
            retry_descs = generate_task_descriptions_batch(retry_names, pdesc, min_words=650)
            for j, idx in enumerate(short_idxs):
                cand = retry_descs[j] if j < len(retry_descs) else ""
                cand = _strip_headings(_normalize_ws(cand))
                cand = _sanitize_text(cand)
                cand = _dedupe_sentences(cand)
                wc2 = _word_count(cand)
                if wc2 > word_counts[idx]:
                    processed[idx] = cand
                    word_counts[idx] = wc2
        except Exception:
            pass

    # Ensure each task description is around 600 words without repetitive phrasing
    for i, name in enumerate(task_names):
        tdesc = processed[i]
        wc = word_counts[i]
        TARGET = 600
        TOL = 20
        MIN_TARGET = TARGET - TOL  # 580
        MAX_TARGET = TARGET + TOL  # 620

        if wc < MIN_TARGET:
            # Add varied, non-repeating sentences that avoid formulaic phrasing
            stems = [
                "It details",
                "It outlines",
                "It specifies",
                "It addresses",
                "It highlights",
                "It explains",
                "It defines",
                "It notes",
                "It covers",
                "It describes",
                "It emphasizes",
                "It summarizes",
            ]
            topics = [
                "stakeholders and accountabilities",
                "required inputs and expected outputs",
                "dependencies, handoffs, and upstream/downstream interfaces",
                "operational risks, controls, and safeguards",
                "service levels, timing, and quality criteria",
                "records kept and evidence retained",
                "tools, systems, and integration touchpoints",
                "regulatory, privacy, and compliance obligations",
                "communication expectations and notifications",
                "monitoring, metrics, and performance indicators",
                "exceptions, edge cases, and escalation paths",
                "assumptions, constraints, and preconditions",
                "authorizations, approvals, and segregation of duties",
                "collaboration with adjacent teams and functions",
                "change management and release considerations",
                "documentation standards and audit readiness",
                "information security and data quality practices",
                "training, readiness, and knowledge transfer",
                "continuous improvement and feedback loops",
                "customer impact and experience considerations",
                "operational boundaries and scope limits",
                "acceptance criteria and definition of done",
                "reporting and stakeholder visibility requirements",
            ]
            endings = [
                " to support consistent and reliable execution.",
                " to drive accountability and traceability.",
                " to align expectations and outcomes.",
                " to reduce ambiguity and operational risk.",
            ]
            s_idx = 0
            t_idx = 0
            e_idx = 0
            while wc < MIN_TARGET and t_idx < len(topics):
                stem = stems[s_idx % len(stems)]
                ending = endings[e_idx % len(endings)]
                sentence = f" {stem} {topics[t_idx]}{ending}"
                tdesc += sentence
                tdesc = _strip_headings(_normalize_ws(tdesc))
                tdesc = _sanitize_text(tdesc)
                tdesc = _dedupe_sentences(tdesc)
                wc = _word_count(tdesc)
                s_idx += 1
                t_idx += 1
                e_idx += 1
            # If still short, add one neutral concluding sentence
            if wc < MIN_TARGET:
                tdesc += (
                    " The description remains intentionally high level and focuses on context, interfaces, controls, and expected outcomes so teams can execute consistently."
                )
                tdesc = _strip_headings(_normalize_ws(tdesc))
                tdesc = _sanitize_text(tdesc)
                tdesc = _dedupe_sentences(tdesc)
                wc = _word_count(tdesc)

        # If we overshoot the soft cap, trim back to ~600 words
        if wc > MAX_TARGET:
            tdesc = _trim_to_words(tdesc, TARGET)
            wc = _word_count(tdesc)
        # Final sanitize and collect
        tdesc = _sanitize_text(tdesc)
        results.append({"name": name, "description": tdesc})

    # Return task_names first, then full task objects
    return {"task_names": task_names, "tasks": results}

# ---------------------- Jobs Generation from Provided Tasks ----------------------
class TaskInput(BaseModel):
    name: str
    description: str


class TaskJobsRequest(BaseModel):
    process_name: str
    process_description: str
    tasks: List[TaskInput]


class JobItem(BaseModel):
    name: str
    description: str
    task_indices: List[int]


class TaskJobsResponse(BaseModel):
    jobs: List[JobItem]


@app.post("/taskjobs", response_model=TaskJobsResponse)
def create_task_jobs(payload: TaskJobsRequest) -> Any:
    # Helpers (local)
    def _normalize_ws(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    def _is_provided(val: str, *, min_len: int = 1) -> bool:
        if val is None:
            return False
        norm = _normalize_ws(val)
        if not norm or norm.lower() == "string":
            return False
        return len(norm) >= min_len

    def _strip_headings(s: str) -> str:
        labels = [
            "Purpose", "Scope", "Objectives", "Stakeholders", "Constraints",
            "Overview", "Outcomes", "Benefits", "Risks", "Process Description",
            "Introduction", "Summary", "Goals", "Deliverables"
        ]
        pattern = r"\b(?:" + "|".join(re.escape(l) for l in labels) + r")\s*[:\-]\s*"
        return re.sub(pattern, "", s)

    def _word_count(s: str) -> int:
        return len(re.findall(r"\b\w+\b", s or ""))

    def _trim_to_words(s: str, target: int) -> str:
        words = re.findall(r"\S+", s or "")
        if len(words) <= target:
            return " ".join(words)
        out = " ".join(words[:target])
        return out.rstrip(" ,;:-") + "."

    def _sanitize_text(s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9 ,.]", " ", s or "")
        return _normalize_ws(s)

    def _dedupe_sentences(paragraph: str) -> str:
        text = paragraph or ""
        text = re.sub(r"[!?]+", ".", text)
        parts = re.split(r"(?<=[.])\s+", text)
        seen = set()
        out = []
        for p in parts:
            sent = p.strip()
            if not sent:
                continue
            key = re.sub(r"\s+", " ", sent.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(sent)
        result = " ".join(out)
        result = _normalize_ws(result)
        if result and not re.search(r"[.]$", result):
            result += "."
        return result

    # Validate inputs
    if not _is_provided(payload.process_name, min_len=2):
        raise HTTPException(status_code=400, detail="'process_name' is required")
    if not _is_provided(payload.process_description, min_len=50):
        raise HTTPException(status_code=400, detail="'process_description' is required and must be substantive")
    if not payload.tasks or not isinstance(payload.tasks, list):
        raise HTTPException(status_code=400, detail="'tasks' must be a non-empty list")
    tasks_clean: List[dict] = []
    for i, t in enumerate(payload.tasks):
        if not _is_provided(t.name, min_len=2):
            raise HTTPException(status_code=400, detail=f"Task at index {i} missing 'name'")
        if not _is_provided(t.description, min_len=30):
            raise HTTPException(status_code=400, detail=f"Task at index {i} missing 'description'")
        tasks_clean.append({"name": _normalize_ws(t.name), "description": _normalize_ws(t.description)})

    # Generate jobs + mapping in one Gemini call
    try:
        raw = generate_jobs_for_tasks_batch(
            _normalize_ws(payload.process_name),
            _strip_headings(_normalize_ws(payload.process_description)),
            tasks_clean,
            approx_words=350,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate jobs: {e}")

    jobs = raw.get("jobs", [])
    task_to_job = raw.get("task_to_job", [])

    # Post-process job descriptions to ~350 words and clean text
    TARGET = 350
    TOL = 20
    MIN_TARGET = TARGET - TOL  # 330
    MAX_TARGET = TARGET + TOL  # 370
    cleaned_jobs: List[JobItem] = []
    for j in jobs:
        jname = _sanitize_text(_normalize_ws(j.get("name", "") or "Job"))
        jdesc = _sanitize_text(_strip_headings(_normalize_ws(j.get("description", "") or "")))
        jdesc = _dedupe_sentences(jdesc)
        wc = _word_count(jdesc)
        if wc < MIN_TARGET:
            # add a single neutral sentence, then recompute
            jdesc += " The description focuses on purpose, scope, interfaces, and quality safeguards to guide consistent, accountable operations."
            jdesc = _sanitize_text(_strip_headings(_normalize_ws(jdesc)))
            jdesc = _dedupe_sentences(jdesc)
            wc = _word_count(jdesc)
        if wc > MAX_TARGET:
            jdesc = _trim_to_words(jdesc, TARGET)
        # Indices: ensure integers within bounds
        idxs_in = j.get("task_indices", []) or []
        idxs = []
        for idx in idxs_in:
            if isinstance(idx, int) and 0 <= idx < len(tasks_clean):
                idxs.append(idx)
        idxs = sorted(set(idxs))
        cleaned_jobs.append(JobItem(name=jname, description=jdesc, task_indices=idxs))

    # Ensure mapping covers all tasks
    if len(task_to_job) != len(tasks_clean):
        mapping = [None] * len(tasks_clean)
        for j in cleaned_jobs:
            for idx in j.task_indices:
                if 0 <= idx < len(mapping):
                    mapping[idx] = j.name
        for i in range(len(mapping)):
            if not mapping[i]:
                # create a job per orphaned task
                orphan_name = f"{tasks_clean[i]['name']} Job"
                cleaned_jobs.append(JobItem(name=orphan_name, description="", task_indices=[i]))
                mapping[i] = orphan_name
        task_to_job = mapping  # type: ignore

    return {"jobs": cleaned_jobs}

# ---------------------- Functions/Departments from Provided Jobs ----------------------
class JobInput(BaseModel):
    name: str
    description: str
    task_indices: List[int] = []


class JobFunctionsRequest(BaseModel):
    process_name: str
    process_description: str
    tasks: List[TaskInput]
    jobs: List[JobInput]


class FunctionItem(BaseModel):
    name: str
    description: str
    job_indices: List[int]


class JobFunctionsResponse(BaseModel):
    functions: List[FunctionItem]


@app.post("/jobfunctions", response_model=JobFunctionsResponse)
def create_job_functions(payload: JobFunctionsRequest) -> Any:
    # Local helpers (duplicated for isolation)
    def _normalize_ws(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    def _is_provided(val: str, *, min_len: int = 1) -> bool:
        if val is None:
            return False
        norm = _normalize_ws(val)
        if not norm or norm.lower() == "string":
            return False
        return len(norm) >= min_len

    def _strip_headings(s: str) -> str:
        labels = [
            "Purpose", "Scope", "Objectives", "Stakeholders", "Constraints",
            "Overview", "Outcomes", "Benefits", "Risks", "Process Description",
            "Introduction", "Summary", "Goals", "Deliverables"
        ]
        pattern = r"\b(?:" + "|".join(re.escape(l) for l in labels) + r")\s*[:\-]\s*"
        return re.sub(pattern, "", s)

    def _word_count(s: str) -> int:
        return len(re.findall(r"\b\w+\b", s or ""))

    def _trim_to_words(s: str, target: int) -> str:
        words = re.findall(r"\S+", s or "")
        if len(words) <= target:
            return " ".join(words)
        out = " ".join(words[:target])
        return out.rstrip(" ,;:-") + "."

    def _sanitize_text(s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9 ,.]", " ", s or "")
        return _normalize_ws(s)

    def _dedupe_sentences(paragraph: str) -> str:
        text = paragraph or ""
        text = re.sub(r"[!?]+", ".", text)
        parts = re.split(r"(?<=[.])\s+", text)
        seen = set()
        out = []
        for p in parts:
            sent = p.strip()
            if not sent:
                continue
            key = re.sub(r"\s+", " ", sent.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(sent)
        result = " ".join(out)
        result = _normalize_ws(result)
        if result and not re.search(r"[.]$", result):
            result += "."
        return result

    # Validate inputs
    if not _is_provided(payload.process_name, min_len=2):
        raise HTTPException(status_code=400, detail="'process_name' is required")
    if not _is_provided(payload.process_description, min_len=50):
        raise HTTPException(status_code=400, detail="'process_description' is required and must be substantive")
    if not payload.tasks or not isinstance(payload.tasks, list):
        raise HTTPException(status_code=400, detail="'tasks' must be a non-empty list")
    if not payload.jobs or not isinstance(payload.jobs, list):
        raise HTTPException(status_code=400, detail="'jobs' must be a non-empty list")

    tasks_clean: List[dict] = []
    for i, t in enumerate(payload.tasks):
        if not _is_provided(t.name, min_len=2):
            raise HTTPException(status_code=400, detail=f"Task at index {i} missing 'name'")
        if not _is_provided(t.description, min_len=30):
            raise HTTPException(status_code=400, detail=f"Task at index {i} missing 'description'")
        tasks_clean.append({"name": _normalize_ws(t.name), "description": _normalize_ws(t.description)})

    jobs_clean: List[dict] = []
    for j_idx, j in enumerate(payload.jobs):
        if not _is_provided(j.name, min_len=2):
            raise HTTPException(status_code=400, detail=f"Job at index {j_idx} missing 'name'")
        if not _is_provided(j.description, min_len=30):
            raise HTTPException(status_code=400, detail=f"Job at index {j_idx} missing 'description'")
        # Bound task indices
        valid_idxs: List[int] = []
        for idx in (j.task_indices or []):
            if isinstance(idx, int) and 0 <= idx < len(tasks_clean):
                valid_idxs.append(idx)
        jobs_clean.append({"name": _normalize_ws(j.name), "description": _normalize_ws(j.description), "task_indices": sorted(set(valid_idxs))})

    # Generate functions + job-to-function mapping
    try:
        raw = generate_functions_for_jobs_batch(
            _normalize_ws(payload.process_name),
            _strip_headings(_normalize_ws(payload.process_description)),
            tasks_clean,
            jobs_clean,
            approx_words=350,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate functions: {e}")

    funcs = raw.get("functions", [])
    job_to_function = raw.get("job_to_function", [])

    # Post-process function descriptions to ~350 words and clean text
    TARGET = 350
    TOL = 20
    MIN_TARGET = TARGET - TOL
    MAX_TARGET = TARGET + TOL
    cleaned_funcs: List[FunctionItem] = []
    for f in funcs:
        fname = _sanitize_text(_normalize_ws(f.get("name", "") or "Operations"))
        fdesc = _sanitize_text(_strip_headings(_normalize_ws(f.get("description", "") or "")))
        fdesc = _dedupe_sentences(fdesc)
        wc = _word_count(fdesc)
        if wc < MIN_TARGET:
            fdesc += " The description outlines scope, interfaces, controls, and quality mechanisms to support consistent, auditable outcomes."
            fdesc = _sanitize_text(_strip_headings(_normalize_ws(fdesc)))
            fdesc = _dedupe_sentences(fdesc)
            wc = _word_count(fdesc)
        if wc > MAX_TARGET:
            fdesc = _trim_to_words(fdesc, TARGET)
        # Job indices: ensure valid
        idxs_in = f.get("job_indices", []) or []
        idxs: List[int] = []
        for idx in idxs_in:
            if isinstance(idx, int) and 0 <= idx < len(jobs_clean):
                idxs.append(idx)
        cleaned_funcs.append(FunctionItem(name=fname, description=fdesc, job_indices=sorted(set(idxs))))

    # Ensure job_to_function covers all jobs
    if len(job_to_function) != len(jobs_clean):
        mapping = [None] * len(jobs_clean)
        for f in cleaned_funcs:
            for idx in f.job_indices:
                if 0 <= idx < len(mapping):
                    mapping[idx] = f.name
        for i in range(len(mapping)):
            if not mapping[i]:
                # create a function per orphaned job using a functionalized name from the job
                orphan_fn = re.sub(r"\b(Specialist|Analyst|Manager|Coordinator|Officer|Technician|Consultant|Assistant|Associate|Lead|Director)\b", "", jobs_clean[i]["name"]).strip()
                orphan_fn = orphan_fn if orphan_fn else "Operations"
                cleaned_funcs.append(FunctionItem(name=orphan_fn, description="", job_indices=[i]))
                mapping[i] = orphan_fn
        job_to_function = mapping  # type: ignore

    return {"functions": cleaned_funcs}

# ---------------------- Process Graph (DOT and Image) ----------------------
class GraphTask(BaseModel):
    name: str


class GraphJob(BaseModel):
    name: str
    task_indices: List[int] = []


class GraphFunction(BaseModel):
    name: str
    job_indices: List[int] = []


class ProcessGraphRequest(BaseModel):
    process_name: str
    tasks: List[GraphTask] = []
    jobs: List[GraphJob] = []
    functions: List[GraphFunction] = []


def _escape_label(s: str) -> str:
    s = s or ""
    s = s.replace("\\", "\\\\").replace("\"", "\\\"")
    return s


def _build_process_graph_dot(payload: ProcessGraphRequest) -> str:
    # Normalize
    pname = re.sub(r"\s+", " ", payload.process_name or "").strip() or "Process"
    tasks = payload.tasks or []
    jobs = payload.jobs or []
    functions = payload.functions or []

    # Node IDs
    pid = "P"
    tid = [f"T{i}" for i in range(len(tasks))]
    jid = [f"J{i}" for i in range(len(jobs))]
    fid = [f"F{i}" for i in range(len(functions))]

    # Build DOT
    lines: List[str] = []
    lines.append("digraph G {")
    lines.append("  graph [rankdir=TB, fontsize=10, fontname=Helvetica, newrank=true, ranksep=1.0, nodesep=0.5];")
    lines.append("  node  [shape=rectangle, style=filled, fillcolor=white, color=gray40, fontname=Helvetica, margin=0.15, width=0, height=0];")
    lines.append("  edge  [color=gray40, fontname=Helvetica, arrowsize=0.7];")

    # Add system boundaries as subgraphs with strict hierarchy
    lines.append("  // System boundaries with strict hierarchy")

    # Process (top level)
    lines.append("  subgraph cluster_process { \
    label = \"Process\"; \
    style=filled; \
    color=lightgray; \
    fillcolor=white; \
    penwidth=2;")
    lines.append(f"    {pid} [label=\"{_escape_label(pname)}\", fillcolor=lightgoldenrod1];")
    lines.append("  }")

    # Tasks (second level)
    if tasks:
        lines.append("  subgraph cluster_tasks { \
    label = \"Tasks\"; \
    style=filled; \
    color=lightgray; \
    fillcolor=white; \
    penwidth=2;")
        for i, t in enumerate(tasks):
            tname = re.sub(r"\s+", " ", (t.name or "")).strip() or f"Task {i+1}"
            lines.append(f"    {tid[i]} [label=\"{_escape_label(tname)}\", fillcolor=lightcyan];")
        lines.append("  }")

    # Jobs (third level)
    if jobs:
        lines.append("  subgraph cluster_jobs { \
    label = \"Jobs\"; \
    style=filled; \
    color=lightgray; \
    fillcolor=white; \
    penwidth=2;")
        for j, job in enumerate(jobs):
            jname = re.sub(r"\s+", " ", (job.name or "")).strip() or f"Job {j+1}"
            lines.append(f"    {jid[j]} [label=\"{_escape_label(jname)}\", fillcolor=lavender];")
        lines.append("  }")

    # Functions (bottom level)
    if functions:
        lines.append("  subgraph cluster_functions { \
    label = \"Functions\"; \
    style=filled; \
    color=lightgray; \
    fillcolor=white; \
    penwidth=2;")
        for f, func in enumerate(functions):
            fname = re.sub(r"\s+", " ", (func.name or "")).strip() or f"Function {f+1}"
            lines.append(f"    {fid[f]} [label=\"{_escape_label(fname)}\", fillcolor=mistyrose];")
        lines.append("  }")

    # Define the strict hierarchy
    lines.append("")
    lines.append("  // Strict hierarchy constraints")

    # Process to Tasks edges (visible with label)
    for i in range(len(tid)):
        lines.append(f"  {pid} -> {tid[i]} [label=\"includes\", color=gray60];")

    # Tasks to Jobs edges (visible with label)
    for j, job in enumerate(jobs):
        for ti in (job.task_indices or []):
            if isinstance(ti, int) and 0 <= ti < len(tid):
                lines.append(f"  {tid[ti]} -> {jid[j]} [label=\"performed by\", color=gray60];")

    # Jobs to Functions edges (visible hierarchy)
    for f, func in enumerate(functions):
        for ji in (func.job_indices or []):
            if isinstance(ji, int) and 0 <= ji < len(jid):
                lines.append(f"  {jid[ji]} -> {fid[f]} [style=solid, color=gray80, dir=back, arrowtail=dot, arrowhead=none];")

    # Actual relationships (visible edges)
    lines.append("")
    lines.append("  // Actual relationships")

    # Job -> Task relationships (removing duplicate relationships)
    # Handled in the Tasks to Jobs section above

    # Function -> Job relationships (using dir=back for better layout)
    # Handled in the Jobs to Functions section above

    # Force the hierarchy levels
    lines.append("")
    lines.append("  // Force hierarchy levels")
    lines.append("  { rank=min; " + pid + " }")
    if tid:
        lines.append("  { rank=same; " + " ".join(tid) + " }")
    if jid:
        lines.append("  { rank=same; " + " ".join(jid) + " }")
    if fid:
        lines.append("  { rank=max; " + " ".join(fid) + " }")

    lines.append("}")
    return "\n".join(lines)



@app.post("/processgraph/image")
def create_process_graph_image(
    payload: ProcessGraphRequest,
    format: str = "png",
) -> Any:
    if not payload.process_name:
        raise HTTPException(status_code=400, detail="'process_name' is required")
    fmt = (format or "png").lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in {"png", "jpeg", "svg"}:
        raise HTTPException(status_code=400, detail="format must be one of: png, jpeg, svg")
    dot = _build_process_graph_dot(payload)
    try:
        import graphviz  # type: ignore
    except Exception:
        raise HTTPException(status_code=501, detail="Graph rendering requires 'graphviz' Python package and Graphviz system binaries (dot). Install and retry.")
    try:
        data = graphviz.Source(dot).pipe(format=fmt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render graph: {e}")

    # Prepare filename
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", payload.process_name or "process").strip("_") or "process"
    filename = f"{base}.{ 'png' if fmt=='png' else ('jpg' if fmt=='jpeg' else 'svg') }"
    media = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "svg": "image/svg+xml",
    }[fmt]
    from fastapi.responses import Response
    headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
    return Response(content=data, media_type=media, headers=headers)

# For local running: `uvicorn api_main:app --reload`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#python -m uvicorn api_main:app --reload