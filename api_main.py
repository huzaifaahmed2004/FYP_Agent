from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any

# Import the retrieval function and ensure heavy resources are available once per process
from query_process import retrieve_processes  # loads models/redis at import

app = FastAPI(title="Process Identification API", version="1.0.0")


class IdentifyRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class Candidate(BaseModel):
    id: Optional[int]
    name: Optional[str]
    description: Optional[str]
    sim: float


class IdentifyResponse(BaseModel):
    best: Optional[Candidate]
    candidates: List[Candidate]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/identify", response_model=IdentifyResponse)
def identify_process(payload: IdentifyRequest) -> Any:
    q = (payload.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    try:
        top_k = payload.top_k if payload.top_k and payload.top_k > 0 else 3
        cands = retrieve_processes(q, top_k=top_k)
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
        return {"best": best, "candidates": serializable}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identification failed: {e}")


# For local running: `uvicorn api_main:app --reload`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
