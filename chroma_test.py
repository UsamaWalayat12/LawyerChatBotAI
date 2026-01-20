# rag_query.py
"""
RAG pipeline (retrieval + generation) for Pakistani legal documents (ChromaDB).
- Robust chroma query handling
- Phrase + keyword re-ranking
- Prompt builder that enforces: "Use ONLY the evidence below"
- Hides filenames in output, extracts judge/court/date/clause metadata heuristically
- Calls Gemini (google-genai) using GEMINI_API_KEY from env
- Produces a structured, clean output with headings and numeric inline citations [1], [2]
"""

import os
import sys
import json
import re
from typing import List, Dict, Any

# sentence_transformers imported conditionally below (only if not using Chroma Cloud)
import chromadb

# Gemini (google-genai) client
try:
    from google import genai
except Exception:
    genai = None

# ---------------------------
# Configuration - adjust paths/model names as needed
# ---------------------------
# ---------------------------
# Configuration - adjust paths/model names as needed
# ---------------------------
# Use environment variable for DB path, default to relative "ChromaDB"
DB_DIR = os.environ.get("CHROMA_DB_DIR", "ChromaDB")
COLLECTION = "pakistan_law"
MODEL_NAME = "all-MiniLM-L6-v2"     # MUST match the model used in Chroma Cloud
TOP_K = 20
RETURN_TOP = 5
DEBUG = True  # Enable debug mode to see what's happening

KEYWORD_BOOST = ["notice", "demand", "breach", "contract", "claim", "payment", "overdue"]
PRIORITY_PHRASES = [
    "notice of demand", "legal notice", "demand notice", "final notice",
    "specimen notice", "format of notice", "draft notice", "specimen",
    "template", "form of notice", "notice to pay", "notice for payment"
]

EXCERPT_CHAR_LIMIT = 1200

# LLM model name for Gemini: set via env GEMINI_MODEL or default
GEMINI_MODEL = os.environ.get("GEMINI_MODEL") or "models/gemini-2.5-flash"

# ---------------------------
# Initialize embedding model (only if not using Chroma Cloud)
# ---------------------------
# Always load embedding model (needed for both local and cloud)
print("Loading embedding model:", MODEL_NAME)
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print(f"✓ Loaded embedding model: {MODEL_NAME}")
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer: {e}")
    model = None

print("=" * 60)
print("CHROMA CONNECTION DIAGNOSTIC")
print("=" * 60)
print("Connecting to ChromaDB...")

# Check environment variables
CHROMA_API_KEY = os.environ.get("CHROMA_API_KEY")
CHROMA_TENANT = os.environ.get("CHROMA_TENANT")
CHROMA_DATABASE = os.environ.get("CHROMA_DATABASE")

print(f"CHROMA_API_KEY present: {bool(CHROMA_API_KEY)}")
print(f"CHROMA_TENANT present: {bool(CHROMA_TENANT)}")
print(f"CHROMA_DATABASE present: {bool(CHROMA_DATABASE)}")

try:
    if CHROMA_API_KEY and CHROMA_TENANT and CHROMA_DATABASE:
        print(f"✓ All Chroma Cloud env vars found!")
        print(f"  Tenant: {CHROMA_TENANT}")
        print(f"  Database: {CHROMA_DATABASE}")
        print(f"  Attempting cloud connection...")
        
        try:
            from chromadb import CloudClient
            import os

            # Make sure your environment variables are set:
            # CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE

            client_chroma = CloudClient(
                tenant=os.environ.get("CHROMA_TENANT"),
                database=os.environ.get("CHROMA_DATABASE"),
                api_key=os.environ.get("CHROMA_API_KEY")
            )

            print(f"  ✓ CloudClient created for Chroma Cloud (v2 API)")
            
            col = client_chroma.get_or_create_collection(
                name=COLLECTION
            )

            print(f"  ✓ Collection '{COLLECTION}' connected (cloud)")
            
            # Test count
            try:
                count = col.count()
                print(f"  ✓ Chroma Cloud connected! Documents: {count}")
            except Exception as e:
                print(f"  ✓ Cloud connection OK (count not available): {e}")
        except Exception as cloud_error:
            print(f"  ✗ Chroma Cloud connection failed: {cloud_error}")
            raise cloud_error
            
    # Fallback to local ChromaDB (for development only, NOT when cloud vars are set)
    elif os.path.exists(DB_DIR):
        print(f"Using local ChromaDB at: {DB_DIR}")
        try:
            client_chroma = chromadb.PersistentClient(path=DB_DIR)
            col = client_chroma.get_or_create_collection(COLLECTION)
            print("✓ Local ChromaDB connected successfully")
        except Exception as local_error:
            print(f"  ✗ Local ChromaDB connection failed: {local_error}")
            raise local_error
    else:
        print(f"WARNING: No ChromaDB connection available!")
        print(f"  - No Chroma Cloud env vars")
        print(f"  - No local DB at: {DB_DIR}")
        client_chroma = None
        col = None
except Exception as e:
    print(f"ERROR connecting to ChromaDB: {e}")
    import traceback
    traceback.print_exc()
    client_chroma = None
    col = None

# ---------------------------
# Configure Gemini from environment and instantiate client
# ---------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not set in environment. Set it and re-run.")
    sys.exit(1)

if genai is None:
    print("ERROR: google-genai SDK not installed. Run: pip install google-genai")
    sys.exit(1)

try:
    client = genai.Client(api_key=GEMINI_API_KEY)  # Explicitly pass API key
except Exception as e:
    print("Failed to instantiate google-genai client:", e)
    sys.exit(1)

# ---------------------------
# Regexes for metadata extraction
# ---------------------------
CASE_FIELDS_REGEX = {
    "pld": re.compile(r'\bPLD\s+\d{4}\s+\w+\s+\d+', re.IGNORECASE),
    "scmr": re.compile(r'\bSCMR\s*\d{1,6}', re.IGNORECASE),
    "judgment_date_iso": re.compile(r'\b(?:Dated|Date of judgment|Judgment Date)[:\s\-]*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})', re.IGNORECASE),
    "judgment_date_text": re.compile(r'\bJudgment\s+dated[:\s]*([A-Za-z]+\s+\d{1,2},\s*\d{4})', re.IGNORECASE),
}

# ---------------------------
# Helpers: scoring + safe extraction
# ---------------------------
def phrase_score(text: str, phrases: List[str] = PRIORITY_PHRASES) -> int:
    t = (text or "").lower()
    return sum(1 for p in phrases if p.lower() in t)

def safe_text_len(text: str) -> int:
    return len(text or "")

# ---------------------------
# Retrieval & filtering
# ---------------------------
def retrieve_and_filter(query: str,
                        top_k: int = TOP_K,
                        keyword_boost: List[str] = KEYWORD_BOOST,
                        return_top: int = RETURN_TOP) -> List[Dict[str, Any]]:
    if col is None:
        print("Error: Collection not initialized.")
        return []

    include = ['documents', 'metadatas', 'distances', 'data']
    
    # Check if using Chroma Cloud (no local model needed)
    CHROMA_API_KEY = os.environ.get("CHROMA_API_KEY")
    using_cloud = CHROMA_API_KEY is not None
    
    try:
        if using_cloud:
            # Chroma Cloud: Still need to generate embeddings locally
            if model is None:
                print("Error: Model not initialized for Chroma Cloud.")
                return []
            print(f"[DEBUG] Using Chroma Cloud with embeddings: {query[:50]}...")
            q_emb = model.encode(query, convert_to_numpy=True).tolist()
            res = col.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
        else:
            # Local ChromaDB: Use embeddings
            if model is None:
                print("Error: Model not initialized for local ChromaDB.")
                return []
            print(f"[DEBUG] Using local ChromaDB with embeddings: {query[:50]}...")
            q_emb = model.encode(query, convert_to_numpy=True).tolist()
            res = col.query(query_embeddings=[q_emb], n_results=top_k, include=include)
    except Exception as e:
        if DEBUG:
            print("Chroma query failed:", e)
        # Fallback for local
        if not using_cloud and model is not None:
            try:
                q_emb = model.encode(query, convert_to_numpy=True).tolist()
                res = col.query(query_embeddings=[q_emb], n_results=top_k, include=['documents', 'metadatas', 'distances'])
            except Exception as e2:
                print(f"Fallback query also failed: {e2}")
                return []
        else:
            return []

    if DEBUG:
        print("chroma keys:", list(res.keys()))

    def safe_get_top_list(key: str, default=None):
        if key in res:
            val = res[key]
            if isinstance(val, list) and len(val) > 0:
                return val[0]
            return val
        return default if default is not None else []

    docs = safe_get_top_list('documents', [])
    metas = safe_get_top_list('metadatas', [])
    dists = safe_get_top_list('distances', [])

    ids = []
    if 'ids' in res:
        ids = safe_get_top_list('ids', [])
    else:
        data_block = safe_get_top_list('data', [])
        if isinstance(data_block, list) and all(isinstance(x, str) for x in data_block):
            ids = data_block
        elif isinstance(data_block, dict):
            ids = data_block.get('ids') or data_block.get('id') or []
        elif isinstance(data_block, list) and len(data_block) > 0 and isinstance(data_block[0], dict):
            first = data_block[0]
            ids = first.get('ids') or first.get('id') or []
        else:
            ids = []

    n = max(len(docs), len(metas), len(dists), len(ids))
    docs = list(docs) + [""] * (n - len(docs))
    metas = list(metas) + [{}] * (n - len(metas))
    dists = list(dists) + [float('inf')] * (n - len(dists))
    ids = list(ids) + [""] * (n - len(ids))

    candidates = []
    for doc, meta, _id, dist in zip(docs, metas, ids, dists):
        text = doc or ""
        text_lower = text.lower()
        kw_count = sum(text_lower.count(kw.lower()) for kw in keyword_boost)
        p_score = phrase_score(text)
        try:
            score = float(dist) if dist is not None else float('inf')
        except Exception:
            score = float('inf')

        candidates.append({
            "id": _id or meta.get('id') or meta.get('doc_id') or "",
            "text": text,
            "meta": meta or {},
            "dist": score,
            "kw": int(kw_count),
            "p_score": int(p_score),
            "len": safe_text_len(text)
        })

    # length penalty
    LENGTH_PENALTY_THRESHOLD = 8000
    for c in candidates:
        if c['len'] > LENGTH_PENALTY_THRESHOLD and c['p_score'] == 0:
            c['dist'] = c['dist'] + 1.0 + (c['len'] - LENGTH_PENALTY_THRESHOLD) / 20000.0

    candidates = sorted(candidates, key=lambda x: (-x['p_score'], -x['kw'], x['dist']))

    # Apply filters with more lenient criteria
    filtered = []
    MIN_LENGTH = 50  # Reduced from 200 to be more lenient
    
    for c in candidates:
        # Filter out very short documents (< 50 chars)
        if c['len'] < MIN_LENGTH:
            if DEBUG:
                print(f"[FILTER] Skipped doc (too short): {c['len']} chars")
            continue
        
        # Allow English, unknown, or missing language metadata
        lang = str(c['meta'].get('lang', 'en')).lower()
        if lang and not lang.startswith('en') and lang != 'unknown' and lang != '':
            if DEBUG:
                print(f"[FILTER] Skipped doc (non-English): lang={lang}")
            continue
        
        filtered.append(c)
    
    # Fallback: if all documents were filtered out, return top unfiltered candidates
    if len(filtered) == 0 and len(candidates) > 0:
        print(f"WARNING: All {len(candidates)} documents were filtered out. Returning top {return_top} unfiltered results.")
        # Return candidates that are at least somewhat relevant (not empty)
        fallback = [c for c in candidates if c['len'] > 0][:return_top]
        if DEBUG:
            for c in fallback:
                print(f"[FALLBACK] Returning: len={c['len']}, lang={c['meta'].get('lang', 'unknown')}, dist={c['dist']:.4f}")
        return fallback

    if DEBUG and len(filtered) > 0:
        print(f"[FILTER] Kept {len(filtered)}/{len(candidates)} documents after filtering")

    return filtered[:return_top]

# ---------------------------
# Prompt assembly functions
# ---------------------------
def build_evidence_enumeration(evidences: List[Dict[str, Any]]) -> str:
    """
    Build a compact enumeration of evidences with numeric citations only.
    Filenames are intentionally NOT shown; citations use numeric indexes only: [1], [2], ...
    """
    parts = []
    for i, e in enumerate(evidences, start=1):
        md = e.get('meta', {})
        doc_id = e.get('id') or f"local-{i}"
        excerpt = (e.get('text') or "")[:EXCERPT_CHAR_LIMIT].strip().replace("\n", " ")
        parts.append(f"[{i}] doc_id={doc_id} source=[REDACTED]\n{excerpt}\n")
    return "\n\n".join(parts)

def build_strict_rag_prompt(query: str, evidences: List[Dict[str, Any]]) -> str:
    """
    Strict instruction for the generator: produce a clean, structured answer with headings.
    The generator must use ONLY the evidence below and cite each factual/interpretive line
    with [doc_index]. If evidence is insufficient for an item, say "insufficient evidence".
    """
    header = (
        "You are an expert Pakistani legal assistant. Use ONLY the evidence below to answer the QUESTION. "
        "Cite each factual/interpretive statement inline using the format [doc_index]. "
        "If the evidence does not support a requested item, say 'insufficient evidence' for that item.\n\n"
    )
    qline = f"QUESTION: {query}\n\n"
    evid = build_evidence_enumeration(evidences)
    instructions = (
        "Produce the final answer with the following exact sections and headings:\n\n"
        "1) SHORT ANSWER (one paragraph): concise direct answer to the question.\n\n"
        "2) OVERVIEW (one paragraph): what Section/issue covers and immediate legal context.\n\n"
        "3) GROUNDS / RULES (numbered bullets): list the legal grounds, short explanation; cite evidence for each ground.\n\n"
        "4) EVIDENCE (numbered): for each evidence item provided, give a one-line descriptor and the excerpted fact used, with citation [doc_index].\n\n"
        "5) JUDGMENT SUMMARIES (for any judgment evidence): for each judgment evidence, produce: Case details (parties, court, citation if found), 1-paragraph factual summary, Holding/Result (1 line), and Reasoning (2 bullets). Cite the document for each sentence.\n\n"
        "6) STATUTORY TEXT (if included in evidence): quote short relevant statutory phrases (<= 25 words) and cite. If not in evidence, state 'statute not found in supplied evidence'.\n\n"
        "7) PRACTICAL TEMPLATE or DRAFT (if question asks to draft): Provide a ready-to-use template/format with placeholders.\n\n"
        # Prompt tweak requesting metadata explicitly
        "For any evidence that is a judgment, explicitly list: Judge(s), Court, Date of judgment, and citation (if present) using only the supplied evidence. "
        "If any of these are not present in the evidence, state 'insufficient evidence' for that field.\n\n"
        "Make the language clear, professional and do not add extraneous commentary. Use only evidence provided; do not invent case names, citations, or facts.\n\n"
    )
    return header + qline + evid + "\n\n" + instructions

# ---------------------------
# Metadata extraction
# ---------------------------
def extract_case_metadata(text: str) -> Dict[str, str]:
    """
    Try to extract case metadata: parties, PLD/SCMR citations, judgment date,
    judge name, court name, statute/clause references.
    This is heuristic — it returns what it finds, or empty dict keys if none.
    """
    out = {}
    snippet = (text or "")[:35000]  # large snippet to search

    # PLD / SCMR / date heuristics
    for k, rx in CASE_FIELDS_REGEX.items():
        m = rx.search(snippet)
        if m:
            out[k] = m.group(0).strip()

    # Parties (try to find 'X v. Y' near top)
    header_lines = snippet.splitlines()[:80]
    header_text = " ".join(header_lines)
    m = re.search(r'([A-Z][\w\.\-\,\s]{1,120}?)\s+(?:v\.|vs\.|versus|v)\s+([A-Z][\w\.\-\,\s]{1,120}?)', header_text)
    if m:
        out['parties'] = m.group(0).strip()

    # Judgment date (text or numeric)
    m = re.search(r'\bJudgment\s+dated[:\s]*([A-Za-z]+\s+\d{1,2},\s*\d{4})', snippet, re.IGNORECASE)
    if m:
        out['judgment_date'] = m.group(1).strip()
    else:
        m2 = re.search(r'\b(?:Dated|Date of judgment|Judgment Date)[:\s]*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})', snippet, re.IGNORECASE)
        if m2:
            out['judgment_date'] = m2.group(1).strip()

    # Judge detection: common patterns
    judge_patterns = [
        r'Hon(?:\'ble)?\s+Mr\.?\s+Justice\s+([A-Z][A-Za-z\-\s\.]+)',
        r'Hon(?:\'ble)?\s+Mr?\.?\s+Justice\s+([A-Z][A-Za-z\-\s\.]+)',
        r'J(?:ustice)?\.\s+([A-Z][A-Za-z\-\s\.]+)',
        r'Before\s+Mr\.?\s+Justice\s+([A-Z][A-Za-z\-\s\.]+)',
        r'Coram[:\s]+([A-Z][A-Za-z\-\s\,\.\s]+)'
    ]
    for p in judge_patterns:
        m = re.search(p, header_text, re.IGNORECASE)
        if m:
            out['judge'] = m.group(1).strip()
            break

    # Court detection
    court_patterns = [
        r'(?:(?:Lahore High Court)|(?:Sindh High Court)|(?:Supreme Court of Pakistan)|(?:High Court of [A-Za-z]+))',
        r'\bHigh Court\b', r'\bDistrict Court\b', r'\bCivil Court\b'
    ]
    for p in court_patterns:
        m = re.search(p, snippet, re.IGNORECASE)
        if m:
            out['court'] = m.group(0).strip()
            break

    # Statute/clause references
    clause_patterns = [
        r'\bSection\s+\d+[A-Za-z0-9\-]*', r'\bSec\.\s*\d+', r'\bClause\s+\d+',
        r'\bOrder\s+XXI\s+Rule\s+\d+', r'\bOrder\s+[IVXLCDM]+\b', r'\bArticle\s+\d+'
    ]
    clauses_found = set()
    for p in clause_patterns:
        for m in re.finditer(p, snippet, re.IGNORECASE):
            clauses_found.add(m.group(0).strip())
    if clauses_found:
        out['clauses'] = "; ".join(sorted(clauses_found))

    return out

# ---------------------------
# Gemini caller (robust multi-shape handling)
# ---------------------------
def call_gemini_chat(prompt_text: str, model: str = GEMINI_MODEL, temperature: float = 0.0, max_tokens: int = 1600) -> str:
    """
    Robust Gemini caller:
    - tries to call with a GenerationConfig if available,
    - falls back to a simple call without config,
    - extracts text from several response shapes.
    """
    try:
        if DEBUG:
            print("\n[DEBUG] Calling Gemini model:", model)
            print("[DEBUG] prompt (first 400 chars):", prompt_text[:400].replace("\n", " "))

        response = None
        last_err = None

        # 1) Try with GenerationConfig if genai.types exists
        try:
            if hasattr(genai, "types") and hasattr(genai.types, "GenerationConfig"):
                cfg = genai.types.GenerationConfig(
                    temperature=float(temperature),
                    max_output_tokens=int(max_tokens),
                )
                response = client.models.generate_content(model=model, contents=[prompt_text], config=cfg)
        except Exception as e:
            last_err = e
            if DEBUG:
                print("[DEBUG] config-based call failed:", repr(e))
            response = None

        # 2) Fallback: simple call without config
        if response is None:
            try:
                response = client.models.generate_content(model=model, contents=[prompt_text])
            except Exception as e:
                last_err = e
                if DEBUG:
                    print("[DEBUG] simple generate_content() call failed:", repr(e))
                # try older/other wrappers if present
                try:
                    if hasattr(genai, "generate"):
                        response = genai.generate(model=model, input=prompt_text)
                    elif hasattr(genai, "generate_text"):
                        response = genai.generate_text(model=model, prompt=prompt_text)
                except Exception as e2:
                    last_err = e2
                    if DEBUG:
                        print("[DEBUG] other fallbacks failed:", repr(e2))
                    response = None

        if response is None:
            raise RuntimeError(f"No response from Gemini client (last error: {last_err})")

        # 3) Extract text from known shapes
        text_output = None

        # direct .text
        if hasattr(response, "text") and response.text:
            text_output = response.text

        # typed .response
        if not text_output and hasattr(response, "response") and response.response:
            text_output = getattr(response, "response")

        # typed candidates
        if not text_output and hasattr(response, "candidates"):
            try:
                cands = getattr(response, "candidates")
                if isinstance(cands, (list, tuple)) and len(cands) > 0:
                    first = cands[0]
                    if hasattr(first, "content") and hasattr(first.content, "parts"):
                        parts = getattr(first.content, "parts")
                        collected = []
                        for p in parts:
                            if hasattr(p, "text"):
                                collected.append(getattr(p, "text"))
                            else:
                                collected.append(str(p))
                        text_output = "".join(collected).strip()
                    elif isinstance(first, dict):
                        text_output = first.get("content") or first.get("text") or first.get("output")
                    else:
                        text_output = str(first)
            except Exception:
                if DEBUG:
                    print("[DEBUG] extracting from response.candidates failed")

        # outputs / output
        if not text_output and (hasattr(response, "outputs") or hasattr(response, "output")):
            out = getattr(response, "outputs", None) or getattr(response, "output", None)
            try:
                if isinstance(out, (list, tuple)):
                    parts = []
                    for b in out:
                        if isinstance(b, dict):
                            parts.append(b.get("text") or b.get("content") or "")
                        elif hasattr(b, "text"):
                            parts.append(getattr(b, "text"))
                        else:
                            parts.append(str(b))
                    text_output = "\n".join(p for p in parts if p).strip()
                elif isinstance(out, str):
                    text_output = out
            except Exception:
                if DEBUG:
                    print("[DEBUG] extracting from outputs failed")

        # dict-like top-level
        if not text_output and isinstance(response, dict):
            if "text" in response:
                text_output = response.get("text")
            elif "candidates" in response and response["candidates"]:
                first = response["candidates"][0]
                if isinstance(first, dict):
                    text_output = first.get("content") or first.get("text") or first.get("output")
            elif "output" in response:
                out = response["output"]
                if isinstance(out, str):
                    text_output = out
                elif isinstance(out, list):
                    parts = []
                    for b in out:
                        if isinstance(b, dict):
                            parts.append(b.get("text") or b.get("content") or "")
                        else:
                            parts.append(str(b))
                    text_output = "\n".join(p for p in parts if p).strip()

        # final fallback
        if not text_output:
            try:
                text_output = str(response)
            except Exception:
                text_output = None

        if not text_output:
            raise RuntimeError("No usable text returned from Gemini client. Enable DEBUG for raw response details.")

        return str(text_output)

    except Exception as exc:
        print("Gemini API call failed:", exc)
        raise

# ---------------------------
# Chat History Management (Supabase)
# ---------------------------
try:
    from supabase import create_client, Client
    SUPABASE_URL = "https://qyzpsokrfvxwwbgkzteq.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF5enBzb2tyZnZ4d3diZ2t6dGVxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI3ODA1MzIsImV4cCI6MjA3ODM1NjUzMn0.DvUq7JpDYOHWRgGBhemTpFN-veli0ebVp3k4btzC7Uo"
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✓ Supabase client initialized for chat history")
    SUPABASE_AVAILABLE = True
except Exception as e:
    print(f"Warning: Supabase not available: {e}")
    SUPABASE_AVAILABLE = False
    supabase = None

def load_history(user_id: str = "default", conversation_id: str = None) -> List[Dict[str, str]]:
    """Load chat history from Supabase."""
    if not SUPABASE_AVAILABLE:
        return []
    
    try:
        query = supabase.table('chat_messages').select('*').eq('user_id', user_id)
        
        # Filter by conversation_id if provided
        if conversation_id:
            query = query.eq('conversation_id', conversation_id)
        
        response = query.order('created_at').execute()
        if response.data:
            return [{"role": msg["role"], "content": msg["content"]} for msg in response.data]
        return []
    except Exception as e:
        print(f"Error loading history from Supabase: {e}")
        return []

def save_history(history: List[Dict[str, str]], user_id: str = "default"):
    """Save chat history to Supabase (deprecated - use add_to_history instead)."""
    # This function is kept for backwards compatibility but does nothing
    # Individual messages are now saved via add_to_history
    pass

def add_to_history(history: List[Dict[str, str]], role: str, content: str, user_id: str = "default", conversation_id: str = "default"):
    """Add a message to history in Supabase."""
    if not SUPABASE_AVAILABLE:
        # Fallback: just append to in-memory history
        history.append({"role": role, "content": content})
        return
    
    try:
        from datetime import datetime
        supabase.table('chat_messages').insert({
            "user_id": user_id,
            "role": role,
            "content": content,
            "conversation_id": conversation_id,
            "created_at": datetime.now().isoformat()
        }).execute()
        # Also add to in-memory history for current session
        history.append({"role": role, "content": content})
    except Exception as e:
        print(f"Error adding to Supabase history: {e}")
        # Fallback: just append to in-memory history
        history.append({"role": role, "content": content})

def format_history_for_prompt(history: List[Dict[str, str]], max_history: int = 5) -> str:
    """Format recent chat history for inclusion in prompt."""
    if not history:
        return ""
    
    recent = history[-(max_history * 2):] if len(history) > max_history * 2 else history
    formatted = "\n\nCHAT HISTORY (for context):\n"
    for msg in recent:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")[:500]
        formatted += f"{role}: {content}\n\n"
    return formatted

def clear_history(user_id: str = "default", conversation_id: str = None):
    """Clear all chat history for a specific user, or a specific conversation."""
    if not SUPABASE_AVAILABLE:
        print("Supabase not available, cannot clear history.")
        return
    
    try:
        query = supabase.table('chat_messages').delete().eq('user_id', user_id)
        if conversation_id:
            query = query.eq('conversation_id', conversation_id)
        
        query.execute()
        
        if conversation_id:
            print(f"Chat history cleared for user: {user_id}, conversation: {conversation_id}")
        else:
            print(f"Chat history cleared for user: {user_id}")
    except Exception as e:
        print(f"Error clearing history: {e}")

# ---------------------------
# PDF Generation Module
# ---------------------------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

def extract_data_from_history(history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    AI-powered extraction of legal document data from chat history.
    Uses Gemini to intelligently understand and extract information.
    """
    # Combine all conversation text
    conversation_text = ""
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        conversation_text += f"{role.upper()}: {content}\n\n"
    
    # Create extraction prompt for AI
    extraction_prompt = f"""
Extract legal document information from this conversation. Return ONLY valid JSON with no additional text.

CONVERSATION:
{conversation_text}

Return this exact JSON structure with extracted information:

{{
    "document_type": "breach_notice",
    "sender_name": "",
    "sender_address": "",
    "recipient_name": "",
    "recipient_address": "",
    "amount": "",
    "contract_date": "",
    "due_date": "",
    "breach_details": "",
    "contract_description": "",
    "notice_date": "",
    "payment_deadline": "",
    "additional_details": ""
}}

Fill in the values based on the conversation. Use empty string "" if information is not mentioned.
Return ONLY the JSON object, nothing else.
"""

    try:
        # Call AI to extract data
        extracted_json = call_gemini_chat(extraction_prompt, model=GEMINI_MODEL, temperature=0.0, max_tokens=1000)
        
        # Parse JSON response
        import json
        try:
            # Clean the response - remove any markdown formatting
            cleaned_json = extracted_json.strip()
            if cleaned_json.startswith('```json'):
                cleaned_json = cleaned_json.replace('```json', '').replace('```', '').strip()
            elif cleaned_json.startswith('```'):
                cleaned_json = cleaned_json.replace('```', '').strip()
            
            data = json.loads(cleaned_json)
            print("✓ AI extraction successful!")
        except Exception as e:
            # Fallback to regex if JSON parsing fails
            print(f"AI extraction failed ({e}), falling back to regex...")
            return extract_data_with_regex(history)
        
        # Set current date if not provided
        if not data.get("notice_date"):
            from datetime import datetime
            data["notice_date"] = datetime.now().strftime("%B %d, %Y")
        
        # Set default payment deadline
        if not data.get("payment_deadline"):
            data["payment_deadline"] = "30 days"
            
        return data
        
    except Exception as e:
        print(f"AI extraction error: {e}")
        print("Falling back to regex extraction...")
        return extract_data_with_regex(history)

def extract_data_with_regex(history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Fallback regex-based extraction (original method).
    """
    data = {
        "document_type": "breach_notice",
        "sender_name": "",
        "sender_address": "",
        "recipient_name": "",
        "recipient_address": "",
        "amount": "",
        "contract_date": "",
        "due_date": "",
        "breach_details": "",
        "contract_description": "",
        "notice_date": "",
        "payment_deadline": "30 days",
        "additional_details": ""
    }
    
    # Combine all user messages
    all_text = " ".join([msg.get("content", "") for msg in history if msg.get("role") == "user"])
    
    # Extract amounts (PKR, Rs, rupees)
    amount_patterns = [
        r'(?:PKR|Rs\.?|rupees)\s*([0-9,]+(?:\.[0-9]{2})?)',
        r'([0-9,]+(?:\.[0-9]{2})?)\s*(?:PKR|Rs\.?|rupees)',
        r'amount.*?([0-9,]+)',
    ]
    for pattern in amount_patterns:
        match = re.search(pattern, all_text, re.IGNORECASE)
        if match:
            data["amount"] = "PKR " + match.group(1)
            break
    
    # Extract dates
    date_patterns = [
        r'(?:dated|date|on)\s+([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})',
        r'([0-9]{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+[0-9]{4})',
    ]
    dates_found = []
    for pattern in date_patterns:
        for match in re.finditer(pattern, all_text, re.IGNORECASE):
            dates_found.append(match.group(1))
    
    if len(dates_found) >= 1:
        data["contract_date"] = dates_found[0]
    if len(dates_found) >= 2:
        data["due_date"] = dates_found[1]
    
    # Extract names (capitalized words, common patterns)
    name_patterns = [
        r'(?:between|from|to|party|plaintiff|defendant|client)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(?:vs|versus|v\.)',
    ]
    names_found = []
    for pattern in name_patterns:
        for match in re.finditer(pattern, all_text):
            name = match.group(1).strip()
            if len(name) > 3 and name not in names_found:
                names_found.append(name)
    
    if len(names_found) >= 1:
        data["sender_name"] = names_found[0]
    if len(names_found) >= 2:
        data["recipient_name"] = names_found[1]
    
    # Extract breach/contract details from context
    breach_keywords = ["breach", "failed", "overdue", "non-payment", "default"]
    for msg in history:
        if msg.get("role") == "user":
            content = msg.get("content", "").lower()
            if any(kw in content for kw in breach_keywords):
                data["breach_details"] = msg.get("content", "")[:300]
                break
    
    # Set current date as notice date
    from datetime import datetime
    data["notice_date"] = datetime.now().strftime("%B %d, %Y")
    
    return data

def detect_document_type(query: str, history: List[Dict[str, str]]) -> str:
    """
    AI-powered detection of document type from conversation context.
    """
    # Combine conversation for context
    conversation_text = ""
    for msg in history[-5:]:  # Last 5 messages for context
        role = msg.get("role", "")
        content = msg.get("content", "")
        conversation_text += f"{role.upper()}: {content}\n"
    
    detection_prompt = f"""
Analyze this legal conversation and determine what type of document the user wants to generate.

CONVERSATION:
{conversation_text}

CURRENT QUERY: {query}

Return ONLY one of these document types:
- breach_notice (for contract breaches, overdue payments, violations)
- legal_notice (for formal legal warnings, demands)
- agreement (for contracts, terms of service)
- petition (for court applications, requests)
- affidavit (for sworn statements)
- power_of_attorney (for authorization documents)
- lease_agreement (for rental/property agreements)
- employment_contract (for job agreements)
- other (if none of the above fit)

Return ONLY the document type, nothing else.
"""

    try:
        doc_type = call_gemini_chat(detection_prompt, model=GEMINI_MODEL, temperature=0.0, max_tokens=50)
        doc_type = doc_type.strip().lower()
        
        # Validate response
        valid_types = ["breach_notice", "legal_notice", "agreement", "petition", "affidavit", 
                      "power_of_attorney", "lease_agreement", "employment_contract", "other"]
        
        if doc_type in valid_types:
            return doc_type
        else:
            return "breach_notice"  # default fallback
            
    except Exception as e:
        print(f"Document type detection error: {e}")
        # Fallback to keyword matching
        text = query.lower() + " " + " ".join([msg.get("content", "").lower() for msg in history[-3:]])
        
        if any(kw in text for kw in ["notice", "demand", "breach", "overdue", "payment"]):
            return "breach_notice"
        elif any(kw in text for kw in ["agreement", "contract"]):
            return "agreement"
        elif any(kw in text for kw in ["legal notice", "formal notice"]):
            return "legal_notice"
        elif any(kw in text for kw in ["petition", "application", "court"]):
            return "petition"
        elif any(kw in text for kw in ["affidavit", "sworn", "statement"]):
            return "affidavit"
        else:
            return "breach_notice"  # default

def generate_breach_notice_pdf(data: Dict[str, Any], output_file: str):
    """
    Generate a professional Breach of Contract Notice PDF.
    """
    if not REPORTLAB_AVAILABLE:
        print("ERROR: reportlab not installed. Run: pip install reportlab")
        return False
    
    doc = SimpleDocTemplate(output_file, pagesize=A4,
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=1*inch, bottomMargin=0.75*inch)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    # Header - Sender Info
    if data.get("sender_name"):
        story.append(Paragraph(data["sender_name"], styles['Normal']))
    if data.get("sender_address"):
        story.append(Paragraph(data["sender_address"], styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Date
    story.append(Paragraph(f"Date: {data.get('notice_date', '[DATE]')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Recipient Info
    story.append(Paragraph("<b>To:</b>", styles['Normal']))
    story.append(Paragraph(data.get("recipient_name", "[RECIPIENT NAME]"), styles['Normal']))
    if data.get("recipient_address"):
        story.append(Paragraph(data["recipient_address"], styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Title
    story.append(Paragraph("<b>NOTICE FOR BREACH OF CONTRACT – OVERDUE PAYMENT</b>", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Salutation
    story.append(Paragraph("Dear Sir/Madam,", body_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Body paragraphs
    para1 = f"""This letter serves as a formal notice regarding your breach of the contract dated 
    <b>{data.get('contract_date', '[CONTRACT DATE]')}</b> concerning 
    {data.get('contract_description', '[DESCRIPTION OF CONTRACT/AGREEMENT]')}."""
    story.append(Paragraph(para1, body_style))
    
    para2 = f"""Under the terms of the aforementioned contract, payment of 
    <b>{data.get('amount', '[AMOUNT]')}</b> was due on 
    <b>{data.get('due_date', '[DUE DATE]')}</b>."""
    story.append(Paragraph(para2, body_style))
    
    para3 = f"""As of the date of this letter, the principal sum of 
    <b>{data.get('amount', '[AMOUNT]')}</b> remains unpaid, constituting a clear breach of our agreement."""
    story.append(Paragraph(para3, body_style))
    
    para4 = """In accordance with the terms of our contract and applicable law, we hereby demand 
    immediate payment of the principal outstanding sum. Please note that you are also liable to pay 
    interest on this principal sum up to the day of payment."""
    story.append(Paragraph(para4, body_style))
    
    para5 = f"""We require you to remit the full outstanding amount, including applicable interest, 
    within <b>{data.get('payment_deadline', '30 days')}</b> from the date of service of this notice."""
    story.append(Paragraph(para5, body_style))
    
    para6 = """Please be advised that failure to make the full payment within the stipulated timeframe 
    will result in further action, including but not limited to, seeking compensation for the loss and 
    damage caused by your breach of contract, and we may not be obligated to perform any further 
    reciprocal promises under the contract."""
    story.append(Paragraph(para6, body_style))
    
    para7 = """We trust you will give this matter your immediate attention to resolve it amicably."""
    story.append(Paragraph(para7, body_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Closing
    story.append(Paragraph("Sincerely,", body_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Signature line
    story.append(Paragraph("_________________________", styles['Normal']))
    story.append(Paragraph(data.get("sender_name", "[YOUR NAME]"), styles['Normal']))
    story.append(Paragraph("[Your Title/Designation]", styles['Normal']))
    story.append(Paragraph("[Contact Information]", styles['Normal']))
    
    # Build PDF
    try:
        doc.build(story)
        return True
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return False

def generate_intelligent_pdf(history: List[Dict[str, str]], data: Dict[str, Any], doc_type: str, output_file: str) -> bool:
    """
    AI-powered PDF generation that creates appropriate legal documents based on conversation context.
    """
    if not REPORTLAB_AVAILABLE:
        print("ERROR: reportlab not installed. Run: pip install reportlab")
        return False
    
    # Combine conversation for context
    conversation_text = ""
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        conversation_text += f"{role.upper()}: {content}\n\n"
    
    # Create document generation prompt
    generation_prompt = f"""
You are a Pakistani legal document expert. Based on the conversation below, generate a professional legal document.

CONVERSATION CONTEXT:
{conversation_text}

EXTRACTED DATA:
{json.dumps(data, indent=2)}

DOCUMENT TYPE: {doc_type}

Generate a complete, professional legal document with the following structure:
1. Header with sender/recipient information
2. Date and reference
3. Subject line
4. Proper salutation
5. Body paragraphs with legal language
6. Demands/requests clearly stated
7. Legal consequences if applicable
8. Professional closing
9. Signature block

REQUIREMENTS:
- Use formal Pakistani legal language and terminology
- Include specific details from the conversation
- Make it legally sound and professional
- Use proper formatting with clear paragraphs
- Include all relevant dates, amounts, and names
- Add appropriate legal clauses and references
- Make it actionable and specific to the situation

Generate ONLY the document content, no explanations or additional text.
"""

    try:
        # Get AI-generated document content
        print("🤖 Generating document content with AI...")
        document_content = call_gemini_chat(generation_prompt, model=GEMINI_MODEL, temperature=0.1, max_tokens=3000)
        
        # Create PDF with the AI-generated content
        return create_formatted_pdf(document_content, data, output_file, doc_type)
        
    except Exception as e:
        print(f"AI document generation error: {e}")
        print("Falling back to template-based generation...")
        # Fallback to original template method
        if doc_type == "breach_notice":
            return generate_breach_notice_pdf(data, output_file)
        else:
            return generate_breach_notice_pdf(data, output_file)  # default template

def create_formatted_pdf(content: str, data: Dict[str, Any], output_file: str, doc_type: str) -> bool:
    """
    Create a well-formatted PDF from AI-generated content.
    """
    doc = SimpleDocTemplate(output_file, pagesize=A4,
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=1*inch, bottomMargin=0.75*inch)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    # Add document header
    from datetime import datetime
    story.append(Paragraph(f"Document Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
                          ParagraphStyle('Small', parent=styles['Normal'], fontSize=8, textColor=colors.grey)))
    story.append(Spacer(1, 0.2*inch))
    
    # Process and format the AI-generated content
    paragraphs = content.split('\n\n')
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # Check if it's a title/heading (all caps or starts with specific words)
        if (para.isupper() and len(para) < 100) or para.startswith(('NOTICE', 'AGREEMENT', 'PETITION', 'AFFIDAVIT')):
            story.append(Paragraph(para, title_style))
        elif para.startswith(('Subject:', 'Re:', 'Reference:')):
            story.append(Paragraph(f"<b>{para}</b>", heading_style))
        elif para.startswith(('Dear', 'To:', 'From:')):
            story.append(Paragraph(para, body_style))
        else:
            # Regular paragraph - make it look professional
            # Bold important legal terms
            legal_terms = ['breach of contract', 'legal notice', 'hereby demand', 'failure to comply', 
                          'legal action', 'without prejudice', 'in accordance with']
            
            formatted_para = para
            for term in legal_terms:
                formatted_para = formatted_para.replace(term, f"<b>{term}</b>")
            
            story.append(Paragraph(formatted_para, body_style))
        
        story.append(Spacer(1, 0.1*inch))
    
    # Add footer with document info
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("---", ParagraphStyle('Separator', parent=styles['Normal'], alignment=TA_CENTER)))
    story.append(Paragraph(f"Document Type: {doc_type.replace('_', ' ').title()}", 
                          ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey)))
    story.append(Paragraph("Generated by Pakistani Legal RAG Assistant", 
                          ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey)))
    
    # Build PDF
    try:
        doc.build(story)
        return True
    except Exception as e:
        print(f"Error creating formatted PDF: {e}")
        return False

def generate_full_content_pdf(history: List[Dict[str, str]], output_file: str) -> bool:
    """
    Generate PDF with full AI-generated content from chat history.
    This includes the complete answer with all evidence, citations, and reasoning.
    """
    if not REPORTLAB_AVAILABLE:
        print("ERROR: reportlab not installed. Run: pip install reportlab")
        return False
    
    # Get the last assistant response (the full AI answer)
    last_answer = None
    last_question = None
    
    for msg in reversed(history):
        if msg.get("role") == "assistant" and not last_answer:
            last_answer = msg.get("content", "")
        elif msg.get("role") == "user" and not last_question:
            last_question = msg.get("content", "")
        if last_answer and last_question:
            break
    
    if not last_answer:
        print("No AI-generated content found in history.")
        return False
    
    doc = SimpleDocTemplate(output_file, pagesize=A4,
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=1*inch, bottomMargin=0.75*inch)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=10
    )
    
    # Header
    from datetime import datetime
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Title
    story.append(Paragraph("<b>PAKISTANI LEGAL ANALYSIS</b>", title_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Question
    story.append(Paragraph("<b>QUERY:</b>", heading_style))
    story.append(Paragraph(last_question, body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Full AI Answer
    story.append(Paragraph("<b>LEGAL ANALYSIS & ANSWER:</b>", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Process the answer - split by sections and format
    answer_lines = last_answer.split('\n')
    for line in answer_lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.1*inch))
            continue
        
        # Detect section headings (numbered or all caps)
        if re.match(r'^\d+\)', line) or (line.isupper() and len(line) < 100):
            story.append(Paragraph(f"<b>{line}</b>", heading_style))
        else:
            # Escape special characters for reportlab
            line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(line, body_style))
    
    # Footer
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<i>This document was generated using AI-powered legal research and analysis.</i>", 
                          styles['Italic']))
    
    # Build PDF
    try:
        doc.build(story)
        return True
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return False

def generate_pdf_from_history(history: List[Dict[str, str]], doc_type: str = None, mode: str = "template") -> str:
    """
    Main function to generate PDF from chat history.
    
    Args:
        history: Chat history
        doc_type: Document type (breach_notice, legal_notice, etc.)
        mode: "template" for template-based, "full" for AI content-based
    
    Returns the output filename.
    """
    if not REPORTLAB_AVAILABLE:
        print("\nERROR: PDF generation requires 'reportlab' library.")
        print("Install it with: pip install reportlab")
        return None
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full content mode - uses AI-generated answer
    if mode == "full":
        output_file = f"legal_analysis_full_{timestamp}.pdf"
        print(f"\nGenerating FULL CONTENT PDF: {output_file}")
        print("Using complete AI-generated answer from chat history...")
        
        success = generate_full_content_pdf(history, output_file)
        
        if success:
            print(f"\n✓ PDF generated successfully: {output_file}")
            print("  Contains: Full AI analysis with evidence and citations")
            return output_file
        else:
            print("\n✗ PDF generation failed")
            return None
    
    # Template mode - uses extracted data with template
    else:
        # Detect document type if not specified
        if not doc_type:
            last_query = history[-1].get("content", "") if history else ""
            doc_type = detect_document_type(last_query, history)
        
        print(f"\nDetected document type: {doc_type}")
        
        # Extract data from history
        print("Extracting information from chat history...")
        data = extract_data_from_history(history)
        
        # Show extracted data
        print("\n--- EXTRACTED DATA ---")
        for key, value in data.items():
            if value:
                print(f"{key}: {value}")
        
        output_file = f"legal_document_{doc_type}_{timestamp}.pdf"
        
        # Generate PDF based on type
        print(f"\nGenerating INTELLIGENT PDF: {output_file}")
        
        # Use AI to generate appropriate document
        success = generate_intelligent_pdf(history, data, doc_type, output_file)
        
        if success:
            print(f"\n✓ PDF generated successfully: {output_file}")
            print("  Contains: Professional template with extracted data")
            return output_file
        else:
            print("\n✗ PDF generation failed")
            return None

# ---------------------------
# CLI main flow
# ---------------------------
def main():
    history = load_history()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "--clear-history":
            clear_history()
            return
        elif command == "--show-history":
            if history:
                print("\n--- CHAT HISTORY ---")
                for i, msg in enumerate(history, 1):
                    role = msg['role'].upper()
                    content = msg['content']
                    # Show full content, not truncated
                    print(f"\n[{i}] {role}:")
                    print("-" * 70)
                    print(content)
                    print("-" * 70)
            else:
                print("No chat history found.")
            return
        elif command == "--generate-pdf":
            if not history:
                print("No chat history found. Have a conversation first.")
                return
            
            # Check for mode and document type arguments
            mode = "template"  # default
            doc_type = None
            
            for arg in sys.argv[2:]:
                if arg.lower() in ["full", "complete", "detailed"]:
                    mode = "full"
                elif arg.lower() == "template":
                    mode = "template"
                else:
                    doc_type = arg.lower()
            
            output_file = generate_pdf_from_history(history, doc_type, mode)
            if output_file:
                print(f"\nYou can now open: {output_file}")
            return
        elif command == "--help":
            print("\n=== PAKISTANI LEGAL RAG ASSISTANT ===")
            print("\nUsage:")
            print("  python chroma_test.py [query]                    - Ask a legal question")
            print("  python chroma_test.py --show-history             - View FULL chat history")
            print("  python chroma_test.py --clear-history            - Clear chat history")
            print("  python chroma_test.py --generate-pdf             - Generate template-based PDF")
            print("  python chroma_test.py --generate-pdf full        - Generate PDF with FULL AI content")
            print("  python chroma_test.py --generate-pdf [type]      - Generate specific document type")
            print("  python chroma_test.py --help                     - Show this help")
            print("\nPDF Generation Modes:")
            print("  template (default) - Professional template with extracted data")
            print("  full              - Complete AI-generated analysis with evidence")
            print("\nDocument types for template mode:")
            print("  - breach_notice (default)")
            print("  - legal_notice")
            print("  - agreement")
            print("\nExamples:")
            print("  python chroma_test.py \"How to draft a breach notice?\"")
            print("  python chroma_test.py --generate-pdf              # Template-based")
            print("  python chroma_test.py --generate-pdf full         # Full AI content")
            print("  python chroma_test.py --generate-pdf breach_notice")
            print("  python chroma_test.py --show-history              # See COMPLETE answers")
            return
        else:
            query = " ".join(sys.argv[1:])
    else:
        query = "How to draft a notice for breach of contract (payment overdue)?"

    print("\nQuery:", query)
    add_to_history(history, "user", query)
    
    evidences = retrieve_and_filter(query)
    if len(evidences) == 0:
        print("No evidences found. Exiting.")
        return

    print(f"Retrieved {len(evidences)} evidences. Building prompt for LLM...")

    # Print metadata extracted for each evidence (helpful to know if judge/court info exists)
    for idx, ev in enumerate(evidences, start=1):
        md = extract_case_metadata(ev.get('text', ''))
        if md:
            print(f"[METADATA] Evidence {idx} -> {md}")
        else:
            print(f"[METADATA] Evidence {idx} -> (no judge/court/clauses found)")

    # build strict prompt with history
    prompt = build_strict_rag_prompt(query, evidences)
    history_context = format_history_for_prompt(history[:-1])
    prompt = prompt + history_context

    # save prompt for inspection (debugging)
    with open("last_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    # call the LLM
    print("Calling Gemini to generate RAG answer (this will consume tokens)...")
    try:
        final_answer = call_gemini_chat(prompt, model=GEMINI_MODEL, temperature=0.0, max_tokens=2000)
    except Exception as exc:
        print("LLM call failed:", exc)
        return

    add_to_history(history, "assistant", final_answer)

    # Save final answer and print clean output
    with open("final_answer.txt", "w", encoding="utf-8") as f:
        f.write(final_answer)

    print("\n--- FINAL RAG ANSWER (clean) ---\n")
    print(final_answer)
    print("\nFinal answer saved to final_answer.txt")
    print(f"Chat history: {len(history)} messages")

if __name__ == "__main__":
    main()
