"""
rag_core.py — ส่วนประมวลผล RAG (ไม่ขึ้นกับ Streamlit จึงใช้ซ้ำใน evaluate.py ได้)

ขั้นตอน
1. Document Loading & Cleaning  : อ่านไฟล์ .md/.txt ใน data/ และทำความสะอาดข้อความ
2. Chunking                     : แบ่งตามหัวข้อ (## heading) แล้วแบ่งย่อยตามขนาด พร้อม overlap
3. Embedding & Vector Search    : multilingual-e5-small + FAISS (cosine similarity)
4. Keyword Search               : BM25 บนคำที่ตัดด้วย PyThaiNLP (ช่วยคำเฉพาะ เช่น EC, pH, NFT)
5. Hybrid Ranking               : รวมผลสองแบบด้วย Reciprocal Rank Fusion (RRF)
6. Prompt                       : บังคับให้ตอบจาก context เท่านั้น อ้างอิง [n] และตอบ "ไม่พบข้อมูล"
"""

from __future__ import annotations

import glob
import os
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# โมเดลขนาดเล็ก (~118M params) รองรับภาษาไทย/อังกฤษ เหมาะกับหน่วยความจำของ Streamlit Cloud
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"

CHUNK_SIZE = 600      # ความยาวสูงสุดของ chunk (ตัวอักษร)
CHUNK_OVERLAP = 120   # ส่วนที่ซ้อนกันระหว่าง chunk เพื่อไม่ให้ใจความขาดตอน
RRF_K = 60

NO_INFO_TEXT = "ไม่พบข้อมูล"


def llm_extra_args(model: str) -> dict:
    """โมเดลแบบ reasoning บน Groq: ลดเวลาคิดของ gpt-oss และซ่อน <think> ของ qwen ไม่ให้ปนในคำตอบ"""
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    if model.startswith("qwen/"):
        return {"reasoning_format": "hidden"}
    return {}


@dataclass
class Chunk:
    chunk_id: int
    source: str      # ชื่อไฟล์
    title: str       # ชื่อเอกสาร (# heading)
    section: str     # หัวข้อย่อย (## heading)
    text: str        # เนื้อหาของ chunk

    @property
    def embed_text(self) -> str:
        # ใส่ชื่อเอกสารและหัวข้อนำหน้า (contextual header) ช่วยให้ค้นเจอแม่นขึ้น
        return f"{self.title} > {self.section}\n{self.text}"


# ---------------------------------------------------------------------------
# 1) Document loading & cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)  # zero-width chars
    text = text.replace("\u0e4d\u0e32", "\u0e33")  # นิคหิต+อา -> สระอำ
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[ \u00a0]+", " ", text)  # space / non-breaking space
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.md")) + glob.glob(os.path.join(data_dir, "*.txt")))
    docs = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            text = clean_text(f.read())
        if not text:
            continue
        m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
        title = m.group(1).strip() if m else os.path.splitext(os.path.basename(path))[0]
        docs.append({"source": os.path.basename(path), "title": title, "text": text})
    return docs


# ---------------------------------------------------------------------------
# 2) Chunking
# ---------------------------------------------------------------------------
def split_sections(text: str, title: str) -> list[tuple[str, str]]:
    """แบ่งเอกสารตามหัวข้อ ## เพื่อให้แต่ละ chunk มีใจความเรื่องเดียว"""
    sections, heading, buf = [], "บทนำ", []
    for line in text.split("\n"):
        if line.startswith("## "):
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = line[3:].strip(), []
        elif line.startswith("# "):
            continue  # ชื่อเอกสาร เก็บไว้ใน title แล้ว
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))
    return sections or [(title, text)]


def _units(text: str, max_len: int) -> list[str]:
    """แตกข้อความเป็นหน่วยย่อย: บรรทัด -> ถ้ายาวเกินให้แตกตามช่องว่าง (ภาษาไทยเว้นวรรคระหว่างวลี)"""
    units = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_len:
            units.append(line)
            continue
        cur = ""
        for piece in re.split(r"(?<=[ .!?])", line):
            if len(cur) + len(piece) > max_len and cur:
                units.append(cur.strip())
                cur = ""
            cur += piece
        if cur.strip():
            units.append(cur.strip())
    # ข้อความยาวที่ไม่มีช่องว่างเลย: ตัดตามความยาวตรง ๆ
    return [u[i:i + max_len] for u in units for i in range(0, len(u), max_len)]


def split_long_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    units = _units(text, chunk_size)
    chunks, cur = [], []
    for unit in units:
        if cur and len("\n".join(cur + [unit])) > chunk_size:
            chunks.append("\n".join(cur))
            # overlap: ยกหน่วยท้าย ๆ ของ chunk ก่อนหน้ามาด้วย
            carry, size = [], 0
            for u in reversed(cur):
                if size + len(u) > overlap:
                    break
                carry.insert(0, u)
                size += len(u)
            cur = carry
        cur.append(unit)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def chunk_documents(docs: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for section, body in split_sections(doc["text"], doc["title"]):
            for piece in split_long_text(body, chunk_size, overlap):
                chunks.append(Chunk(len(chunks), doc["source"], doc["title"], section, piece))
    return chunks


# ---------------------------------------------------------------------------
# 3-5) Retrieval: FAISS vector search + BM25 keyword search + RRF
# ---------------------------------------------------------------------------
def tokenize_for_bm25(text: str) -> list[str]:
    from pythainlp.tokenize import word_tokenize

    tokens = word_tokenize(text.lower(), engine="newmm", keep_whitespace=False)
    return [t for t in tokens if re.search(r"\w", t)]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], model_name: str = EMBED_MODEL_NAME):
        import faiss
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model = SentenceTransformer(model_name, device="cpu")

        # e5 ต้องใส่ prefix "passage: " / "query: "
        emb = self.model.encode(
            ["passage: " + c.embed_text for c in chunks],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])  # inner product ของเวกเตอร์ normalize = cosine
        self.index.add(emb)

        self.bm25 = BM25Okapi([tokenize_for_bm25(c.embed_text) for c in chunks])

    def search(self, queries: str | list[str], top_k: int = 4, candidates: int = 15) -> list[dict]:
        """ค้นหาด้วยคำถามหนึ่งหรือหลายแบบ (เช่น คำถามเดิม + คำแปลอีกภาษา) แล้วรวมทุก ranking ด้วย RRF"""
        if isinstance(queries, str):
            queries = [queries]
        queries = [q for q in dict.fromkeys(q.strip() for q in queries) if q]
        if not queries or not self.chunks:
            return []
        candidates = min(candidates, len(self.chunks))

        q_emb = self.model.encode(["query: " + q for q in queries], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q_emb, candidates)

        rankings: list[list[int]] = []
        cosine: dict[int, float] = {}
        bm25_best = np.zeros(len(self.chunks))
        for n, query in enumerate(queries):
            rankings.append([int(i) for i in ids[n] if i >= 0])
            for i, s in zip(ids[n], scores[n]):
                if i >= 0:
                    cosine[int(i)] = max(cosine.get(int(i), -1.0), float(s))

            bm25_scores = self.bm25.get_scores(tokenize_for_bm25(query))
            bm25_best = np.maximum(bm25_best, bm25_scores)
            rankings.append([int(i) for i in np.argsort(bm25_scores)[::-1][:candidates] if bm25_scores[i] > 0])

        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        # ความหลากหลาย: เลือกได้ 1 chunk ต่อหัวข้อ (chunk ติดกันในหัวข้อเดียวมี overlap ซ้ำกันอยู่แล้ว)
        best, seen = [], set()
        for i in sorted(fused, key=fused.get, reverse=True):
            key = (self.chunks[i].source, self.chunks[i].section)
            if key in seen:
                continue
            seen.add(key)
            best.append(i)
            if len(best) == top_k:
                break
        missing = [i for i in best if i not in cosine]
        if missing:  # คำนวณ cosine ให้ chunk ที่มาจาก BM25 อย่างเดียว
            vecs = np.vstack([self.index.reconstruct(i) for i in missing])
            for i, s in zip(missing, (vecs @ q_emb.T).max(axis=1)):
                cosine[i] = float(s)
        return [
            {"chunk": self.chunks[i], "rrf": fused[i], "cosine": cosine[i], "bm25": float(bm25_best[i])}
            for i in best
        ]


# ---------------------------------------------------------------------------
# 6) Prompt engineering
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""คุณคือ "HydroBuddy" ผู้ช่วยตอบคำถามเรื่องการปลูกผักไฮโดรโปนิกส์ในบ้าน

กฎที่ต้องปฏิบัติอย่างเคร่งครัด:
1. ตอบโดยใช้ข้อมูลจาก <context> ที่ให้มาเท่านั้น ห้ามใช้ความรู้ภายนอก ห้ามเดา และห้ามแต่งตัวเลขเพิ่ม
2. ทุกประโยคที่ใช้ข้อมูลจากเอกสาร ให้ใส่เลขอ้างอิงในวงเล็บเหลี่ยมท้ายประโยค เช่น [1] หรือ [1][3] ตามหมายเลขเอกสารใน <context>
3. ถ้า <context> ไม่มีข้อมูลที่ตอบคำถามได้ หรือคำถามไม่เกี่ยวกับการปลูกพืชไฮโดรโปนิกส์ ให้ตอบขึ้นต้นด้วยคำว่า "{NO_INFO_TEXT}" ตามด้วยคำอธิบายสั้น ๆ ว่าเอกสารไม่ครอบคลุมเรื่องนี้ และห้ามตอบเนื้อหาจากความรู้ทั่วไป
4. ถ้ามีข้อมูลเพียงบางส่วน ให้ตอบเฉพาะส่วนที่มีในเอกสาร แล้วบอกว่าส่วนที่เหลือไม่พบในเอกสาร
5. ตอบเป็นภาษาเดียวกับคำถาม (ถามไทยตอบไทย ถามอังกฤษตอบอังกฤษ แต่ถ้าไม่พบข้อมูลให้ขึ้นต้นด้วย "{NO_INFO_TEXT}" เสมอ)
6. ตอบกระชับ เป็นมิตร ใช้หัวข้อย่อยหรือขั้นตอนเมื่อเหมาะสม
7. ปิดท้ายคำตอบด้วยบรรทัด "แหล่งอ้างอิง:" แล้วระบุชื่อไฟล์ที่ใช้ตอบจริง ถ้าไม่พบข้อมูลห้ามใส่บรรทัดนี้
8. ใช้เลขอ้างอิงรูปแบบ [1] เท่านั้น ห้ามใช้รูปแบบอื่น เช่น 【1】"""

QUERY_PROMPT = """คุณเป็นผู้ช่วยเตรียมคำค้นสำหรับระบบค้นหาเอกสาร (ห้ามตอบคำถามเอง)
1. "standalone": เขียนคำถามล่าสุดใหม่ให้สมบูรณ์ในตัวเอง โดยแทนคำสรรพนามหรือคำที่อ้างถึงสิ่งที่คุยไปแล้วด้วยชื่อจริงจากประวัติการสนทนา ใช้ภาษาเดียวกับคำถามล่าสุด (ถ้าไม่มีประวัติหรือสมบูรณ์อยู่แล้ว ให้คงคำถามเดิม)
2. "translation": แปล standalone เป็นอีกภาษา (ไทย -> อังกฤษ, อังกฤษ -> ไทย) เพื่อค้นเอกสารได้ทั้งสองภาษา
ตอบเป็น JSON เท่านั้น: {"standalone": "...", "translation": "..."}"""

HISTORY_CHARS = 500


def expand_query(client, model: str, history: list[dict], question: str) -> tuple[str, str | None]:
    """Query rewriting + cross-lingual query expansion ด้วย LLM หนึ่งครั้ง
    คืนค่า (คำถามที่สมบูรณ์ในตัวเอง, คำแปลอีกภาษา) ถ้าเรียกไม่สำเร็จจะคืนคำถามเดิม"""
    import json

    convo = "\n".join(f"{m['role']}: {m['content'][:HISTORY_CHARS]}" for m in history)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QUERY_PROMPT},
                {"role": "user", "content": f"ประวัติการสนทนา:\n{convo or '(ไม่มี)'}\n\nคำถามล่าสุด: {question}"},
            ],
            temperature=0,
            max_tokens=600,
            response_format={"type": "json_object"},
            **llm_extra_args(model),
        )
        data = json.loads(resp.choices[0].message.content)
        standalone = (data.get("standalone") or "").strip() or question
        translation = (data.get("translation") or "").strip() or None
        return standalone, translation
    except Exception:
        return question, None


def format_context(results: list[dict]) -> str:
    blocks = []
    for n, r in enumerate(results, 1):
        c: Chunk = r["chunk"]
        blocks.append(f"[{n}] (ไฟล์: {c.source} | หัวข้อ: {c.title} > {c.section})\n{c.text}")
    return "\n\n".join(blocks)


def answer_language(question: str) -> str:
    return "ภาษาไทย" if re.search(r"[฀-๿]", question) else "English"


def build_user_prompt(question: str, results: list[dict]) -> str:
    return f"""<context>
{format_context(results)}
</context>

คำถาม: {question}

ตอบตามกฎใน system prompt โดยใช้เฉพาะข้อมูลใน <context> และอ้างอิงด้วย [หมายเลข]
ภาษาของคำตอบ: {answer_language(question)} (ยกเว้นคำว่า "{NO_INFO_TEXT}" ที่ต้องขึ้นต้นเสมอเมื่อไม่พบข้อมูล)"""


def normalize_citations(text: str) -> str:
    text = text.replace("【", "[").replace("】", "]")
    return re.sub(r"\[+\s*(\d+)\s*\]+", r"[\1]", text)  # [[1]] -> [1]


def is_no_info(answer: str) -> bool:
    return NO_INFO_TEXT in answer.lstrip("*_# ")[: len(NO_INFO_TEXT) + 10]


def clean_answer(answer: str) -> str:
    """เก็บกวาดคำตอบสุดท้าย: แปลงรูปแบบอ้างอิง และลบบรรทัด "แหล่งอ้างอิง:" ที่ว่างหรือไม่มีความหมาย"""
    answer = normalize_citations(answer).strip()
    lines = answer.split("\n")
    # บรรทัด "แหล่งอ้างอิง:" ท้ายคำตอบที่ไม่มีชื่อไฟล์ตามมา
    while lines and re.fullmatch(r"[\W_]*แหล่งอ้างอิง[\W_]*(ไม่พบข้อมูล|ไม่มี|-)?[\W_]*", lines[-1].strip()):
        lines.pop()
    if is_no_info(answer):  # ปฏิเสธแล้ว ไม่ต้องมีแหล่งอ้างอิง
        lines = [re.sub(r"\s*[*_]*แหล่งอ้างอิง[*_]*\s*:.*$", "", l) for l in lines]
        lines = [l for l in lines if l.strip()]
    return "\n".join(lines).strip()
