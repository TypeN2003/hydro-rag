"""
HydroBuddy 🥬 — แชตบอต RAG ผู้ช่วยปลูกผักไฮโดรโปนิกส์ในบ้าน
รัน: streamlit run app.py   (ต้องตั้งค่า GROQ_API_KEY ใน .streamlit/secrets.toml หรือ Secrets ของ Streamlit Cloud)
"""

import os

import streamlit as st

from rag_core import (
    EMBED_MODEL_NAME,
    NO_INFO_TEXT,
    SYSTEM_PROMPT,
    HybridRetriever,
    build_user_prompt,
    chunk_documents,
    expand_query,
    llm_extra_args,
    clean_answer,
    is_no_info,
    load_documents,
    normalize_citations,
)

st.set_page_config(page_title="HydroBuddy - ผู้ช่วยปลูกผักไฮโดรโปนิกส์", page_icon="🥬", layout="centered")

LLM_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
REWRITE_MODEL = "openai/gpt-oss-20b"
HISTORY_TURNS = 4  # จำนวนข้อความย้อนหลังที่ส่งให้ LLM เพื่อคุยต่อเนื่อง

EXAMPLE_QUESTIONS = [
    "ผสมปุ๋ย A กับ B ยังไงให้ถูกต้อง?",
    "ค่า pH ที่เหมาะกับผักสลัดคือเท่าไหร่ ปรับยังไง?",
    "ใบอ่อนเหลืองแต่เส้นใบยังเขียว เกิดจากอะไร?",
    "มือใหม่ปลูกในคอนโดควรเริ่มระบบไหนดี?",
    "How do I prevent root rot?",
    "ปลูกทุเรียนในดินเหนียวต้องใส่ปุ๋ยอะไร?",
]


# ---------------------------------------------------------------------------
# โหลดเอกสาร / โมเดล / index เพียงครั้งเดียว (cache_resource ใช้ร่วมกันทุก session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลดเอกสารและสร้าง Vector Index (ครั้งแรกใช้เวลาประมาณ 1 นาที)...")
def get_retriever():
    docs = load_documents()
    chunks = chunk_documents(docs)
    return docs, HybridRetriever(chunks)


def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")


@st.cache_resource
def get_groq_client(api_key: str):
    from groq import Groq

    return Groq(api_key=api_key)


def stream_answer(client, model: str, history: list[dict], question: str, results: list[dict]):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": build_user_prompt(question, results)})
    stream = client.chat.completions.create(
        model=model, messages=messages, temperature=0.2, max_tokens=2048, stream=True, **llm_extra_args(model)
    )
    for part in stream:
        if not part.choices:
            continue
        delta = part.choices[0].delta.content
        if delta:
            yield delta


def llm_history(messages: list[dict]) -> list[dict]:
    """ประวัติที่ส่งให้ LLM: ตัดข้อความ error ออก และเอาเฉพาะ HISTORY_TURNS ข้อความล่าสุด"""
    return [{"role": m["role"], "content": m["content"]} for m in messages if not m.get("error")][-HISTORY_TURNS:]


def render_sources(sources: list[dict], answered: bool):
    if not sources:
        return
    label = "📚 เอกสารอ้างอิงที่ใช้ตอบ" if answered else "📚 เอกสารที่ค้นพบ (ไม่มีคำตอบของคำถามนี้)"
    with st.expander(f"{label} ({len(sources)})", expanded=False):
        for n, s in enumerate(sources, 1):
            st.markdown(
                f"**[{n}] `{s['source']}`** — {s['title']} › *{s['section']}*  \n"
                f"<small>cosine = {s['cosine']:.3f} · BM25 = {s['bm25']:.2f} · RRF = {s['rrf']:.4f}</small>",
                unsafe_allow_html=True,
            )
            st.caption(s["text"])
            if n < len(sources):
                st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
docs, retriever = get_retriever()

with st.sidebar:
    st.header("🥬 HydroBuddy")
    st.caption("ผู้ช่วยตอบคำถามการปลูกผักไฮโดรโปนิกส์ในบ้าน ตอบจากคลังเอกสารเท่านั้น พร้อมแหล่งอ้างอิง")

    st.subheader("⚙️ ตั้งค่า")
    model = st.selectbox("LLM (Groq)", LLM_MODELS, index=0)
    top_k = st.slider("จำนวนเอกสารที่ค้นมาใช้ตอบ (Top-K)", 2, 8, 4)
    min_score = st.slider(
        "เกณฑ์ความเกี่ยวข้องขั้นต่ำ (cosine)",
        0.60, 0.90, 0.75, 0.01,
        help="ถ้าเอกสารที่ค้นได้มีคะแนนต่ำกว่าเกณฑ์นี้ทั้งหมด ระบบจะตอบ 'ไม่พบข้อมูล' ทันทีโดยไม่เรียก LLM",
    )
    use_rewrite = st.toggle(
        "Query Rewriting + ค้นสองภาษา (ไทย/อังกฤษ)",
        value=True,
        help="เขียนคำถามต่อเนื่องให้สมบูรณ์ และแปลคำค้นเป็นอีกภาษา เพื่อค้นได้ทั้งเอกสารไทยและอังกฤษ",
    )

    if st.button("🗑️ ล้างประวัติแชต", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.subheader("💡 ตัวอย่างคำถาม")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            st.session_state.pending_question = q

    st.subheader("📂 คลังเอกสาร")
    total_chars = sum(len(d["text"]) for d in docs)
    c1, c2, c3 = st.columns(3)
    c1.metric("ไฟล์", len(docs))
    c2.metric("Chunks", len(retriever.chunks))
    c3.metric("ตัวอักษร", f"{total_chars/1000:.1f}k")
    with st.expander("รายชื่อเอกสาร"):
        for d in docs:
            st.markdown(f"- `{d['source']}` — {d['title']}")
    st.caption(f"Embedding: `{EMBED_MODEL_NAME}` · Vector DB: FAISS · Keyword: BM25 + PyThaiNLP")


# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------
st.title("🥬 HydroBuddy")
st.markdown("ถามเรื่อง **การปลูกผักไฮโดรโปนิกส์** ได้ทั้งภาษาไทยและอังกฤษ — ระบบ, ปุ๋ย A-B, ค่า pH/EC, การเพาะกล้า, โรคและแมลง, การแก้ปัญหา, การเก็บเกี่ยว")

api_key = get_api_key()
if not api_key:
    st.error("ไม่พบ `GROQ_API_KEY` — กรุณาตั้งค่าใน `.streamlit/secrets.toml` (เครื่องตัวเอง) หรือในเมนู Settings › Secrets ของ Streamlit Community Cloud")
    st.stop()
client = get_groq_client(api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🥬"):
        st.markdown("สวัสดีครับ ผม **HydroBuddy** 🌱 มีอะไรเกี่ยวกับการปลูกผักไร้ดินให้ช่วยไหมครับ? ลองกดตัวอย่างคำถามที่แถบด้านซ้ายก็ได้ครับ")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🥬" if msg["role"] == "assistant" else "🧑‍🌾"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("search_query"):
                st.caption(f"🔎 คำค้นที่ใช้: {msg['search_query']}")
            render_sources(msg.get("sources", []), msg.get("answered", True))

question = (st.chat_input("พิมพ์คำถามเกี่ยวกับการปลูกผักไฮโดรโปนิกส์...") or "").strip()
pending = st.session_state.pop("pending_question", None)
if not question and pending:
    question = pending

if question:
    history = llm_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🥬"):
        with st.spinner("กำลังค้นหาเอกสาร..."):
            if use_rewrite:
                search_query, translation = expand_query(client, REWRITE_MODEL, history, question)
            else:
                search_query, translation = question, None
            # ค้นด้วยคำถามเดิม + คำถามที่เขียนใหม่ + คำแปล แล้วรวมผลด้วย RRF (ทนต่อการเขียนใหม่ที่คลาดเคลื่อน)
            results = retriever.search([question, search_query, translation or ""], top_k=top_k)

        query_note = " · ".join(q for q in (search_query, translation) if q and q != question)
        if query_note:
            st.caption(f"🔎 คำค้นที่ใช้: {query_note}")

        error = False
        best = max((r["cosine"] for r in results), default=0.0)
        if best < min_score:
            # Retrieval guard: เอกสารไม่เกี่ยวข้องพอ ไม่ต้องเรียก LLM
            answer = f"{NO_INFO_TEXT}ในคลังเอกสารที่เกี่ยวข้องกับคำถามนี้ครับ (คะแนนความเกี่ยวข้องสูงสุด {best:.2f} ต่ำกว่าเกณฑ์ {min_score:.2f}) ลองถามเรื่องการปลูกผักไฮโดรโปนิกส์ดูนะครับ"
            st.markdown(answer)
        else:
            placeholder = st.empty()
            answer = ""
            try:
                for delta in stream_answer(client, model, history, search_query, results):
                    answer += delta
                    placeholder.markdown(normalize_citations(answer) + "▌")
                answer = clean_answer(answer)
                if not answer:
                    raise RuntimeError("LLM ไม่ได้ส่งคำตอบกลับมา (คำตอบว่าง) ลองถามใหม่หรือเปลี่ยนโมเดล")
                placeholder.markdown(answer)
            except Exception as e:
                error = True
                answer = f"⚠️ เรียกใช้ LLM ไม่สำเร็จ: {e}"
                placeholder.error(answer)

        answered = not error and not is_no_info(answer)
        sources = [
            {
                "source": r["chunk"].source,
                "title": r["chunk"].title,
                "section": r["chunk"].section,
                "text": r["chunk"].text,
                "cosine": r["cosine"],
                "bm25": r["bm25"],
                "rrf": r["rrf"],
            }
            for r in results
        ]
        render_sources(sources, answered)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "answered": answered,
            "error": error,
            "search_query": query_note or None,
        }
    )
