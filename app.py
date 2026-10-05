"""
HydroBuddy 🥬 — แชตบอต RAG ผู้ช่วยปลูกผักไฮโดรโปนิกส์ในบ้าน
รัน: streamlit run app.py
ต้องตั้งค่า GROQ_API_KEY และ/หรือ GEMINI_API_KEY ใน .streamlit/secrets.toml หรือ Secrets ของ Streamlit Cloud
"""

import os
import time

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

# ผู้ให้บริการ LLM: ใช้ตัวที่เลือกเป็นหลัก ถ้าถูกบล็อก (403) / เกินโควตา (429) / ล่ม (5xx) จะสลับไปอีกตัวอัตโนมัติ
PROVIDERS = {
    "Groq": {
        "secret": "GROQ_API_KEY",
        "models": ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
        "rewrite_model": "openai/gpt-oss-20b",
    },
    "Gemini": {
        "secret": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-flash-latest", "gemini-flash-lite-latest"],
        "rewrite_model": "gemini-flash-lite-latest",
    },
}
FALLBACK_STATUS = {403, 429, 500, 502, 503, 504}
BLOCK_SECONDS = 600  # ข้ามผู้ให้บริการที่ตอบ 403 ไป 10 นาที แล้วค่อยลองใหม่
HISTORY_TURNS = 4  # จำนวนข้อความย้อนหลังที่ส่งให้ LLM เพื่อคุยต่อเนื่อง

# คำถามที่เอกสารมีคำตอบ
ANSWERABLE_EXAMPLES = [
    "ผสมปุ๋ย A กับ B ยังไงให้ถูกต้อง?",
    "ค่า pH ที่เหมาะกับผักสลัดคือเท่าไหร่ ปรับยังไง?",
    "ใบอ่อนเหลืองแต่เส้นใบยังเขียว เกิดจากอะไร?",
    "มือใหม่ปลูกในคอนโดควรเริ่มระบบไหนดี?",
    "กรีนโอ๊คใช้เวลากี่วันถึงเก็บเกี่ยวได้?",
    "How do I prevent root rot?",
]

# คำถามที่เอกสารไม่มีคำตอบ (ทดสอบว่าระบบตอบ "ไม่พบข้อมูล")
UNANSWERABLE_EXAMPLES = [
    "ปลูกทุเรียนในดินเหนียวต้องใส่ปุ๋ยอะไร?",
    "ราคาผักสลัดขายส่งที่ตลาดไทวันนี้กิโลละเท่าไร?",
    "What is the best setup for growing cannabis indoors?",
]


# ---------------------------------------------------------------------------
# โหลดเอกสาร / โมเดล / index เพียงครั้งเดียว (cache_resource ใช้ร่วมกันทุก session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลดเอกสารและสร้าง Vector Index (ครั้งแรกใช้เวลาประมาณ 1 นาที)...")
def get_retriever():
    docs = load_documents()
    chunks = chunk_documents(docs)
    return docs, HybridRetriever(chunks)


def get_secret(name: str):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


@st.cache_resource
def get_client(provider: str, api_key: str):
    if provider == "Groq":
        from groq import Groq

        return Groq(api_key=api_key)
    from openai import OpenAI  # Gemini มี API แบบเดียวกับ OpenAI

    return OpenAI(api_key=api_key, base_url=PROVIDERS[provider]["base_url"])


@st.cache_resource
def _blocked_at() -> dict:
    return {}


def blocked_providers() -> set:
    """ผู้ให้บริการที่ตอบ 403 (บล็อก IP ของเซิร์ฟเวอร์นี้) ภายใน BLOCK_SECONDS ล่าสุด จำร่วมกันทุก session
    พ้นเวลาแล้วจะกลับไปลองใหม่ เผื่อผู้ให้บริการปลดบล็อกแล้ว"""
    now = time.time()
    return {p for p, t in _blocked_at().items() if now - t < BLOCK_SECONDS}


def provider_chain(selected: str, clients: dict) -> list[str]:
    """ลำดับผู้ให้บริการที่จะลอง: ตัวที่เลือกก่อน แล้วตามด้วยตัวอื่น โดยข้ามตัวที่ถูกบล็อก (ถ้ายังมีตัวอื่นเหลือ)"""
    order = [selected] + [p for p in clients if p != selected]
    usable = [p for p in order if p not in blocked_providers()]
    return usable or order


def should_fallback(e: Exception, provider: str) -> bool:
    status = getattr(e, "status_code", None)
    if status == 403:
        _blocked_at()[provider] = time.time()
    return status in FALLBACK_STATUS


def error_message(e: Exception, provider: str) -> str:
    status = getattr(e, "status_code", None)
    if status == 429:
        return f"⚠️ ใช้งาน {provider} เกินโควตาชั่วคราว (rate limit) กรุณารอประมาณ 1 นาทีแล้วลองใหม่ หรือเปลี่ยนโมเดลที่แถบด้านซ้าย"
    if status == 403:
        return f"⚠️ {provider} ปฏิเสธการเชื่อมต่อจากเซิร์ฟเวอร์นี้ (403) กรุณาเพิ่ม API key ของผู้ให้บริการอื่นใน Secrets"
    return f"⚠️ เรียกใช้ LLM ไม่สำเร็จ ({provider}): {e}"


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
clients = {p: get_client(p, key) for p, cfg in PROVIDERS.items() if (key := get_secret(cfg["secret"]))}

with st.sidebar:
    st.header("🥬 HydroBuddy")
    st.caption("ผู้ช่วยตอบคำถามการปลูกผักไฮโดรโปนิกส์ในบ้าน ตอบจากคลังเอกสารเท่านั้น พร้อมแหล่งอ้างอิง")

    st.subheader("⚙️ ตั้งค่า")
    model_options = [(p, m) for p in clients for m in PROVIDERS[p]["models"]]
    choice = st.selectbox(
        "LLM",
        model_options,
        format_func=lambda o: f"{o[0]} · {o[1]}",
        help="ถ้าผู้ให้บริการที่เลือกใช้งานไม่ได้ (403 / 429 / 5xx) ระบบจะสลับไปใช้อีกผู้ให้บริการโดยอัตโนมัติ",
    )
    if blocked_providers() & set(clients):
        st.caption(f"⚠️ {', '.join(sorted(blocked_providers()))} ถูกบล็อกจากเซิร์ฟเวอร์นี้ จึงใช้ผู้ให้บริการอื่นแทนอัตโนมัติ (จะลองใหม่ทุก 10 นาที)")
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
    st.caption("✅ มีคำตอบในเอกสาร")
    for q in ANSWERABLE_EXAMPLES:
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            st.session_state.pending_question = q
    st.caption("❌ ไม่มีคำตอบในเอกสาร (ระบบควรตอบว่า \"ไม่พบข้อมูล\")")
    for q in UNANSWERABLE_EXAMPLES:
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            st.session_state.pending_question = q

    st.subheader("📂 คลังเอกสาร")
    total_chars = sum(len(d["text"]) for d in docs)
    c1, c2, c3 = st.columns(3)
    c1.metric("ไฟล์", len(docs))
    c2.metric("Chunks", len(retriever.chunks))
    c3.metric("ตัวอักษร", f"{total_chars // 1000}k")
    with st.expander("รายชื่อเอกสาร"):
        for d in docs:
            st.markdown(f"- `{d['source']}` — {d['title']}")
    st.caption(f"Embedding: `{EMBED_MODEL_NAME}` · Vector DB: FAISS · Keyword: BM25 + PyThaiNLP")


# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------
st.title("🥬 HydroBuddy")
st.markdown("ถามเรื่อง **การปลูกผักไฮโดรโปนิกส์** ได้ทั้งภาษาไทยและอังกฤษ — ระบบ, ปุ๋ย A-B, ค่า pH/EC, การเพาะกล้า, โรคและแมลง, การแก้ปัญหา, การเก็บเกี่ยว")

if not clients:
    st.error("ไม่พบ API key — กรุณาตั้งค่า `GROQ_API_KEY` และ/หรือ `GEMINI_API_KEY` ใน `.streamlit/secrets.toml` (เครื่องตัวเอง) หรือในเมนู Settings › Secrets ของ Streamlit Community Cloud")
    st.stop()
selected_provider, selected_model = choice


def llm_attempts() -> list[tuple[str, str]]:
    """ลำดับ (ผู้ให้บริการ, โมเดล) ที่จะลอง: โมเดลที่เลือกก่อน แล้วโมเดลอื่นของผู้ให้บริการเดียวกัน แล้วจึงผู้ให้บริการอื่น"""
    attempts = []
    for p in provider_chain(selected_provider, clients):
        models = PROVIDERS[p]["models"]
        if p == selected_provider:
            models = [selected_model] + [m for m in models if m != selected_model]
        attempts += [(p, m) for m in models]
    return attempts

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
            if msg.get("llm"):
                st.caption(f"🤖 ตอบโดย {msg['llm']}")
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
            search_query, translation = question, None
            if use_rewrite:
                for p in provider_chain(selected_provider, clients):
                    try:
                        search_query, translation = expand_query(
                            clients[p], PROVIDERS[p]["rewrite_model"], history, question, raise_errors=True
                        )
                        break
                    except Exception as e:
                        if not should_fallback(e, p):
                            break
            # ค้นด้วยคำถามเดิม + คำถามที่เขียนใหม่ + คำแปล แล้วรวมผลด้วย RRF (ทนต่อการเขียนใหม่ที่คลาดเคลื่อน)
            results = retriever.search([question, search_query, translation or ""], top_k=top_k)

        query_note = " · ".join(q for q in (search_query, translation) if q and q != question)
        if query_note:
            st.caption(f"🔎 คำค้นที่ใช้: {query_note}")

        error = False
        llm_used = None
        best = max((r["cosine"] for r in results), default=0.0)
        if best < min_score:
            # Retrieval guard: เอกสารไม่เกี่ยวข้องพอ ไม่ต้องเรียก LLM
            answer = f"{NO_INFO_TEXT}ในคลังเอกสารที่เกี่ยวข้องกับคำถามนี้ครับ (คะแนนความเกี่ยวข้องสูงสุด {best:.2f} ต่ำกว่าเกณฑ์ {min_score:.2f}) ลองถามเรื่องการปลูกผักไฮโดรโปนิกส์ดูนะครับ"
            st.markdown(answer)
        else:
            placeholder = st.empty()
            answer, last_error = "", None
            for p, m in llm_attempts():
                if p in blocked_providers() and len(blocked_providers() & set(clients)) < len(clients):
                    continue  # ผู้ให้บริการนี้ถูกบล็อกระหว่างทาง ข้ามไปตัวอื่น
                try:
                    for delta in stream_answer(clients[p], m, history, search_query, results):
                        answer += delta
                        placeholder.markdown(normalize_citations(answer) + "▌")
                    llm_used, last_error = f"{p} · {m}", None
                    break
                except Exception as e:
                    last_error = (e, p)
                    # สลับผู้ให้บริการได้เฉพาะเมื่อยังไม่มีคำตอบออกมา และเป็น error ที่สลับแล้วน่าจะหาย
                    if answer or not should_fallback(e, p):
                        break

            answer = clean_answer(answer)
            if last_error or not answer:
                error = True
                answer = error_message(*last_error) if last_error else "⚠️ LLM ไม่ได้ส่งคำตอบกลับมา (คำตอบว่าง) ลองถามใหม่หรือเปลี่ยนโมเดล"
                placeholder.error(answer)
            else:
                placeholder.markdown(answer)
                note = f"🤖 ตอบโดย {llm_used}"
                if llm_used != f"{selected_provider} · {selected_model}":
                    note += f" (สลับอัตโนมัติ เพราะ {selected_provider} · {selected_model} ใช้งานไม่ได้ชั่วคราว)"
                    llm_used += " (สำรอง)"
                st.caption(note)

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
            "llm": llm_used,
            "search_query": query_note or None,
        }
    )
