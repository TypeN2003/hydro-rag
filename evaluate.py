"""
evaluate.py — ทดสอบระบบด้วย test_questions.csv

    python evaluate.py              # ทดสอบเฉพาะ retrieval (ไม่ต้องใช้ API key)
    python evaluate.py --llm        # ทดสอบ retrieval + ให้ LLM ตอบจริง (ต้องมี GROQ_API_KEY ใน env)

Retrieval hit = ไฟล์ที่คาดหวังอยู่ใน Top-K ผลการค้นหา
Refusal check = คำถามที่ไม่มีคำตอบ ต้องได้คำตอบขึ้นต้นด้วย "ไม่พบข้อมูล"
"""

import argparse
import csv
import os

from rag_core import (
    SYSTEM_PROMPT,
    HybridRetriever,
    build_user_prompt,
    chunk_documents,
    clean_answer,
    expand_query,
    is_no_info,
    llm_extra_args,
    load_documents,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    args = ap.parse_args()

    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} files, {sum(len(d['text']) for d in docs):,} chars, {len(chunks)} chunks\n")
    retriever = HybridRetriever(chunks)

    client = None
    if args.llm:
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])

    with open(os.path.join(HERE, "test_questions.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    hits = answerable = refusals_ok = unanswerable = 0
    for row in rows:
        queries = [row["question"]]
        if client:  # ค้นแบบเดียวกับหน้าเว็บ: คำถาม + คำแปลอีกภาษา
            queries = list(expand_query(client, "openai/gpt-oss-20b", [], row["question"]))
        results = retriever.search([q for q in queries if q], top_k=args.top_k)
        found = [r["chunk"].source for r in results]
        best = max(r["cosine"] for r in results)
        line = f"#{row['id']:>2} best_cos={best:.3f} top={found[0]}"

        if row["answerable"] == "yes":
            answerable += 1
            expected = [s.strip() for s in row["source_file"].split("/")]
            hit = any(e in found for e in expected)
            hits += hit
            line += f"  retrieval={'HIT ' if hit else 'MISS'}"
        else:
            unanswerable += 1

        if client:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(row["question"], results)},
                ],
                temperature=0,
                **llm_extra_args(args.model),
            )
            answer = clean_answer(resp.choices[0].message.content or "")
            refused = is_no_info(answer)
            if row["answerable"] == "no":
                refusals_ok += refused
                line += f"  refusal={'OK' if refused else 'FAIL'}"
            print(line)
            print(f"    Q: {row['question']}\n    A: {answer[:300]}\n")
        else:
            print(line + f"  | {row['question']}")

    print(f"\nRetrieval hit-rate @top{args.top_k}: {hits}/{answerable}")
    if client:
        print(f"Correct refusals: {refusals_ok}/{unanswerable}")


if __name__ == "__main__":
    main()
