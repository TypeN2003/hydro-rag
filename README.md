# 🥬 HydroBuddy — ผู้ช่วยปลูกผักไฮโดรโปนิกส์ในบ้าน (RAG Chatbot)

แชตบอตตอบคำถามเรื่อง **การปลูกผักไฮโดรโปนิกส์ (ปลูกพืชไร้ดิน)** สำหรับมือใหม่ที่อยากปลูกผักกินเองที่บ้านหรือคอนโด
ใช้เทคนิค **RAG (Retrieval-Augmented Generation)** ค้นข้อมูลจากคลังเอกสารภาษาไทย/อังกฤษ แล้วให้ LLM ตอบ **จากเอกสารเท่านั้น** พร้อมแสดงแหล่งอ้างอิงทุกครั้ง และตอบว่า **"ไม่พบข้อมูล"** เมื่อเอกสารไม่มีคำตอบ

- 🌐 **Demo:** https://hydrobuddy-rag.streamlit.app/
- 💻 **Source:** https://github.com/TypeN2003/hydro-rag
- 🧠 **LLM:** Groq API (`openai/gpt-oss-120b`)
- 🔎 **Retrieval:** `intfloat/multilingual-e5-small` + FAISS + BM25 (PyThaiNLP) รวมผลด้วย Reciprocal Rank Fusion

---

## 1. แนวคิดของ Domain

การปลูกผักไฮโดรโปนิกส์กำลังเป็นที่นิยมในคนเมือง แต่มือใหม่มักเจอปัญหาเฉพาะทาง เช่น ผสมปุ๋ย A-B ผิด ค่า pH/EC ไม่เหมาะ ผักใบเหลือง รากเน่า ข้อมูลเหล่านี้กระจัดกระจายตามเว็บและกลุ่มเฟซบุ๊ก
HydroBuddy จึงรวบรวมความรู้เป็นคลังเอกสารเดียว และให้ AI ช่วยตอบแบบอ้างอิงได้ เหมือนมี "พี่เลี้ยงฟาร์ม" ที่ตอบตามคู่มือจริง ไม่มั่ว

ตัวอย่างคำถามที่ตอบได้: วิธีผสมปุ๋ย, ค่า pH/EC ที่เหมาะสม, เลือกระบบ NFT/DWC/Kratky, เพาะเมล็ด, อาการขาดธาตุ, โรคและแมลง, การเก็บเกี่ยว, ต้นทุน

## 2. โครงสร้างไฟล์

```
hydro-rag/
├── app.py                    # Streamlit App (หน้าแชต)
├── rag_core.py               # Loading, Cleaning, Chunking, Embedding, FAISS, BM25, Prompt
├── evaluate.py               # สคริปต์ทดสอบด้วย test_questions.csv
├── requirements.txt
├── test_questions.csv        # คำถามทดสอบ 15 ข้อ (ไม่มีคำตอบในเอกสาร 3 ข้อ)
├── data/                     # คลังเอกสารความรู้ 11 ไฟล์ (~26,000 ตัวอักษร)
│   ├── 01_intro_hydroponics.md
│   ├── 02_hydroponic_systems.md
│   ├── 03_nutrient_solution.md
│   ├── 04_ph_ec_management.md
│   ├── 05_seed_germination.md
│   ├── 06_vegetable_guide.md
│   ├── 07_light_environment.md
│   ├── 08_pests_diseases.md
│   ├── 09_troubleshooting.md
│   ├── 10_harvest_storage_cost.md
│   └── 11_english_faq.md
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example  # ตัวอย่าง (ไฟล์จริง secrets.toml ถูก .gitignore)
└── .gitignore
```

## 3. สถาปัตยกรรม RAG

```
คำถามผู้ใช้ ──► Query Rewriting + แปลคำค้นไทย↔อังกฤษ (openai/gpt-oss-20b)
               │
               ├─► (ค้นทั้งคำถามเดิมและคำแปล)
               ├─► Vector Search: e5-small embedding ─► FAISS IndexFlatIP (cosine)
               └─► Keyword Search: PyThaiNLP ตัดคำ ─► BM25
                          │
                  Reciprocal Rank Fusion ─► Top-K chunks
                          │
            Retrieval Guard: cosine สูงสุด < เกณฑ์ ─► ตอบ "ไม่พบข้อมูล" ทันที
                          │
         Prompt (System rules + context [1..K] + ประวัติแชต) ─► Groq LLM ─► คำตอบ + [อ้างอิง]
```

| เทคนิค | การนำไปใช้ |
|---|---|
| **Document Loading & Cleaning** | อ่าน `.md/.txt` ใน `data/`, Unicode NFC, ลบ zero-width chars, แก้ "ํา"→"ำ", ยุบช่องว่าง/บรรทัดว่างซ้ำ |
| **Chunking** | แบ่งตามหัวข้อ `##` ก่อน (1 chunk = 1 ใจความ) ถ้ายาวเกิน 600 ตัวอักษรแบ่งย่อยตามบรรทัด/วลี พร้อม overlap 120 ตัวอักษร และใส่ "ชื่อเอกสาร > หัวข้อ" นำหน้า chunk ก่อน embed |
| **Embedding** | `intfloat/multilingual-e5-small` (รองรับไทย-อังกฤษ, ขนาดเล็กเหมาะ Streamlit Cloud) ใช้ prefix `passage:` / `query:` และ normalize เวกเตอร์ |
| **Vector DB** | FAISS `IndexFlatIP` (inner product ของเวกเตอร์ normalize = cosine similarity) |
| **Hybrid Search** | BM25 บนคำที่ตัดด้วย PyThaiNLP (`newmm`) ช่วยคำเฉพาะ เช่น EC, pH, NFT, Kratky แล้วรวมกับผล vector ด้วย RRF และเลือกได้ไม่เกิน 1 chunk ต่อหัวข้อ เพื่อไม่ให้ผลลัพธ์ซ้ำกัน |
| **Prompt Engineering** | System prompt บังคับให้ตอบจาก `<context>` เท่านั้น, อ้างอิง `[n]`, ตอบ "ไม่พบข้อมูล" เมื่อไม่มีคำตอบ, ตอบภาษาเดียวกับคำถาม, ปิดท้ายด้วยชื่อไฟล์อ้างอิง |
| **LLM** | Groq API, temperature 0.2, streaming |
| **Chatbot Interface** | `st.chat_message` + `st.chat_input`, จำประวัติใน `session_state`, Query Rewriting ทำให้ถามต่อเนื่องได้, แสดงเอกสารอ้างอิง (ไฟล์, หัวข้อ, คะแนน, เนื้อหา) ทุกคำตอบ |
| **Deploy** | `@st.cache_resource` โหลดโมเดลและสร้าง index ครั้งเดียว, API key อ่านจาก `st.secrets["GROQ_API_KEY"]` |

## 4. วิธีใช้งาน

### รันบนเครื่องตัวเอง
```bash
pip install -r requirements.txt
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # แล้วแก้ใส่ GROQ_API_KEY จริง
streamlit run app.py
```
ขอ API key ฟรีได้ที่ https://console.groq.com/keys

### Deploy บน Streamlit Community Cloud
1. Push โค้ดขึ้น GitHub (ตรวจว่า **ไม่มี** `.streamlit/secrets.toml` ใน repo: `git ls-files | grep secrets`)
2. ไปที่ https://share.streamlit.io → **Create app** → เลือก repo, branch `main`, main file `app.py`
3. **Advanced settings** → Python 3.12 → ช่อง **Secrets** ใส่ `GROQ_API_KEY = "gsk_..."`
4. กด Deploy (ครั้งแรกติดตั้งไลบรารีและดาวน์โหลดโมเดลประมาณ 3-5 นาที)
5. ทดสอบเปิด URL ในโหมด Incognito

### การใช้งานหน้าเว็บ
- พิมพ์คำถามในช่องแชตด้านล่าง หรือกด **ตัวอย่างคำถาม** ที่แถบด้านซ้าย
- กด **📚 เอกสารอ้างอิงที่ใช้ตอบ** ใต้คำตอบ เพื่อดูไฟล์ หัวข้อ และเนื้อหาที่ระบบค้นมา
- ถามต่อเนื่องได้ เช่น "ผสมปุ๋ย A-B ยังไง" → "แล้วต้องเปลี่ยนบ่อยแค่ไหน"
- ปรับ Top-K, เกณฑ์ความเกี่ยวข้อง, เลือกโมเดล LLM และล้างแชตได้ที่แถบด้านซ้าย

### ทดสอบด้วย test_questions.csv
```bash
python evaluate.py          # วัด retrieval hit-rate (ไม่ต้องใช้ API key)
GROQ_API_KEY=gsk_... python evaluate.py --llm   # ให้ LLM ตอบจริง + เช็คการปฏิเสธ
```

## 5. แหล่งที่มาของเอกสาร

เอกสารใน `data/` **เรียบเรียงขึ้นด้วยความช่วยเหลือของ AI (Claude)** จากความรู้ทั่วไปด้านการปลูกพืชไร้ดิน (อนุญาตตามโจทย์) โดยอ้างอิงแนวทางเนื้อหาจากแหล่งความรู้สาธารณะ เช่น
- กรมส่งเสริมการเกษตร / กรมวิชาการเกษตร — เอกสารเผยแพร่เรื่องการปลูกพืชไร้ดิน
- เอกสารเผยแพร่ของมหาวิทยาลัยด้านเกษตร เช่น มหาวิทยาลัยเกษตรศาสตร์
- University of Hawaii — งานของ Dr. B.A. Kratky เรื่องระบบน้ำนิ่งไม่หมุนเวียน
- Cornell CEA / University extension guides เรื่อง hydroponic lettuce

> ⚠️ ตัวเลข (เช่น ค่า EC, ราคาอุปกรณ์) เป็นค่าประมาณเพื่อการศึกษา ควรตรวจสอบกับฉลากผลิตภัณฑ์และแหล่งข้อมูลทางการก่อนใช้งานจริง

## 6. ตัวอย่าง Prompt ที่ใช้

### 6.1 System Prompt ของแชตบอต (ใน `rag_core.py`)
```text
คุณคือ "HydroBuddy" ผู้ช่วยตอบคำถามเรื่องการปลูกผักไฮโดรโปนิกส์ในบ้าน

กฎที่ต้องปฏิบัติอย่างเคร่งครัด:
1. ตอบโดยใช้ข้อมูลจาก <context> ที่ให้มาเท่านั้น ห้ามใช้ความรู้ภายนอก ห้ามเดา และห้ามแต่งตัวเลขเพิ่ม
2. ทุกประโยคที่ใช้ข้อมูลจากเอกสาร ให้ใส่เลขอ้างอิงในวงเล็บเหลี่ยมท้ายประโยค เช่น [1] หรือ [1][3]
3. ถ้า <context> ไม่มีข้อมูลที่ตอบคำถามได้ หรือคำถามไม่เกี่ยวกับการปลูกพืชไฮโดรโปนิกส์
   ให้ตอบขึ้นต้นด้วยคำว่า "ไม่พบข้อมูล" ตามด้วยคำอธิบายสั้น ๆ และห้ามตอบจากความรู้ทั่วไป
4. ถ้ามีข้อมูลเพียงบางส่วน ให้ตอบเฉพาะส่วนที่มีในเอกสาร แล้วบอกว่าส่วนที่เหลือไม่พบในเอกสาร
5. ตอบเป็นภาษาเดียวกับคำถาม
6. ตอบกระชับ เป็นมิตร ใช้หัวข้อย่อยหรือขั้นตอนเมื่อเหมาะสม
7. ปิดท้ายคำตอบด้วยบรรทัด "แหล่งอ้างอิง:" แล้วระบุชื่อไฟล์ที่ใช้ตอบจริง ถ้าไม่พบข้อมูลห้ามใส่บรรทัดนี้
8. ใช้เลขอ้างอิงรูปแบบ [1] เท่านั้น ห้ามใช้รูปแบบอื่น เช่น 【1】
```

### 6.2 User Prompt (สร้างทุกครั้งที่ถาม)
```text
<context>
[1] (ไฟล์: 03_nutrient_solution.md | หัวข้อ: สารละลายธาตุอาหารและปุ๋ย A-B > วิธีผสมปุ๋ย A-B ที่ถูกต้อง)
1. เติมน้ำสะอาดลงในถังพัก...
[2] ...
</context>

คำถาม: ผสมปุ๋ย A กับ B ยังไงให้ถูกต้อง?

ตอบตามกฎใน system prompt โดยใช้เฉพาะข้อมูลใน <context> และอ้างอิงด้วย [หมายเลข]
```

### 6.3 Query Rewriting Prompt
```text
จากประวัติการสนทนาและคำถามล่าสุด ให้เขียนคำถามล่าสุดใหม่เป็นคำถามที่สมบูรณ์ในตัวเอง
(แทนคำสรรพนามหรือคำที่อ้างถึงสิ่งที่พูดไปแล้วด้วยชื่อจริง) เพื่อนำไปค้นหาเอกสาร
ตอบเฉพาะคำถามที่เขียนใหม่ บรรทัดเดียว ห้ามตอบคำถาม ห้ามอธิบาย
```

### 6.4 ตัวอย่าง Prompt ที่ใช้สั่ง AI ช่วยพัฒนา
- *"ช่วยเขียนเอกสารความรู้ภาษาไทยเรื่องการปลูกผักไฮโดรโปนิกส์ 10 หัวข้อ แบ่งหัวข้อย่อยด้วย ## มีตัวเลขเฉพาะ เช่น ค่า pH, EC, จำนวนวัน เพื่อใช้เป็นคลังเอกสาร RAG"*
- *"เขียน Streamlit app ระบบ RAG ภาษาไทย ใช้ sentence-transformers + FAISS + Groq ให้แบ่ง chunk ตามหัวข้อ มี overlap, แสดงแหล่งอ้างอิงทุกคำตอบ, และตอบ 'ไม่พบข้อมูล' เมื่อไม่มีคำตอบ"*
- *"เพิ่ม hybrid search ด้วย BM25 + PyThaiNLP และรวมผลด้วย Reciprocal Rank Fusion"*
- *"สร้าง test_questions.csv 15 ข้อ พร้อมคำตอบที่ถูกต้อง โดยมีคำถามที่ไม่มีคำตอบในเอกสารอย่างน้อย 2 ข้อ"*

## 7. ข้อจำกัด
- ตอบได้เฉพาะเนื้อหาที่อยู่ในคลังเอกสาร (ตั้งใจออกแบบให้ปฏิเสธเมื่อไม่มีข้อมูล)
- เกณฑ์ cosine ของ e5 มักอยู่ในช่วงแคบ (~0.70-0.90) สามารถปรับเกณฑ์ที่ sidebar หรือดูค่าจาก `evaluate.py`
