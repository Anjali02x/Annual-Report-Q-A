import io, time, requests
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Annual Report Q&A", page_icon="📊")
st.title("📊 Annual Report Q&A")
st.caption("Upload a company annual report (PDF) and ask questions. "
           "For education only, not financial advice.")

H = {"Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}"}
MODEL = "openai/gpt-oss-20b"

def ask(prompt):
    for _ in range(4):
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=H,
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}]}, timeout=90)
        if r.ok:
            return r.json()["choices"][0]["message"]["content"].strip()
        if r.status_code == 429:
            time.sleep(20)
            continue
        return f"Error {r.status_code}: {r.text[:150]}"
    return "Rate limited right now, please try again in a minute."

@st.cache_data(show_spinner="Reading the report...")
def load_pdf(data: bytes):
    chunks, pages = [], []
    for i, page in enumerate(PdfReader(io.BytesIO(data)).pages, start=1):
        text = (page.extract_text() or "").strip()
        for j in range(0, len(text), 1200):
            piece = text[j:j + 1200]
            if len(piece) > 100:
                chunks.append(piece)
                pages.append(i)
    if not chunks:
        return [], [], None, None
    vec = TfidfVectorizer(stop_words="english")
    return chunks, pages, vec, vec.fit_transform(chunks)

def answer(question, chunks, pages, vec, matrix, k=4):
    scores = cosine_similarity(vec.transform([question]), matrix)[0]
    top = scores.argsort()[::-1][:k]
    context = "\n\n".join(f"[Page {pages[i]}]\n{chunks[i]}" for i in top)
    return ask("Answer the question using ONLY the report excerpts below. "
               "If the answer isn't there, say you couldn't find it. "
               "Mention the page numbers you used.\n\n"
               f"Excerpts:\n{context}\n\nQuestion: {question}")

file = st.file_uploader("Upload an annual report (PDF)", type="pdf")

if file:
    chunks, pages, vec, matrix = load_pdf(file.getvalue())
    if not chunks:
        st.error("Couldn't read text from this PDF. It may be a scanned document.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    if q := st.chat_input("Ask a question about the report"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                a = answer(q, chunks, pages, vec, matrix)
            st.write(a)
        st.session_state.messages.append({"role": "assistant", "content": a})
else:
    st.info("Upload a PDF to start.")
