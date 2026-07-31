import time
import streamlit as st
import rag

PDF_URL = "https://ozone.unep.org/sites/default/files/2023-03/cc0923en.pdf"

EXAMPLES = [
    "What is precooling?",
    "How much food is lost due to lack of refrigeration?",
    "What share of emissions comes from the cold chain?",
    "Why do developing countries lose more food?",
]

st.set_page_config(page_title="Cold Chain Assistant", page_icon="❄️")


@st.cache_resource
def setup():
    return rag.init()


with st.spinner("Loading knowledge base and models..."):
    sources, documents = setup()


def show_sources(results):
    with st.expander("Sources"):
        for i, score in results:
            st.caption(f"**{score:.3f}** — {sources[i]}")
            st.text(documents[i][:300] + "...")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

# --- Sidebar ---
with st.sidebar:
    st.header("❄️ Cold Chain Assistant")

    st.subheader("Why it matters")
    st.metric("Food lost to poor refrigeration", "526M tons/yr")
    st.metric("Cost to the global economy", "$936 billion/yr")
    st.metric("Food that gets refrigerated", "45%")
    st.caption("Figures from the source report below.")

    st.divider()

    st.subheader("Try asking")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q

    st.divider()

    st.subheader("Settings")
    top_k = st.slider("Chunks retrieved (top-k)", 1, 6, 3)
    st.caption(
        "How many text passages the assistant pulls from the report "
        "to answer. Fewer = focused but may miss context. "
        "More = broader but adds noise."
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(f"**{len(documents)}** chunks indexed")
    st.caption("**Source:** UNEP/FAO Sustainable Food Cold Chains (2022)")
    st.link_button("Open source PDF", PDF_URL, use_container_width=True)
    st.caption("**Embedding:** qwen3-embedding-0.6b")
    st.caption("**LLM:** phi-3.5-mini")
    st.caption("Optimized for English. Other languages are experimental.")
    st.caption(
        "Runs entirely on this machine. No cloud, no API keys, "
        "no outbound network calls."
    )

# --- Main ---
st.title("Cold Chain Assistant")
st.caption("Ask about cold chain, refrigeration, food loss.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])
        if msg.get("results"):
            show_sources(msg["results"])

question = st.chat_input("Ask a question...")
if st.session_state.pending:
    question = st.session_state.pending
    st.session_state.pending = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and answering..."):
            start = time.time()
            try:
                answer, results = rag.answer_query(question, top_k)
            except Exception as e:
                answer, results = f"Error: {e}", None
            elapsed = time.time() - start

        st.write(answer)
        meta = f"{elapsed:.1f}s · top-{top_k} · local inference"
        st.caption(meta)
        if results:
            show_sources(results)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "results": results, "meta": meta}
    )