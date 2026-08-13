import requests
import streamlit as st


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="NeuraDocs | Intelligent Document AI",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# API CONFIG
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 20px;
        opacity: 0.7;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🧠 NeuraDocs</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent Document AI powered by RAG</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload multiple documents and interact with them using AI."
)

st.divider()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📚 NeuraDocs")

    st.write(
        "Your AI-powered document assistant."
    )

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    st.subheader("⚙️ System")

    # Check API status

    try:

        health_response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )

        if health_response.status_code == 200:

            st.success("🟢 API Connected")

        else:

            st.error("🔴 API Error")

    except requests.exceptions.RequestException:

        st.error("🔴 API Offline")

    st.write("**RAG Engine:** Active")

    st.write("**Vector Database:** FAISS")

    st.write("**Embeddings:** MiniLM")

    st.write("**LLM:** Llama 3.2")

    st.write("**Backend:** FastAPI")

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# --------------------------------------------------
# NO DOCUMENT
# --------------------------------------------------

if not uploaded_files:

    st.info(
        "👈 Upload one or more PDF documents from the sidebar."
    )

    st.markdown(
        """
        ### 🚀 What NeuraDocs can do

        - 📄 Upload multiple PDF documents
        - 🔎 Search across documents
        - 🧠 Generate contextual answers
        - 📚 Show source pages
        - 💬 Maintain conversation history
        - 🔗 Compare information across documents
        """)


# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

else:

    current_names = [
        file.name
        for file in uploaded_files
    ]

    current_names.sort()


    # Only upload when files change

    if current_names != st.session_state.uploaded_names:

        with st.spinner(
            "🧠 Uploading and indexing documents..."
        ):

            try:

                files_payload = []

                for file in uploaded_files:

                    files_payload.append(
                        (
                            "files",
                            (
                                file.name,
                                file.getvalue(),
                                "application/pdf"
                            )
                        )
                    )


                response = requests.post(
                    f"{API_URL}/upload",
                    files=files_payload,
                    timeout=300
                )


                if response.status_code == 200:

                    result = response.json()

                    st.session_state.uploaded_names = (
                        current_names
                    )

                    st.session_state.chat_history = []

                    st.success(
                        f"✅ {len(current_names)} document(s) indexed successfully!"
                    )

                else:

                    st.error(
                        f"Upload failed: {response.text}"
                    )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"Could not connect to FastAPI: {e}"
                )


    # --------------------------------------------------
    # DOCUMENT LIST
    # --------------------------------------------------

    st.subheader("📚 Uploaded Documents")

    for name in current_names:

        st.write(
            f"📄 {name}"
        )

    st.divider()


    # --------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------

    for message in st.session_state.chat_history:

        if message["role"] == "user":

            with st.chat_message("user"):

                st.write(
                    message["content"]
                )

        else:

            with st.chat_message("assistant"):

                st.write(
                    message["content"]
                )

                if message.get("sources"):

                    with st.expander(
                        "📚 View Sources"
                    ):

                        for source in message["sources"]:

                            st.markdown(
                                f"**📄 {source.get('file', 'Unknown')} — Page {source.get('page', 'Unknown')}**"
                            )

                            st.write(
                                source.get(
                                    "content",
                                    ""
                                )
                            )


    # --------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------

    question = st.chat_input(
        "Ask something about your documents..."
    )


    if question:

        with st.chat_message("user"):

            st.write(question)


        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )


        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching documents..."
            ):

                try:

                    response = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "question": question
                        },
                        timeout=300
                    )


                    if response.status_code == 200:

                        result = response.json()

                        answer = result.get(
                            "answer",
                            "No answer returned."
                        )

                        sources = result.get(
                            "sources",
                            []
                        )

                        st.write(answer)


                        # --------------------------
                        # SOURCES
                        # --------------------------

                        with st.expander(
                            "📚 View Sources"
                        ):

                            for source in sources:

                                st.markdown(
                                    f"**📄 {source.get('file', 'Unknown')} — Page {source.get('page', 'Unknown')}**"
                                )

                                st.write(
                                    source.get(
                                        "content",
                                        ""
                                    )
                                )


                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            }
                        )


                    else:

                        st.error(
                            f"API Error: {response.text}"
                        )


                except requests.exceptions.RequestException as e:

                    st.error(
                        f"Could not connect to FastAPI: {e}"
                    )