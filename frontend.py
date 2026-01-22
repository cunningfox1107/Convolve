import streamlit as st
import uuid
import logging
from backend import build_app, list_threads_with_names
from langgraph.checkpoint.postgres import PostgresSaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("frontend")

st.set_page_config(
    page_title="Healthcare Memory Assistant",
    layout="wide"
)


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.name = ""
    st.session_state.age = None
    st.session_state.chat_resumed = False
    st.session_state.patient_saved = False 



st.sidebar.title("🧑 Patient Information")

name_input = st.sidebar.text_input(
    "Name",
    value=st.session_state.name
)

age_input = st.sidebar.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=st.session_state.age if st.session_state.age is not None else 0
)

if st.sidebar.button("💾 Save Patient Info"):
    st.session_state.name = name_input
    st.session_state.age = age_input
    st.session_state.patient_saved = True 
    logger.info(f"Saved patient info | {name_input}, {age_input}")
    st.sidebar.success("Patient details saved successfully")


if not st.session_state.patient_saved:
    st.sidebar.warning("⚠️ Please enter and save Name & Age to enable submission.")
else:
    st.sidebar.success("✅ Patient information saved")



st.sidebar.markdown("---")

if st.sidebar.button("➕ New Chat"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.name = ""
    st.session_state.age = None
    st.session_state.messages = []
    st.session_state.patient_saved = False

    st.session_state.clinical_note = ""
    st.session_state.user_question = ""
    st.session_state.lab_file = None
    st.session_state.xray_file = None

    st.success("New chat started")


st.sidebar.markdown("---")
st.sidebar.markdown("### 🔁 Resume Chat")

threads = list_threads_with_names()

if not threads:
    st.sidebar.info("No previous chats available")
else:
    for t in threads:
        label = f"👤 {t['name']} ({t['age']})"
        if st.sidebar.button(label, key=f"resume-{t['thread_id']}"):
            st.session_state.thread_id = t["thread_id"]
            st.session_state.name = t["name"]
            st.session_state.age = t["age"]
            st.session_state.patient_saved = True
            st.success("Chat resumed")


st.title("🩺 Healthcare Memory Assistant")
st.markdown(
    "This assistant helps you store, retrieve, and summarize clinical notes, lab reports, and X-ray images using long-term memory."
)

st.subheader("📤 Upload Medical Data")

lab_file = st.file_uploader(
    "Upload Lab Report/ X-ray Report (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)


clinical_note = st.text_area(
    "📝 Enter Clinical Note",
    placeholder="e.g. mild fever since morning..."
)

user_question = st.text_input(
    "❓ Optional Question",
    placeholder="e.g. Is this something serious?"
)


submit_disabled = not st.session_state.patient_saved

if st.button("🚀 Submit", disabled=submit_disabled):

    logger.info("Submitting user input")

    with PostgresSaver.from_conn_string(
        "postgresql://postgres:Cunningfox%401107@localhost:5432/langgraph"
    ) as checkpointer:

        app = build_app(checkpointer)

        result = app.invoke(
            {
                "thread_id": st.session_state.thread_id,
                "name": st.session_state.name,
                "age": st.session_state.age,
                "lab_image_path": lab_file,
                "clinical_text": clinical_note,
                "user_query": user_question,
                "messages": st.session_state.messages,
            },
            config={"configurable": {"thread_id": st.session_state.thread_id}}
        )

    st.session_state.messages.append(result["final_summary"])

    st.markdown("### 🧾 Summary")
    st.write(result["final_summary"])
