from typing import TypedDict, Optional, List, Annotated
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import torch
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import JsonOutputParser
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_huggingface import HuggingFaceEmbeddings
from transformers import CLIPModel, CLIPProcessor

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import uuid


def init_collections(qdrant):
    logger.info("Initializing Qdrant collections")
    collections = qdrant.get_collections().collections
    existing = {c.name for c in collections}


    if "clinical_notes" not in existing:
        qdrant.create_collection(
            "clinical_notes",
            VectorParams(size=384, distance=Distance.COSINE)
        )

    if "lab_reports" not in existing:
        qdrant.create_collection(
            "lab_reports",
            VectorParams(size=384, distance=Distance.COSINE)
        )

    if "patient_profiles" not in existing:
        qdrant.create_collection(
            "patient_profiles",
            VectorParams(size=1, distance=Distance.COSINE)
        )



qdrant = QdrantClient(url="http://localhost:6333")
init_collections(qdrant)

text_embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


class MedState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    lab_image_path: Optional[str]
    clinical_text: Optional[str]
    user_query: Optional[str]
    thread_id: str
    name: Optional[str]
    age: Optional[str]
    parsed_lab: Optional[dict]
    clinical_note: Optional[dict]

    past_labs: Optional[List[dict]]
    past_notes: Optional[List[dict]]
    past_xrays: Optional[List[dict]]
    trends: Optional[List[dict]]
  
    final_summary: Optional[str]


def embed_text(text: str) -> List[float]:
    return text_embedder.embed_query(text)


def classify_importance(text: str) -> str:
    return llm.invoke(f"""
Classify the medical record below as:
- high
- medium
- low

Return ONLY one word.

Record:
{text}
""").content.strip().lower()


def embed_medical_image(path: str) -> List[float]:
    image = Image.open(path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return (features / features.norm(dim=-1, keepdim=True))[0].cpu().tolist()


def retrieve_history(state: MedState):
    logger.info("retrieve_history node")
    tid = state["thread_id"]

    text_vector = embed_text(state.get("clinical_text") or state.get("user_query") or "")

    def fetch_text(collection):
        return [
            p.payload
            for p in qdrant.query_points(
                collection,
                query=text_vector,
                with_payload=True,
                limit=5,
                query_filter=Filter(
                    must=[FieldCondition(key="thread_id", match=MatchValue(value=tid))]
                ),
            ).points
        ]

    past_notes = fetch_text("clinical_notes")
    past_labs = fetch_text("lab_reports")



    return {
        "past_notes": past_notes,
        "past_labs": past_labs,
        
    }

def save_patient_profile(thread_id: str, name: str, age: int):
    logger.info(f"Saving patient profile | {name}, {age}")
    qdrant.upsert(
        "patient_profiles",
        [
            PointStruct(
                id=thread_id,
                vector=[0.0],
                payload={
                    "thread_id": thread_id,
                    "name": name,
                    "age": age,
                    "created_at": datetime.now().isoformat(),
                },
            )
        ],
    )


def load_patient_profile(thread_id: str):
    points, _ = qdrant.scroll(
        "patient_profiles",
        limit=1,
        scroll_filter=Filter(
            must=[FieldCondition(
                key="thread_id",
                match=MatchValue(value=thread_id)
            )]
        ),
    )
    return points[0].payload if points else None

def input_node(state: MedState):
    logger.info("input node")
    profile = load_patient_profile(state["thread_id"])
    if profile:
        return {
            "name": profile["name"],
            "age": profile["age"],
        }
    return {}

def user_input(state: MedState):
    logger.info("user_input node")
    return {
        "thread_id": state["thread_id"],
        "name": state.get("name"),
        "age": state.get("age"),
    }


def user_query_node(state: MedState):
    logger.info("User query node")
    return {}


def parse_lab_image(state: MedState):
    logger.info("parse_lab_image node")

    if not state.get("lab_image_path"):
        logger.info("No lab image provided — skipping lab parsing")
        return {}

    raw = pytesseract.image_to_string(
        Image.open(state["lab_image_path"])
    ).strip()

    if not raw:
        logger.warning("OCR returned empty text — skipping lab parsing")
        return {}

    prompt = f"""
Extract lab values from the text below.

Rules:
- Return ONLY valid JSON
- Use test names as keys and numeric values as numbers
- If NO lab values are present, return an EMPTY JSON object: {{}}

Text:
{raw}
"""

    llm_output = llm.invoke(prompt).content.strip()

    try:
        parsed = JsonOutputParser().parse(llm_output)
    except Exception as e:
        logger.error(f"Lab JSON parsing failed — skipping lab. Output was: {llm_output}")
        return {}

    if not parsed:
        logger.info("No lab values detected in image")
        return {}

    logger.info("Lab values successfully parsed")

    return {
        "parsed_lab": {
            "date": datetime.now().isoformat(),
            "values": parsed,
        }
    }



def parse_clinical_note(state: MedState):
    logger.info("Parsing clinical note")
    if not state.get("clinical_text"):
        return {}
    return {"clinical_note": {"text": state["clinical_text"], "date": datetime.now().isoformat()}}


def store_memory(state: MedState):
    logger.info("store_memory node")
    tid = state["thread_id"]

    if state.get("parsed_lab"):
        qdrant.upsert(
            "lab_reports",
            [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embed_text(str(state["parsed_lab"]["values"])),
                    payload={
                        "thread_id": tid,
                        "importance": classify_importance(str(state["parsed_lab"])),
                        **state["parsed_lab"],
                    },
                )
            ],
        )

    if state.get("clinical_note"):
        qdrant.upsert(
            "clinical_notes",
            [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embed_text(state["clinical_note"]["text"]),
                    payload={
                        "thread_id": tid,
                        "importance": classify_importance(state["clinical_note"]["text"]),
                        **state["clinical_note"],
                    },
                )
            ],
        )


    return {}



def retrieve_history(state: MedState):
    logger.info("retrieve_history node")
    tid = state["thread_id"]

    text_vector = embed_text(state.get("clinical_text") or state.get("user_query") or "")

    def fetch_text(collection):
        return [
            p.payload
            for p in qdrant.query_points(
                collection,
                query=text_vector,
                with_payload=True,
                limit=5,
                query_filter=Filter(
                    must=[FieldCondition(key="thread_id", match=MatchValue(value=tid))]
                ),
            ).points
        ]

    past_notes = fetch_text("clinical_notes")
    past_labs = fetch_text("lab_reports")



    return {
        "past_notes": past_notes,
        "past_labs": past_labs,
    }

def trend_node(state: MedState):
    logger.info("trend_node")
    combined = (state.get("past_labs") or []) + (state.get("past_notes") or [])
    return {"trends": sorted(combined, key=lambda x: x["date"])}


def summary_node(state: MedState):
    logger.info("Entered summary_node")

    has_past_data = bool(
        (state.get("past_labs") or [])
        or (state.get("past_notes") or [])
        or (state.get("trends") or [])
    )

    show_labs = bool(state.get("lab_image_path"))


    logger.info(f"Summary decision | show_labs={show_labs}")

    labs_section = (
        f"\nLabs:\n{state.get('past_labs')}\n" if show_labs else ""
    )

    if has_past_data:
        prompt = f"""
You are a healthcare memory assistant.

Your task:
1. Summarize the patient's historical medical information.
2. Identify observable trends over time.
3. Provide high-level, non-diagnostic feedback.

Rules:
- Do NOT diagnose.
- Do NOT recommend treatment.
- Use cautious, observational language.
-Do not show patient id in the final summary.Instead display the name of the patient.

Patient History:
{labs_section}
Clinical Notes:
{state.get("past_notes")}

Trends:
{state.get("trends")}

User Question (optional):
{state.get("user_query")}

Make sure to remember the full context of the conversation from the message history list: {state['messages']}
"""
    else:
        prompt = f"""
You are a healthcare memory assistant.
Patient Details:
Name: {state['name']}
Age: {state['age']}
Give the summary stating the {state['name']} of the person.
Your task:
1. Summarize the current clinical note.
2. Identify observable trends.
3. Provide high-level, non-diagnostic feedback.

Rules:
- Do NOT diagnose.
- Do NOT recommend treatment.
- Use cautious language.

Clinical Note:
{state.get("clinical_text")}

If the user has asked a question, answer it using only the above note. If no question is present, skip the Q&A section.
User Question (optional):
{state.get("user_query")}
Make sure to remember the full context of the conversation from the message history list.Conversation history:
{chr(10).join([f"{m.type.upper()}: {m.content}" for m in state['messages']])}

"""
    logger.info("Invoking LLM for summary")
    response = llm.invoke(prompt)

    logger.info("Exiting summary_node")

    return {
        "final_summary": response.content,
        "messages": [AIMessage(content=response.content)]
    }


def forgetting_node(_: MedState):
    logger.info("Forgetting node")
    return {}

def build_app(checkpointer):

    graph = StateGraph(MedState)

    graph.add_node("input", input_node)
    graph.add_node("parse_lab", parse_lab_image)
    graph.add_node("parse_note", parse_clinical_note)
    graph.add_node("store", store_memory)
    graph.add_node("retrieve", retrieve_history)
    graph.add_node("trend", trend_node)
    graph.add_node("summary", summary_node)
    graph.add_node("forget", forgetting_node)

    graph.add_edge(START, "input")
    graph.add_edge("input", "parse_lab")
    graph.add_edge("parse_lab", "parse_note")
    graph.add_edge("parse_note", "store")
    graph.add_edge("store", "retrieve")
    graph.add_edge("retrieve", "trend")
    graph.add_edge("trend", "summary")
    graph.add_edge("summary", "forget")
    graph.add_edge("forget", END)

    app=graph.compile(checkpointer=checkpointer)
    return app

def list_threads_with_names():
    """
    Returns:
    [
      {"thread_id": "...", "name": "...", "age": ...},
      ...
    ]
    """

    logger.info("Listing threads with patient names")

    points, _ = qdrant.scroll(
        collection_name="clinical_notes",
        with_payload=True,
        limit=200,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="type",
                    match=MatchValue(value="patient_meta")
                )
            ]
        ),
    )

    threads = []
    seen_threads = set()

    for p in points:
        payload = p.payload
        tid = payload.get("thread_id")

        if not tid or tid in seen_threads:
            continue

        threads.append({
            "thread_id": tid,
            "name": payload.get("name", "Unknown"),
            "age": payload.get("age"),
        })

        seen_threads.add(tid)

    return threads
