
# Your medical assistant 
This study introduces the design and implementation of a Healthcare Memory
Assistant, an intelligent conversational system developed with LangGraph,
Qdrant, and Large Language Models (LLMs). The system ingests multimodal
medical data, stores it as long-term memory, retrieves relevant historical context,
and generates safe, non-diagnostic summaries tailored to user queries.
The assistant is architected with memory persistence, session continuity,
multi-threaded conversations, and explainable summarization as first-class
design goals.<br><br>
## High-Level System Architecture<br><br>

<img width="126" height="928" alt="langgraph_workflow" src="https://github.com/user-attachments/assets/d136b73c-dda5-4546-ac7a-943c7bf44b65" />

The system has a modular, layered structure:
3.1 Core Components<br><br>
● Frontend (Streamlit)<br><br>
● Patient input (user query)<br><br>
● Chat management (New Chat / Resume Chat)<br><br>
● File uploads<br><br>
● Summary visualization<br><br>
● Backend (LangGraph Workflow)<br><br>
● Node-based execution<br><br>
● Stateful memory management<br><br>
● Decision routing<br><br>
● Decision<br><br>
● Vector Memory Store (Qdrant)<br><br>
● Persistent semantic memory (PostgreMemorySaver)<br><br>
● Thread-level isolation<br><br>
● Multimodal Vector Storage<br><br>
● Checkpoint Store (PostgreSQL)<br><br>
● Execution State<br><br>
● Conversation resumption<br><br>
● LLMs & Models<br>
[GPT-based reasoning and summarization]
[SentenceTransformers for text embeddings]
<br><br>
## Qdrant: The Core Memory Engine
Qdrant is the heart of the intelligence in the system. The choice made is valid due
to the following core capabilities:<br><br>
● Native Vector Similarity Search<br><br>
● High-performance filtering<br><br>
● Payload-based metadata storage<br><br>
● Persistent, disk-backed collections<br><br>
● Multimodal compatibility with user input
<br><br>
Frontend Architecture and UX Design<br><br>
 <img width="1910" height="866" alt="Screenshot 2026-01-23 035300" src="https://github.com/user-attachments/assets/64f5ef98-dc2b-4c1f-8e63-38005e8cebe2" />
 <br><br>
## Conclusion 

The current project outlines the architecture for a production-level
memory-augmented healthcare assistant. By combining the stateful workflows of
LangGraph with the vector memory of Qdrant, the current system moves beyond
the capabilities of a typical chatbot.The separation of concerns in architecture,
strong memory guarantees, multimodal capabilities, and ethical foundations
together endow technical rigor and social responsibility on the system.
Most importantly, the research proves that modern AI systems should be imagined
not only as isolated models but also as integrated systems that have memory,
identity, continuity, and governance.<br><br>
This work provides a strong foundation for future research and operational use in
areas such as clinical decision support, patient education, and management of
healthcare data.<br><br>
The current project outlines the architecture for a production-level
memory-augmented healthcare assistant. By combining the stateful workflows of
LangGraph with the vector memory of Qdrant, the current system moves beyond
the capabilities of a typical chatbot.The separation of concerns in architecture,
strong memory guarantees, multimodal capabilities, and ethical foundations
together endow technical rigor and social responsibility on the system.
Most importantly, the research proves that modern AI systems should be imagined
not only as isolated models but also as integrated systems that have memory,
identity, continuity, and governance.<br><br>
This work provides a strong foundation for future research and operational use in
areas such as clinical decision support, patient education, and management of
healthcare data.
