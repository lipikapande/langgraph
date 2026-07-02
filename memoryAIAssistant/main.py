from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import json

console = Console()

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.3
)

# --------------------------------------------------
# SESSION MEMORY
# --------------------------------------------------

session_memory = {
    "user_name": None,
    "user_interests": [],
    "user_preferences": {},
    "extracted_facts": [],
    "conversation_history": []
}

# --------------------------------------------------
# STATE
# --------------------------------------------------

class MemoryState(TypedDict):
    user_input: str
    user_name: Optional[str]
    user_interests: List[str]
    user_preferences: dict
    extracted_facts: List[str]
    answer: str

# --------------------------------------------------
# NODE 1: EXTRACT MEMORY
# --------------------------------------------------

def extract_memory(state: MemoryState) -> MemoryState:

    prompt = f"""
    You are a memory extraction system.

    Read this message and extract personal facts.

    Message:
    "{state['user_input']}"

    Respond with ONLY valid JSON.

    {{
      "name": null,
      "interests": [],
      "preferences": {{}},
      "other_facts": []
    }}
    """
    result = llm.invoke(prompt)

    raw = (
        result.content
        .strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        extracted = json.loads(raw)

        if extracted.get("name"):
            session_memory["user_name"] = extracted["name"]

        for interest in extracted.get("interests", []):
            if interest not in session_memory["user_interests"]:
                session_memory["user_interests"].append(interest)

        session_memory["user_preferences"].update(
            extracted.get("preferences", {})
        )

        for fact in extracted.get("other_facts", []):
            if fact not in session_memory["extracted_facts"]:
                session_memory["extracted_facts"].append(fact)

    except json.JSONDecodeError:
        pass

    state["user_name"] = session_memory["user_name"]
    state["user_interests"] = list(session_memory["user_interests"])
    state["user_preferences"] = dict(session_memory["user_preferences"])
    state["extracted_facts"] = list(session_memory["extracted_facts"])

    return state

def generate_answer(state:MemoryState)->MemoryState:
    memory_lines=[]

    if state["user_name"]:
        memory_lines.append(f"Name: {state['user_name']}")
    
    if state["user_interests"]:
        memory_lines.append(
            "Interests: " + ", ".join(state["user_interests"])
        )

    for key, value in state["user_preferences"].items():
        memory_lines.append(f"{key}: {value}")

    for fact in state["extracted_facts"]:
        memory_lines.append(f"Fact: {fact}")

    memory_block = (
        "\n".join(memory_lines)
        if memory_lines
        else "Nothing known yet."
    )

    history_text = ""
    for turn in session_memory["conversation_history"][-6:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_text += f"{role}: {turn['text']}\n"

    prompt = f"""
You are a friendly personal assistant with a good memory.

--- WHAT YOU KNOW ABOUT THIS PERSON ---
{memory_block}

--- RECENT CONVERSATION ---
{history_text}

--- CURRENT MESSAGE ---
User: {state['user_input']}

Instructions:
- Use their name naturally if you know it
- Reference past info only when genuinely helpful
- If they ask what you remember, list everything
- Be concise and warm

Assistant:
"""
    result = llm.invoke(prompt)

    state["answer"] = result.content.strip()

    session_memory["conversation_history"].append({
        "role": "user",
        "text": state["user_input"]
    })

    session_memory["conversation_history"].append({
        "role": "assistant",
        "text": state["answer"]
    })

    return state

# --------------------------------------------------
# BUILD GRAPH
# --------------------------------------------------

builder = StateGraph(MemoryState)

builder.add_node("extract_memory", extract_memory)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(
    "extract_memory",
    "generate_answer"
)

builder.set_entry_point("extract_memory")
builder.set_finish_point("generate_answer")

graph = builder.compile()

def show_memory():

    table = Table(
        title="Current Memory State",
        style="bold cyan"
    )

    table.add_column("Field", width=20)
    table.add_column("Value", width=50)

    table.add_row(
        "Name",
        str(session_memory["user_name"] or "-")
    )

    table.add_row(
        "Interests",
        ", ".join(session_memory["user_interests"]) or "-"
    )

    table.add_row(
        "Preferences",
        str(session_memory["user_preferences"]) or "-"
    )

    table.add_row(
        "Other Facts",
        str(session_memory["extracted_facts"]) or "-"
    )

    table.add_row(
        "Turns",
        str(len(session_memory["conversation_history"]) // 2)
    )

    console.print(table)

# --------------------------------------------------
# CHAT LOOP
# --------------------------------------------------

if __name__ == "__main__":

    console.print(
        "\n[bold magenta]=== Memory Enabled AI Assistant ===[/bold magenta]"
    )

    console.print(
        "[dim]Commands: memory | clear | quit[/dim]\n"
    )

    while True:

        user_input = console.input(
            "[bold cyan]You: [/bold cyan]"
        ).strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            console.print("[yellow]Goodbye![/yellow]")
            break

        if user_input.lower() == "memory":
            show_memory()
            continue

        if user_input.lower() == "clear":
            session_memory.update({
                "user_name": None,
                "user_interests": [],
                "user_preferences": {},
                "extracted_facts": [],
                "conversation_history": []
            })

            console.print(
                "[yellow]Memory cleared.[/yellow]"
            )
            continue

        result = graph.invoke({
            "user_input": user_input,
            "user_name": None,
            "user_interests": [],
            "user_preferences": {},
            "extracted_facts": [],
            "answer": ""
        })

        console.print(
            Panel(
                result["answer"],
                title="[bold green]Assistant[/bold green]",
                border_style="green"
            )
        )