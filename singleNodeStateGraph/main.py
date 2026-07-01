from typing import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name: str
    greeting: str

def generate_greeting(state: AgentState) -> AgentState:
    """
    Node: Greeting Generator
    Reads 'name' from the state and generates a greeting message.
    """

    name=state['name']
    greeting = f"Hello, {name}! Welcome to the LangGraph training."
    state["greeting"] = greeting
    return state

builder=StateGraph(AgentState)

builder.add_node("greet",generate_greeting)

builder.set_entry_point("greet")

builder.set_finish_point("greet")

graph=builder.compile()

if __name__ == "__main__":
    result=graph.invoke({"name": "Alice"})
    print(result["greeting"])  # Output: Hello, Alice! Welcome to the LangGraph training.

    for name in ["Bob", "Charlie", "Diana"]:
        result=graph.invoke({"name": name})
        print(result["greeting"])  # Output: Hello, Bob! Welcome to the LangGraph training. etc.