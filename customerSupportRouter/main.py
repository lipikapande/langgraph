from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.panel import Panel

console = Console()

llm=ChatOllama(model="qwen2.5:3b",temperature=0)

class SupportState(TypedDict):
    query:str
    category:str
    answer:str

def classify(state: SupportState)-> SupportState:
    """
    Uses Qwen 2.5 to classify the query into a category.
    """

    prompt = f"""You are a customer support classifier.
    Classify the following customer query into exactly ONE category.
    Query: {state['query']}
    Rules:
    - Reply with ONLY one word: billing, technical, or general
    - Do NOT add explanations, punctuation, or extra words
    - billing = payment, invoice, refund, subscription, charges
    - technical = error, bug, crash, not working, installation, login issues
    - general = hours, location, product info, policies, other
    """

    result=llm.invoke(prompt)

    raw=result.content.strip().lower()

    valid_categories = {"billing", "technical", "general"}
    state["category"]=raw if raw in valid_categories else "general"
    console.print(f"[bold cyan]Classified as: {state['category'].upper()}[/bold cyan]")
    return state

def billing_agent(state: SupportState) -> SupportState:
    state["answer"] = (
    "BILLING SUPPORT\n"
    "Your query has been received by our Billing Team.\n"
    "Typical issues handled: payment failures, invoice disputes, refunds.\n"
    "Expected resolution: 24-48 hours.\n"
    "Reference: " + state["query"][:30] + "..."
    )
    return state

def technical_agent(state: SupportState) -> SupportState:
    state["answer"] = (
    "TECHNICAL SUPPORT\n"
    "Your query has been assigned to our Technical Team.\n"
    "Please provide: device type, OS version, error message screenshot.\n"
    "Expected resolution: 2-4 hours.\n"
    "Reference: " + state["query"][:30] + "..."
    )
    return state

def general_agent(state: SupportState) -> SupportState:
    state["answer"] = (
    "GENERAL SUPPORT\n"
    "Thank you for contacting us.\n"
    "Our support team will get back to you shortly.\n"
    "Operating hours: Mon-Fri, 9 AM - 6 PM IST.\n"
    "Reference: " + state["query"][:30] + "..."
    )
    return state

def router(state: SupportState) -> SupportState:
    """
    Returns the category string that maps to the next node.
    LangGraph uses this return value to pick which branch to follow.
    """
    return state["category"]

builder = StateGraph(SupportState)

builder.add_node("classify", classify)
builder.add_node("billing", billing_agent)
builder.add_node("technical", technical_agent)
builder.add_node("general", general_agent)

builder.add_conditional_edges("classify",router,{
    "billing":"billing",
    "technical":"technical",
    "general":"general",
})

builder.set_entry_point("classify")

builder.set_finish_point("billing")
builder.set_finish_point("technical")
builder.set_finish_point("general")

graph=builder.compile()

test_queries = [
 "My payment failed yesterday and I was charged twice",
 "The app crashes every time I try to login from my iPhone",
 "What are your office hours?",
 "I need a refund for my last invoice",
 "How do I reset my password? The reset email never arrives",
 "Do you offer student discounts?",
]

if __name__=="__main__":
    console.print("\n[bold magenta]=== AI Customer Support Router ===[/bold magenta]\n")

    for query in test_queries:
        console.print(f"\n[bold white]Customer:[/bold white] {query}")

        result=graph.invoke({
            "query":query,
            "category":"",
            "answer":""
        })

        console.print(Panel(
          result["answer"],
          title=f"[bold green]Response ({result['category'].upper()})[/bold green]",
          border_style="green"
        ))

    console.print("\n[bold yellow]All queries routed successfully.[/bold yellow]")   
