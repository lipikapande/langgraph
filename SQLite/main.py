from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from rich.console import Console
from rich.panel import Panel
import sqlite3
import os

console=Console()
llm=ChatOllama(model="qwen2.5:3b",temperature=0.3);

DB_PATH="memory.db";

class ChatState(TypedDict):
  messages: Annotated[list,add_messages] 
  # annotated makes sure the old messages do not get replaced, its only added

def chat_node(state: ChatState) -> ChatState:
  """
  Receives the full message history, generates a response.
  The checkpointer saves the updated state to SQLite automatically.
  """
  system = (
  "You are a helpful, friendly AI assistant with perfect memory. "
  "You remember everything the user has told you in this conversation. "
  "Refer to past information naturally when relevant."
  )

  history_text="";
  for msg in state["messages"][:-1]: #excluding current/latest message
    if hasattr(msg,"content"):
      role="User" if msg.__class__.__name__=="HumanMessage" else "Assistant"
      history_text+=role+ ": "+msg.content+"\n"
  
  current_message=state["messages"][-1].content

  prompt=(
    system+"\n\n"
    "Conversation so far:\n"+history_text+"\n"
    "User: "+current_message+"\n\n"
    "Assistant: "
  )

  result=llm.invoke(prompt)

  return {"messages":[AIMessage(content=result.content.strip())]}

def build_graph_with_memory(db_path:str):
  """
  Creates the graph and attaches a SQLite checkpointer.
  The checkpointer intercepts every state update and saves it to disk.
  """

  builder=StateGraph(ChatState)
  builder.add_node("chat",chat_node)
  builder.set_entry_point("chat")
  builder.set_finish_point("chat")

  conn=sqlite3.connect(db_path,check_same_thread=False)
  memory=SqliteSaver(conn)
  graph=builder.compile(checkpointer=memory)

  return graph

def show_history(db_path: str, thread_id: str):
  """
  Directly reads from SQLite to show stored conversation history.
  """

  if not os.path.exists(db_path):
    console.print("[yellow]No database found yet.[/yellow]")
    return
  
  try:
    conn=sqlite3.connect(db_path)
    cursor=conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables=[row[0] for row in cursor.fetchall()]
    console.print(f"[dim]Database tables: {tables}[/dim]")
    conn.close()
  except Exception as e:
    console.print(f"[red]DB read error: {e}[/red]")

if __name__=="__main__":
  console.print("\n[bold magenta]=== Persistent Memory AI(SQLite)===[/bold magenta]")
  console.print(f"[dim]Database: {os.path.abspath(DB_PATH)}[/dim]")
  console.print("\n[cyan]Enter a Thread ID (press Enter for default 'user_001'):[/cyan]")
  thread_id_input = console.input("[bold cyan]Thread ID: [/bold cyan]").strip()
  thread_id = thread_id_input if thread_id_input else "user_001"

  console.print(f"\n[green]Using thread: [bold]{thread_id}[/bold][/green]")
  console.print("[dim]Type 'quit' to exit. Your conversation is saved automatically.[/dim]\n")

  graph=build_graph_with_memory(DB_PATH)

  config={"configurable": {"thread_id":thread_id}}

  while True:
    user_input=console.input("[bold cyan]You: [/bold cyan]").strip()
    if not user_input:
      continue
    if user_input.lower()=="quit":
      console.print(f"[yellow]Conversation saved to {DB_PATH}. See you next time![/yellow]")
      break
    
    result=graph.invoke({"messages":[HumanMessage(content=user_input)]},config=config)

    last_message = result["messages"][-1]
    console.print(Panel(
    last_message.content,
    title=f"[bold green]Assistant (thread: {thread_id})[/bold green]",
    border_style="green"
    ))
  console.print(f"\n[dim]Memory file location: {os.path.abspath(DB_PATH)}[/dim]")