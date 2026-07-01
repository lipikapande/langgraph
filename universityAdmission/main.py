from typing import TypedDict
from langgraph.graph import StateGraph
import random
import datetime
from rich.console import Console
from rich.table import Table

console = Console()

class StudentState(TypedDict):
    name: str
    age: int
    marks: float
    student_id: str
    department: str
    scholarship:str
    message:str

def create_id(state: StudentState) -> StudentState:
    """
    Node 1: Create a unique student ID.
    Pattern:VIT-YEAR-RANDOM_NUMBER
    """
  
    year = datetime.datetime.now().year
    random_number = random.randint(1000, 9999)
    student_id = f"VIT-{year}-{random_number}"
    state["student_id"] = student_id
    console.print(f"[cyan]-> ID Generated: {state['student_id']}[/cyan]")
    return state

def assign_department(state: StudentState) -> StudentState:
    """
    Node 2: Assign department based on entrance marks.
    Marks 90+ -> AI & Data Science
    Marks 75-89 -> Computer Science Engineering
    Marks 60-74 -> Electronics & Communication
    Marks below -> Mechanical Engineering
    """

    marks=state["marks"]

    if marks >= 90:
        state["department"] = "AI & Data Science"
    elif marks >= 75:
        state["department"] = "Computer Science Engineering"
    elif marks >= 60:
        state["department"] = "Electronics & Communication"
    else:
        state["department"] = "Mechanical Engineering"

    console.print(f"[green]-> Department Assigned: {state['department']}[/green]")
    return state

def check_scholarship(state: StudentState) -> StudentState:
    """
    Node 3: Determine scholarship eligibility.
    Marks 95+ -> Full Scholarship (100%)
    Marks 85+ -> Partial Scholarship (50%)
    Marks 75+ -> Merit Award (25%)
    Below 75 -> No Scholarship
    """

    marks=state["marks"]

    if marks >= 95:
        state["scholarship"] = "Full Scholarship (100%)"
    elif marks >= 85:
        state["scholarship"] = "Partial Scholarship (50%)"
    elif marks >= 75:
        state["scholarship"] = "Merit Award (25%)"
    else:
        state["scholarship"] = "No Scholarship"

    console.print(f"[yellow]-> Scholarship Status: {state['scholarship']}[/yellow]")
    return state

def generate_welcome(state: StudentState) -> StudentState:
    """
    Node 4: Compose the final welcome message using all state fields.
    This node only READS -- it doesn't call any external service.
    """

    state["message"] = (
        f"Dear {state['name']},\n"
        f"Welcome to VIT University!\n"
        f"Your Student ID: {state['student_id']}\n"
        f"Department: {state['department']}\n"
        f"Scholarship: {state['scholarship']}\n"
        f"We look forward to your journey with us."
    )

    return state

builder=StateGraph(StudentState)

builder.add_node("create_id", create_id)
builder.add_node("assign_department", assign_department)
builder.add_node("check_scholarship", check_scholarship)
builder.add_node("generate_welcome", generate_welcome)

builder.add_edge("create_id","assign_department")
builder.add_edge("assign_department","check_scholarship")
builder.add_edge("check_scholarship","generate_welcome")

builder.set_entry_point("create_id")
builder.set_finish_point("generate_welcome")

graph=builder.compile()

def print_result(result: StudentState):
  table = Table(title=f"Admission Result -- {result['name']}", style="bold")
  table.add_column("Field", style="cyan", width=20)
  table.add_column("Value", style="white", width=40)
  table.add_row("Student ID", result["student_id"])
  table.add_row("Department", result["department"])
  table.add_row("Scholarship", result["scholarship"])
  console.print(table)
  console.print(f"\n[bold green]{result['message']}[/bold green]\n")
  console.print("-" * 60)

if __name__ == "__main__":
  students = [
  {"name": "Ananya", "age": 18, "marks": 97.0, "student_id": "", "department": "",
  "scholarship": "", "message": ""},
  {"name": "Ravi", "age": 19, "marks": 82.5, "student_id": "", "department": "",
  "scholarship": "", "message": ""},
  {"name": "John", "age": 18, "marks": 63.0, "student_id": "", "department": "",
  "scholarship": "", "message": ""},
  {"name": "Priya", "age": 18, "marks": 55.0, "student_id": "", "department": "",
  "scholarship": "", "message": ""},
  ]

  for student in students:
    console.print(f"\n[bold magenta]Processing: {student['name']} (Marks: {student['marks']})[/bold magenta]")
    result = graph.invoke(student)
    print_result(result)

