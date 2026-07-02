from typing import TypedDict,Optional
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.panel import Panel
import math
import json

console = Console()

llm=ChatOllama(model="qwen2.5:3b",temperature=0)

class AgentState(TypedDict):
  query: str
  tool_name: str # Which tool the LLM chose
  tool_input: str # Input to pass to the tool
  tool_result: str # Output from the tool
  final_answer: str # Human-readable final response

def calculator_tool(expression: str) -> str:
  """
  Safe mathematical expression evaluator.
  Supports: +, -, *, /, **, sqrt, sin, cos, log, pi, e
  """
  try:
    expression = expression.replace("^", "**")
    expression = expression.replace("sqrt(", "math.sqrt(")
    expression = expression.replace("sin(", "math.sin(")
    expression = expression.replace("cos(", "math.cos(")
    expression = expression.replace("log(", "math.log(")
    expression = expression.replace("pi", str(math.pi))
    expression = expression.replace("e", str(math.e))
    allowed = set("0123456789+-*/.() math")
    if not all(c in allowed or c.isalpha() for c in expression):
      return "Error: Invalid characters in expression"
    result = eval(expression, {"__builtins__": {}, "math": math})
    return f"{result:.4f}" if isinstance(result, float) else str(result)
  except Exception as ex:
    return f"Calculator Error: {str(ex)}"

def weather_tool(city: str) -> str:
  """
  Mock weather tool -- simulates a real weather API response.
  In production, replace with: requests.get(f'https://api.openweathermap.org/...')
  """
  mock_weather_db = {
    "chennai": {"temp": 34, "humidity": 78, "condition": "Partly Cloudy", "wind": "18km/h"},
    "coimbatore": {"temp": 29, "humidity": 65, "condition": "Clear Sky", "wind": "12km/h"},
    "mumbai": {"temp": 31, "humidity": 85, "condition": "Humid and Hazy", "wind": "22km/h"},
    "delhi": {"temp": 28, "humidity": 55, "condition": "Sunny", "wind": "10km/h"},
    "bangalore": {"temp": 25, "humidity": 60, "condition": "Light Drizzle", "wind": "8km/h"},
    "hyderabad": {"temp": 32, "humidity": 58, "condition": "Mostly Sunny", "wind": "15km/h"},
    "london": {"temp": 12, "humidity": 80, "condition": "Overcast", "wind": "25km/h"},
    "new york": {"temp": 18, "humidity": 62, "condition": "Partly Cloudy", "wind": "20km/h"},
  }
  city_lower = city.lower().strip()
  if city_lower in mock_weather_db:
    w = mock_weather_db[city_lower]
    return (
      f"Weather in {city.title()}: {w['condition']}, "
      f"Temp: {w['temp']}C, Humidity: {w['humidity']}%, "
      f"Wind: {w['wind']}"
    )
  return f"Weather data not available for '{city}'. Try: Chennai, Coimbatore, Mumbai, Delhi, Bangalore."

def unit_converter_tool(query: str) -> str:
  """
  Converts common units.
  Query format examples:
  '100 km to miles'
  '50 celsius to fahrenheit'
  '5 kg to pounds'
  """
  query = query.lower()
  try:
    if "celsius" in query and "fahrenheit" in query:
      val = float(''.join(c for c in query.split("celsius")[0] if c.isdigit() or c == '.'))
      return f"{val}C = {(val * 9/5 + 32):.2f}F"
    if "fahrenheit" in query and "celsius" in query:
      val = float(''.join(c for c in query.split("fahrenheit")[0] if c.isdigit() or c == '.'))
      return f"{val}F = {((val - 32) * 5/9):.2f}C"
    if "km" in query and "miles" in query:
      val = float(''.join(c for c in query.split("km")[0] if c.isdigit() or c == '.'))
      return f"{val} km = {(val * 0.6214):.4f} miles"
    if "miles" in query and "km" in query:
      val = float(''.join(c for c in query.split("miles")[0] if c.isdigit() or c == '.'))
      return f"{val} miles = {(val * 1.6093):.4f} km"
    if "kg" in query and ("lb" in query or "pound" in query):
      val = float(''.join(c for c in query.split("kg")[0] if c.isdigit() or c == '.'))
      return f"{val} kg = {(val * 2.2046):.4f} pounds"
    return "Unit conversion not recognised. Try: '100 km to miles', '37 celsius to fahrenheit', '70 kg to pounds'"
  except Exception:
    return "Could not parse unit conversion query."
  
def decide_tool(state:AgentState)->AgentState:
  """
  LLM reads the query and decides which tool to use.
  Returns a JSON object with tool_name and tool_input.
  """

  prompt=f"""You are a smart assistant with access to these tools:
  1. calculator(expression) - Evaluates mathematical expressions.
  2. weather(city) - Provides current weather for a city.
  3. unit_converter(query) - Converts units (e.g., km/miles, celsius/fahrenheit, kg/pounds).
  4. none - for general queries, explanations, or if no tool is needed.
  Your task is to read the user's query and decide which tool to use.

  User query: {state['query']}
  Respond with ONLY a valid JSON object, no explanation, no markdown:
  {{"tool": "calculator", "input": "876 * 54"}}
  {{"tool": "weather", "input": "Chennai"}}
  {{"tool": "unit_converter", "input": "100 km to miles"}}
  {{"tool": "none", "input": ""}}
  """

  result=llm.invoke(prompt)
  raw=result.content.strip()
  raw = raw.replace("```json", "").replace("```", "").strip()

  try:
    parsed=json.loads(raw)
    state["tool_name"]=parsed.get("tool","none")
    state["tool_input"]=parsed.get("input","")
  except json.JSONDecodeError:
    state["tool_name"]="none"
    state["tool_input"]=""
  
  console.print(f"[cyan]LLM chose tool: [bold]{state['tool_name']}[/bold] | input: {state['tool_input']}[/cyan]")

  return state

def run_calculator(state:AgentState)->AgentState:
  console.print(f"[yellow]Calculator: {state['tool_input']}[/yellow]")
  state["tool_result"]=calculator_tool(state["tool_input"])
  return state

def run_weather(state: AgentState) -> AgentState:
  console.print(f"[blue]Weather lookup: {state['tool_input']}[/blue]")
  state["tool_result"] = weather_tool(state["tool_input"])
  return state

def run_unit_converter(state: AgentState) -> AgentState:
  console.print(f"[magenta]Unit converter: {state['tool_input']}[/magenta]")
  state["tool_result"] = unit_converter_tool(state["tool_input"])
  return state

def direct_answer(state:AgentState)->AgentState:
  """
  LLM generates a direct answer without using any tools.
  """
  console.print(f"[green]Direct LLM answer[/green]")
  result=llm.invoke(state["query"])
  state["tool_result"]=result.content.strip()
  return state

def synthesize(state:AgentState)->AgentState:
  """
  Takes the tool result and generates a final human-readable answer.
  Skipped if direct_answer already handled it.
  """

  if state["tool_name"]=="none":
    state["final_answer"]=state["tool_result"]
    return state
  
  prompt=f"""Original question: {state['query']}
  Tool used: {state['tool_name']}
  Tool result: {state['tool_result']}

  Write a concise, human-readable answer to the original question based on the tool result.
  """
  
  result=llm.invoke(prompt)
  state["final_answer"]=result.content.strip()
  return state

def tool_router(state: AgentState) -> str:
  tool = state["tool_name"]
  if tool == "calculator":
    return "calculator"
  elif tool == "weather":
    return "weather"
  elif tool == "unit_converter":
    return "unit_converter"
  else:
    return "direct"

builder=StateGraph(AgentState)

builder.add_node("decide_tool", decide_tool)
builder.add_node("calculator", run_calculator)
builder.add_node("weather", run_weather)
builder.add_node("unit_converter", run_unit_converter)
builder.add_node("direct", direct_answer)
builder.add_node("synthesize", synthesize)
builder.add_conditional_edges(
 "decide_tool",
 tool_router,
 {
    "calculator": "calculator",
    "weather": "weather",
    "unit_converter": "unit_converter",
    "direct": "direct",
 }
)

builder.add_edge("calculator", "synthesize")
builder.add_edge("weather", "synthesize")
builder.add_edge("unit_converter", "synthesize")
builder.add_edge("direct", "synthesize")
builder.set_entry_point("decide_tool")
builder.set_finish_point("synthesize")

graph = builder.compile()

queries = [
 "What is 876 * 54?",
 "What is sqrt(144) + 25?",
 "What is the weather in Chennai?",
 "How is the weather in Coimbatore today?",
 "Convert 120 km to miles",
 "Convert 37 celsius to fahrenheit",
 "What is the capital of France?",
 "What is 15% of 3500?",
]
if __name__ == "__main__":
  console.print("\n[bold magenta]=== ReAct Agent with Tool Calling ===[/bold magenta]\n")
  for query in queries:
    console.print(f"\n[bold white]Query:[/bold white] {query}")
    result = graph.invoke({
    "query": query,
    "tool_name": "",
    "tool_input": "",
    "tool_result": "",
    "final_answer": "",
    })
    console.print(Panel(
    result["final_answer"],
    title="[bold green]Answer[/bold green]",
    border_style="green"
    ))



