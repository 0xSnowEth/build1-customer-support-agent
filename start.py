"""
My First AI Agent — using the Anthropic Python SDK
----------------------------------------------------
This agent accepts a goal, then loops using Claude's tool-use
feature until the task is complete.

Tools available:
  • web_search  — simulated search (returns mock data)
  • calculator  — evaluates a math expression safely

Setup:
  pip install anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."
  python agent.py
"""

import os
import math
import anthropic

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
MAX_TOKENS = 4096

# ── Tool definitions (sent to Claude so it knows what tools exist) ─────────────
#
# Each tool has:
#   name        – how Claude will call it
#   description – what it does (Claude reads this to decide when to use it)
#   input_schema – JSON Schema for the arguments Claude must supply

TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for current information on any topic. "
            "Use this when you need facts, news, or data you don't already know."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression and return the numeric result. "
            "Supports standard arithmetic, exponentiation (**), and basic math "
            "functions like sqrt(), sin(), cos(), log(), etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python-style math expression, e.g. '2 ** 10' or 'sqrt(144)'.",
                }
            },
            "required": ["expression"],
        },
    },
]

# ── Tool implementations ───────────────────────────────────────────────────────

def web_search(query: str) -> str:
    """
    Simulated web search — returns mock results.
    In a real project you'd call a search API (e.g. Brave, Serper, Tavily).
    """
    mock_results = {
        "default": (
            f"[Mock search results for '{query}']\n"
            "• Result 1: According to Wikipedia, this topic has a rich history dating back centuries.\n"
            "• Result 2: A 2024 study found significant developments in this area.\n"
            "• Result 3: Experts generally agree on three main points regarding this subject.\n"
            "Note: These are simulated results. Replace this function with a real search API for production use."
        ),
        "weather": (
            f"[Mock weather search for '{query}']\n"
            "Current conditions: 22°C, partly cloudy, humidity 65%.\n"
            "Forecast: Mild temperatures expected throughout the week."
        ),
        "price": (
            f"[Mock price search for '{query}']\n"
            "Current market price: approximately $42.50 USD.\n"
            "Price trend: +3.2% over the past 30 days."
        ),
    }

    # Pick a relevant mock result based on keywords in the query
    query_lower = query.lower()
    if any(word in query_lower for word in ["weather", "temperature", "forecast"]):
        return mock_results["weather"]
    elif any(word in query_lower for word in ["price", "cost", "dollar", "usd"]):
        return mock_results["price"]
    else:
        return mock_results["default"]


def calculator(expression: str) -> str:
    """
    Safely evaluate a math expression.
    Only allows numeric operations and functions from the math module.
    """
    # Build a safe namespace with math functions only
    safe_globals = {
        "__builtins__": {},  # block all built-ins
        **{name: getattr(math, name) for name in dir(math) if not name.startswith("_")},
    }

    try:
        result = eval(expression, safe_globals)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"


# ── Tool dispatcher ────────────────────────────────────────────────────────────

def run_tool(tool_name: str, tool_input: dict) -> str:
    """Route a tool call from Claude to the correct Python function."""
    if tool_name == "web_search":
        return web_search(tool_input["query"])
    elif tool_name == "calculator":
        return calculator(tool_input["expression"])
    else:
        return f"Unknown tool: {tool_name}"


# ── The agentic loop ───────────────────────────────────────────────────────────

def run_agent(goal: str) -> None:
    """
    Core agentic loop:

    1. Send the user's goal (plus any previous messages) to Claude.
    2. Check `stop_reason` in the response:
       - "tool_use"  → Claude wants to call a tool; execute it and loop back.
       - "end_turn"  → Claude is done; print the final answer and exit.
    3. Append every message (assistant + tool results) to `messages` so
       Claude keeps full context across iterations.
    """
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url="https://openrouter.ai/api"
    )
    
    
    

    # The conversation history — starts with just the user's goal
    messages = [{"role": "user", "content": goal}]

    print("\n" + "=" * 60)
    print(f"🎯  Goal: {goal}")
    print("=" * 60)

    step = 0

    while True:
        step += 1
        print(f"\n── Step {step}: Asking Claude ──────────────────────────────")

        # ── Send the current conversation to Claude ────────────────────────
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,         # make tools available to Claude
            messages=messages,   # full conversation history
            extra_headers={
                "HTTP-Referer": "https://github.com/yourname/first-ai-agent",
                "X-Title": "My First AI Agent",
            },
        )

        print(f"   stop_reason: {response.stop_reason}")

        # ── Claude is finished — print final answer and exit ───────────────
        if response.stop_reason == "end_turn":
            # Extract all text blocks from the final response
            final_text = " ".join(
                block.text
                for block in response.content
                if hasattr(block, "text")
            )
            print("\n" + "=" * 60)
            print("✅  Final Answer:")
            print("=" * 60)
            print(final_text)
            break

        # ── Claude wants to use one or more tools ─────────────────────────
        if response.stop_reason == "tool_use":
            # Append Claude's full response (which contains the tool call) to history
            messages.append({"role": "assistant", "content": response.content})

            # Collect results for all tool calls in this response
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n   🔧  Tool call : {block.name}")
                    print(f"       Input     : {block.input}")

                    # Execute the tool
                    result_text = run_tool(block.name, block.input)

                    print(f"       Result    : {result_text[:120]}{'...' if len(result_text) > 120 else ''}")

                    # Build a tool_result block for this specific call
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,   # must match the tool_use id
                        "content": result_text,
                    })

            # Append all tool results as a single user message
            # (Anthropic requires tool results in a "user" role message)
            messages.append({"role": "user", "content": tool_results})

            # Loop back → send updated conversation to Claude
            continue

        # ── Unexpected stop_reason — print whatever Claude returned ────────
        print(f"⚠️  Unexpected stop_reason: {response.stop_reason}")
        print(response.content)
        break


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Welcome to your first AI agent!")
    print("Type a goal and press Enter. Examples:")
    print("  • What is 2 to the power of 32?")
    print("  • Search for information about the Eiffel Tower and tell me 3 facts.")
    print("  • What is the square root of 1764, and search for info about that number.")
    print()

    user_goal = input("Your goal: ").strip()

    if not user_goal:
        user_goal = "What is 2 to the power of 10, and then search for information about that number?"
        print(f"(Using default goal: {user_goal})")

    run_agent(user_goal)