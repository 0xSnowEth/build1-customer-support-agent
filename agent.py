
import os
import anthropic
  

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
MAX_TOKENS = 4096

TOOLS = [
    {
        "name": "read_document",
        "description": "Read a company document or FAQ file from disk. Use this FIRST before answering any questions so you have the source material",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the .txt document file."
                }
            },
            "required": ["file_path"]
        }
    }
]


# Tool Implementations
def read_document(file_path: str) -> str:
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"
    
    
# Tool Dispatcher

def run_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "read_document":
        return read_document(tool_input["file_path"])
    else:
        return f"Unknown tool: {tool_name}"
    
    
def run_agent(goal: str) -> None:
    
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url="https://openrouter.ai/api"
    )
    
    messages = [{"role": "user", "content": goal}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,
            messages=messages,
            system = "You are a customer support agent. ALWAYS use read_document FIRST before answering anything. Answer ONLY based on what is in the document. If the answer is not in the document say exactly: I don't have that information. Let me connect you with a human agent. Never make up answers."
        )

        if response.stop_reason == "end_turn":
            final_text = " ".join(
                block.text for block in response.content 
                if hasattr(block, "text")
            )
            print(final_text)
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result_text = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })
            
            messages.append({"role": "user", "content": tool_results})
            continue
        
    
if __name__ == "__main__":
    user_goal = input("Ask a question: ").strip()
    run_agent(user_goal)
    
    