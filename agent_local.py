"""
AI Agent System using Qwen 3 and Ollama

This script demonstrates how to:
1. Create custom tools for agents
2. Build an AI agent that can use tools
3. Execute agent queries with tool calling
"""

import datetime
from langchain.agents import tool
from langchain_ollama import ChatOllama
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor


# ============================================================================
# STEP 1: Define Custom Tools
# ============================================================================

@tool
def get_current_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Returns the current date and/or time formatted according to Python strftime format codes.

    Use this tool when the user asks for the current date, time, or both.

    Common format examples:
    - '%Y-%m-%d' for date only (e.g., 2024-01-15)
    - '%H:%M:%S' for time only (e.g., 14:30:45)
    - '%Y-%m-%d %H:%M:%S' for both (e.g., 2024-01-15 14:30:45)
    - '%B %d, %Y' for long date (e.g., January 15, 2024)
    - '%I:%M %p' for 12-hour time (e.g., 02:30 PM)

    Args:
        format: A strftime format string (default: "%Y-%m-%d %H:%M:%S")

    Returns:
        A formatted string with the current date/time
    """
    try:
        return datetime.datetime.now().strftime(format)
    except Exception as e:
        return f"Error formatting datetime: {e}"


@tool
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.

    Use this tool to perform calculations when users ask math questions.

    Examples:
    - "2 + 2" returns "4"
    - "10 * 5" returns "50"
    - "(100 - 20) / 4" returns "20.0"

    Args:
        expression: A mathematical expression as a string

    Returns:
        The result of the calculation as a string
    """
    try:
        # Note: eval() is used here for simplicity in a local environment
        # In production, use a safer alternative like ast.literal_eval or a math parser
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error calculating: {e}"


# Define all available tools
tools = [get_current_datetime, calculate]


# ============================================================================
# STEP 2-4: Agent Setup Functions
# ============================================================================

def get_agent_llm(model_name="qwen3:8b", temperature=0):
    """
    Initialize the language model for the agent.

    Args:
        model_name: Qwen model to use (qwen3:8b recommended)
        temperature: Controls randomness (0 = deterministic, 1 = creative)

    Returns:
        ChatOllama instance
    """
    llm = ChatOllama(
        model=model_name,
        temperature=temperature
    )
    return llm


def get_agent_prompt(prompt_hub_name="hwchase17/openai-tools-agent"):
    """
    Pull a prompt template from LangChain Hub.

    Args:
        prompt_hub_name: Name of the prompt in LangChain Hub

    Returns:
        Prompt template
    """
    prompt = hub.pull(prompt_hub_name)
    return prompt


def build_agent(llm, tools, prompt):
    """
    Create the agent with tools and prompt.

    Args:
        llm: Language model instance
        tools: List of available tools
        prompt: Prompt template

    Returns:
        Agent runnable
    """
    agent = create_tool_calling_agent(llm, tools, prompt)
    return agent


# ============================================================================
# STEP 5-6: Agent Executor and Execution
# ============================================================================

def create_agent_executor(agent, tools):
    """
    Create an executor that runs the agent with tools.

    Args:
        agent: Agent runnable
        tools: List of available tools

    Returns:
        AgentExecutor instance
    """
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True  # Set to False to reduce output
    )
    return agent_executor


def run_agent(executor, user_input):
    """
    Execute the agent with a user query.

    Args:
        executor: AgentExecutor instance
        user_input: User's question or command

    Returns:
        Agent's response
    """
    print(f"\n{'='*60}")
    print(f"User: {user_input}")
    print(f"{'='*60}\n")

    response = executor.invoke({"input": user_input})

    print(f"\n{'='*60}")
    print(f"Final Answer: {response['output']}")
    print(f"{'='*60}\n")

    return response['output']


def main():
    """Main execution function."""
    print("\n🤖 Starting AI Agent System...\n")

    # Initialize components
    print("⏳ Loading language model...")
    agent_llm = get_agent_llm(model_name="qwen3:8b")
    print("✓ Language model loaded")

    print("⏳ Loading agent prompt...")
    agent_prompt = get_agent_prompt()
    print("✓ Prompt loaded")

    print("⏳ Building agent...")
    agent_runnable = build_agent(agent_llm, tools, agent_prompt)
    print("✓ Agent built")

    print("⏳ Creating agent executor...")
    agent_executor = create_agent_executor(agent_runnable, tools)
    print("✓ Agent ready\n")

    # Example queries - modify these to test different scenarios!
    print("📝 Running sample queries...\n")

    run_agent(agent_executor, "What is the current date?")
    run_agent(agent_executor, "What time is it in HH:MM format?")
    run_agent(agent_executor, "Calculate 25 * 4 + 10")
    run_agent(agent_executor, "What is 100 divided by 5?")

    print("\n✅ Agent system demo complete!")
    print("\n💡 Tip: Add more @tool decorated functions to extend agent capabilities")
    print("💡 Tip: Modify the queries in main() to test your own questions")


if __name__ == "__main__":
    main()
