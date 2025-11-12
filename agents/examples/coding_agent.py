"""
Coding Agent Example
===================

Demonstrates using the CodingAgent for code generation tasks.
"""

import asyncio
from agents import LocalOllamaClient, CodingAgent


async def main():
    # Create a local client
    client = LocalOllamaClient(
        model_name="codellama:latest",  # Use a code-specialized model
        api_base="http://localhost:11434"
    )

    # Create a coding agent
    agent = CodingAgent(client)

    print("=== Coding Agent Example ===\n")

    # Request code generation
    messages = [
        {
            "role": "user",
            "content": "Write a Python function to calculate the Fibonacci sequence up to n terms using memoization."
        }
    ]

    print("Generating code...\n")
    code = agent.get_full_response(messages)
    print(code)

    # Another example: code review
    print("\n" + "="*50 + "\n")
    print("Requesting code explanation...\n")

    messages = [
        {
            "role": "user",
            "content": """Explain this code and suggest improvements:

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
"""
        }
    ]

    explanation = agent.get_full_response(messages)
    print(explanation)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
