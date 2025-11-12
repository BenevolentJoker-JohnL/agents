"""
Basic Chat Example
==================

Demonstrates simple chat interaction with a local Ollama instance.
"""

import asyncio
from agents import LocalOllamaClient, ChatAgent


async def main():
    # Create a local client
    client = LocalOllamaClient(
        model_name="llama3:latest",
        api_base="http://localhost:11434"
    )

    # Create a chat agent
    agent = ChatAgent(client)

    print("=== Basic Chat Example ===\n")

    # Single message
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]

    # Stream the response
    print("Streaming response:")
    async for chunk in agent.run(messages):
        print(chunk.content, end="", flush=True)
    print("\n")

    # Get full response at once
    response = agent.get_full_response(messages)
    print(f"\nFull response:\n{response}")

    # Close the client
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
