"""
Distributed Client Example
===========================

Demonstrates using DistributedOllamaClient with SOLLOL for load balancing
across multiple Ollama nodes.

Requirements:
- SOLLOL package installed: pip install sollol
- Multiple Ollama instances running on different nodes
"""

import asyncio
from agents import DistributedOllamaClient, ChatAgent


async def main():
    print("=== Distributed Client Example ===\n")

    try:
        # Create a distributed client with SOLLOL
        client = DistributedOllamaClient(
            model_name="llama3:latest",
            # SOLLOL will auto-discover nodes, or you can specify them:
            nodes=[
                "http://192.168.1.100:11434",
                "http://192.168.1.101:11434",
                "http://localhost:11434",
            ],
            enable_intelligent_routing=True,
            health_check_interval=120,  # Check node health every 2 minutes
        )

        # Create an agent
        agent = ChatAgent(client)

        print("Distributed client created with intelligent routing enabled.\n")

        # Send multiple requests - they will be distributed across nodes
        questions = [
            "What is machine learning?",
            "Explain quantum computing.",
            "What is blockchain technology?",
        ]

        for i, question in enumerate(questions, 1):
            print(f"\nRequest {i}: {question}")
            print("-" * 50)

            messages = [{"role": "user", "content": question}]

            # Stream response
            async for chunk in agent.run(messages):
                if chunk.content:
                    print(chunk.content, end="", flush=True)

                # Check if routing metadata is available
                if chunk.metadata and "node_url" in chunk.metadata:
                    node = chunk.metadata["node_url"]
                    print(f"\n[Routed to: {node}]")

            print("\n")

    except ImportError:
        print("Error: SOLLOL package not installed.")
        print("Install it with: pip install sollol")
        print("\nFalling back to local client...\n")

        # Fallback to local client
        from agents import LocalOllamaClient

        client = LocalOllamaClient(
            model_name="llama3:latest",
            api_base="http://localhost:11434"
        )

        agent = ChatAgent(client)
        messages = [{"role": "user", "content": "Hello!"}]
        response = agent.get_full_response(messages)
        print(response)

    finally:
        if hasattr(client, 'close'):
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())
