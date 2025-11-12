#!/usr/bin/env python3
"""
Example Application - Demonstrates using agents in an external application

This is a standalone application that imports and uses agents.
Run this from anywhere after installing agents.

Installation:
    cd /path/to/OllamaAgent
    pip install -e .

Usage:
    python example_app.py
"""

import asyncio
import sys
from typing import List, Dict

# Import from agents (this works after installation)
from agents import (
    LocalOllamaClient,
    ChatAgent,
    CodingAgent,
    ReasoningAgent,
    check_node_health,
)


class MyApplication:
    """Example application class that uses agents."""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """Initialize the application with Ollama integration."""
        print(f"Initializing application with Ollama at {ollama_url}...")

        # Check if Ollama is available
        if not check_node_health(ollama_url):
            print(f"⚠️  Warning: Ollama at {ollama_url} is not responding!")
            print("   Make sure Ollama is running: ollama serve")
            sys.exit(1)

        # Create a shared client
        self.client = LocalOllamaClient(
            model_name="llama3:latest",
            api_base=ollama_url,
            keep_alive="10m"  # Keep model in memory for 10 minutes
        )

        # Create different agents for different tasks
        self.chat_agent = ChatAgent(self.client)
        self.code_agent = CodingAgent(self.client)
        self.reasoning_agent = ReasoningAgent(self.client)

        print("✓ Application initialized successfully\n")

    async def process_user_query(self, query: str) -> str:
        """Process a user query using the appropriate agent."""
        # Simple routing logic based on keywords
        if any(word in query.lower() for word in ['code', 'function', 'implement', 'program']):
            print("→ Routing to Coding Agent")
            agent = self.code_agent
        elif any(word in query.lower() for word in ['analyze', 'explain', 'why', 'how']):
            print("→ Routing to Reasoning Agent")
            agent = self.reasoning_agent
        else:
            print("→ Routing to Chat Agent")
            agent = self.chat_agent

        # Get response
        messages = [{"role": "user", "content": query}]
        return agent.get_full_response(messages)

    async def interactive_mode(self):
        """Run in interactive mode."""
        print("=" * 60)
        print("Interactive Mode - Type 'quit' to exit")
        print("=" * 60)

        while True:
            try:
                query = input("\nYou: ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not query:
                    continue

                print("\nAssistant: ", end="", flush=True)
                response = await self.process_user_query(query)
                print(response)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    async def batch_mode(self, queries: List[str]):
        """Process multiple queries in batch."""
        print("=" * 60)
        print(f"Batch Mode - Processing {len(queries)} queries")
        print("=" * 60)

        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Query: {query}")
            print("-" * 60)

            response = await self.process_user_query(query)
            print(f"Response:\n{response[:200]}...")  # First 200 chars

            results.append({
                "query": query,
                "response": response
            })

        return results

    async def streaming_demo(self):
        """Demonstrate streaming responses."""
        print("=" * 60)
        print("Streaming Demo")
        print("=" * 60)

        query = "Tell me a short story about a robot learning to paint."
        print(f"\nQuery: {query}\n")
        print("Response: ", end="", flush=True)

        messages = [{"role": "user", "content": query}]
        async for chunk in self.chat_agent.run(messages):
            print(chunk.content, end="", flush=True)

        print("\n")

    async def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        await self.client.close()
        print("✓ Cleanup complete")


async def main():
    """Main entry point."""
    print("""
╔══════════════════════════════════════════════════════════╗
║         agents - Example Application               ║
║                                                          ║
║  Demonstrates how to integrate agents into         ║
║  your Python application                                 ║
╚══════════════════════════════════════════════════════════╝
""")

    # Create application instance
    app = MyApplication()

    try:
        # Demo 1: Single query
        print("\n" + "=" * 60)
        print("Demo 1: Single Query Processing")
        print("=" * 60)

        query = "What is machine learning?"
        print(f"\nQuery: {query}\n")
        response = await app.process_user_query(query)
        print(f"Response:\n{response[:300]}...\n")

        # Demo 2: Batch processing
        print("\n" + "=" * 60)
        print("Demo 2: Batch Processing")
        print("=" * 60)

        queries = [
            "What is Python?",
            "Write a function to check if a number is prime",
            "Explain why the sky is blue"
        ]

        await app.batch_mode(queries)

        # Demo 3: Streaming
        await app.streaming_demo()

        # Demo 4: Interactive mode (optional)
        print("\n" + "=" * 60)
        print("Would you like to try interactive mode? (y/n)")
        print("=" * 60)

        # For demo purposes, skip interactive mode in automated runs
        # Uncomment the next line to enable interactive mode
        # await app.interactive_mode()

    finally:
        await app.cleanup()


if __name__ == "__main__":
    # Run the application
    asyncio.run(main())
