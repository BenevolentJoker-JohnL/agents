#!/usr/bin/env python3
"""
Simple test to verify agents can be used as a library.

This script can be run from anywhere to test that agents
works as an importable package.
"""

import sys
import os

# Add the parent directory to path for testing without installation
# (Remove this if you've installed via pip install -e .)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all main components can be imported."""
    print("Testing imports...")

    try:
        from agents import (
            # Clients
            BaseOllamaClient,
            LocalOllamaClient,
            DistributedOllamaClient,
            # Data classes
            ChatMessage,
            ChatResponse,
            # Agents
            BaseAgent,
            ChatAgent,
            CodingAgent,
            ReasoningAgent,
            ResearchAgent,
            SummarizationAgent,
            EmbeddingAgent,
            # Utils
            parse_json_response,
            check_node_health,
            retry_with_backoff,
            format_chat_history,
            estimate_tokens,
            truncate_to_token_limit,
            merge_response_chunks,
        )
        print("✓ All imports successful!\n")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}\n")
        return False


def test_basic_usage():
    """Test basic usage without actually calling Ollama."""
    print("Testing basic instantiation...")

    try:
        from agents import LocalOllamaClient, ChatAgent, ChatMessage

        # Create client (won't connect yet)
        client = LocalOllamaClient(
            model_name="llama3:latest",
            api_base="http://localhost:11434"
        )
        print(f"✓ Created client: {client.model_name}")

        # Create agent
        agent = ChatAgent(client)
        print(f"✓ Created agent: {agent.__class__.__name__}")

        # Create message
        msg = ChatMessage(role="user", content="Test")
        print(f"✓ Created message: {msg.role} - {msg.content}")

        print("\n✓ Basic usage test passed!\n")
        return True

    except Exception as e:
        print(f"✗ Basic usage test failed: {e}\n")
        return False


def test_utilities():
    """Test utility functions."""
    print("Testing utility functions...")

    try:
        from agents import (
            parse_json_response,
            estimate_tokens,
            truncate_to_token_limit,
        )

        # Test JSON parsing
        json_text = '{"key": "value"}'
        result = parse_json_response(json_text)
        assert result == {"key": "value"}
        print("✓ JSON parsing works")

        # Test token estimation
        text = "This is a test sentence."
        tokens = estimate_tokens(text)
        assert tokens > 0
        print(f"✓ Token estimation works ({tokens} tokens)")

        # Test truncation
        truncated = truncate_to_token_limit("Long text here", token_limit=2)
        assert len(truncated) <= 11  # 2 tokens * 4 chars + "..." = 11
        print("✓ Text truncation works")

        print("\n✓ Utility tests passed!\n")
        return True

    except Exception as e:
        print(f"✗ Utility test failed: {e}\n")
        return False


def test_as_dependency():
    """Simulate how another package would use agents."""
    print("Testing usage as a dependency...")

    try:
        # This is how another app/package would use agents
        from agents import LocalOllamaClient, ChatAgent

        class MyCustomApp:
            """Example: Using agents in your own class."""

            def __init__(self):
                self.client = LocalOllamaClient(
                    model_name="llama3:latest",
                    api_base="http://localhost:11434"
                )
                self.agent = ChatAgent(self.client)

            def process(self, message: str) -> dict:
                """Process a message (sync version for demo)."""
                return {
                    "status": "ready",
                    "message": message,
                    "agent": self.agent.__class__.__name__
                }

        # Instantiate the custom app
        app = MyCustomApp()
        result = app.process("Test message")

        assert result["status"] == "ready"
        assert result["agent"] == "ChatAgent"

        print(f"✓ Custom app integration works: {result}")
        print("\n✓ Dependency test passed!\n")
        return True

    except Exception as e:
        print(f"✗ Dependency test failed: {e}\n")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("agents - Library Usage Test")
    print("="*60)
    print()

    results = []

    # Run tests
    results.append(("Import Test", test_imports()))
    results.append(("Basic Usage Test", test_basic_usage()))
    results.append(("Utility Test", test_utilities()))
    results.append(("Dependency Test", test_as_dependency()))

    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)

    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {test_name}")
        if result:
            passed += 1

    print("-"*60)
    print(f"Results: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! agents is working as a library.")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
