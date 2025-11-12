"""
Simple Test Example
===================

Demonstrates basic testing patterns for agents.
For production, use pytest with async support.
"""

import asyncio
from agents import (
    LocalOllamaClient,
    ChatAgent,
    CodingAgent,
    parse_json_response,
    check_node_health,
)


async def test_basic_chat():
    """Test basic chat functionality."""
    print("Test 1: Basic Chat")
    print("-" * 50)

    client = LocalOllamaClient(
        model_name="llama3:latest",
        api_base="http://localhost:11434"
    )

    agent = ChatAgent(client)

    messages = [{"role": "user", "content": "Say 'Hello, World!'"}]

    try:
        response = agent.get_full_response(messages)
        assert len(response) > 0, "Response should not be empty"
        print("✓ Basic chat test passed")
        print(f"  Response: {response[:100]}...\n")
        return True
    except Exception as e:
        print(f"✗ Basic chat test failed: {e}\n")
        return False
    finally:
        await client.close()


async def test_streaming():
    """Test streaming responses."""
    print("Test 2: Streaming")
    print("-" * 50)

    client = LocalOllamaClient(
        model_name="llama3:latest",
        api_base="http://localhost:11434"
    )

    agent = ChatAgent(client)
    messages = [{"role": "user", "content": "Count to 5"}]

    try:
        chunk_count = 0
        async for chunk in agent.run(messages, stream=True):
            chunk_count += 1

        assert chunk_count > 0, "Should receive at least one chunk"
        print(f"✓ Streaming test passed ({chunk_count} chunks)\n")
        return True
    except Exception as e:
        print(f"✗ Streaming test failed: {e}\n")
        return False
    finally:
        await client.close()


def test_json_parsing():
    """Test JSON parsing utility."""
    print("Test 3: JSON Parsing")
    print("-" * 50)

    # Test cases
    tests = [
        ('{"key": "value"}', True),
        ('```json\n{"key": "value"}\n```', True),
        ('Some text {"key": "value"} more text', True),
        ('Not JSON at all', False),
    ]

    passed = 0
    for text, should_succeed in tests:
        result = parse_json_response(text, strict=False)
        if should_succeed:
            if result is not None:
                passed += 1
            else:
                print(f"  ✗ Failed to parse: {text[:50]}")
        else:
            if result is None:
                passed += 1
            else:
                print(f"  ✗ Should not have parsed: {text[:50]}")

    total = len(tests)
    print(f"✓ JSON parsing test: {passed}/{total} passed\n")
    return passed == total


def test_health_check():
    """Test node health checking."""
    print("Test 4: Health Check")
    print("-" * 50)

    # Test localhost
    is_healthy = check_node_health("http://localhost:11434", timeout=2.0)

    if is_healthy:
        print("✓ Health check passed: localhost is healthy\n")
        return True
    else:
        print("⚠ Health check: localhost appears to be down\n")
        return False


async def test_agent_types():
    """Test different agent types."""
    print("Test 5: Agent Types")
    print("-" * 50)

    client = LocalOllamaClient(
        model_name="llama3:latest",
        api_base="http://localhost:11434"
    )

    try:
        # Test coding agent
        coding_agent = CodingAgent(client)
        messages = [{"role": "user", "content": "Write a hello world function"}]

        response = coding_agent.get_full_response(messages)
        assert len(response) > 0, "Coding agent response should not be empty"

        print("✓ Agent types test passed\n")
        return True
    except Exception as e:
        print(f"✗ Agent types test failed: {e}\n")
        return False
    finally:
        await client.close()


async def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("agents Framework Test Suite")
    print("=" * 60)
    print()

    results = []

    # Async tests
    results.append(await test_basic_chat())
    results.append(await test_streaming())
    results.append(await test_agent_types())

    # Sync tests
    results.append(test_json_parsing())
    results.append(test_health_check())

    # Summary
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Test Summary: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} test(s) failed")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
