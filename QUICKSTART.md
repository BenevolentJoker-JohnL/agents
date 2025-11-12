# agents - Quick Start Guide

Get up and running with agents in 5 minutes!

## Prerequisites

1. Python 3.8 or higher
2. Ollama installed and running locally (or accessible via network)
3. At least one model pulled (e.g., `ollama pull llama3`)

## Installation

### Option 1: From PyPI (Easiest)

```bash
pip install local-agents
```

### Option 2: From Source

```bash
git clone https://github.com/BenevolentJoker-JohnL/agents.git
cd agents
pip install -e .
```

## Your First Agent

Create a file called `my_first_agent.py`:

```python
import asyncio
from agents import LocalOllamaClient, ChatAgent


async def main():
    # Create a client pointing to your Ollama instance
    client = LocalOllamaClient(
        model_name="llama3:latest",
        api_base="http://localhost:11434"
    )

    # Create a chat agent
    agent = ChatAgent(client)

    # Ask a question
    response = agent.get_full_response([
        {"role": "user", "content": "What is Python?"}
    ])

    print(response)

    # Clean up
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python my_first_agent.py
```

## Examples

The framework includes several examples in `agents/examples/`:

### Run the Basic Chat Example

```bash
python agents/examples/basic_chat.py
```

### Run the Coding Agent Example

```bash
python agents/examples/coding_agent.py
```

### Run the Test Suite

```bash
python agents/examples/test_framework.py
```

## Common Use Cases

### 1. Simple Q&A

```python
from agents import LocalOllamaClient, ChatAgent

client = LocalOllamaClient("llama3:latest", "http://localhost:11434")
agent = ChatAgent(client)

answer = agent.get_full_response([
    {"role": "user", "content": "Explain quantum computing"}
])
print(answer)
```

### 2. Code Generation

```python
from agents import LocalOllamaClient, CodingAgent

client = LocalOllamaClient("codellama:latest", "http://localhost:11434")
agent = CodingAgent(client)

code = agent.get_full_response([
    {"role": "user", "content": "Write a binary search in Python"}
])
print(code)
```

### 3. Streaming Responses

```python
import asyncio

async def stream_example():
    client = LocalOllamaClient("llama3:latest", "http://localhost:11434")
    agent = ChatAgent(client)

    messages = [{"role": "user", "content": "Count to 10"}]

    async for chunk in agent.run(messages):
        print(chunk.content, end="", flush=True)

    await client.close()

asyncio.run(stream_example())
```

### 4. Multi-Turn Conversation

```python
client = LocalOllamaClient("llama3:latest", "http://localhost:11434")
agent = ChatAgent(client)

conversation = [
    {"role": "user", "content": "Hello!"},
]

# First response
response1 = agent.get_full_response(conversation)
conversation.append({"role": "assistant", "content": response1})

# Follow-up
conversation.append({"role": "user", "content": "What's your name?"})
response2 = agent.get_full_response(conversation)
print(response2)
```

## Next Steps

1. **Explore Examples**: Check out all examples in `agents/examples/`
2. **Read the README**: Full documentation in `agents/README.md`
3. **Create Custom Agents**: Extend `BaseAgent` for specialized tasks
4. **Try Distributed Mode**: Install SOLLOL for multi-node deployments

## Troubleshooting

### Ollama Not Running

```bash
# Start Ollama
ollama serve

# In another terminal, verify it's running
curl http://localhost:11434/api/tags
```

### Model Not Found

```bash
# List available models
ollama list

# Pull a model if needed
ollama pull llama3
```

### Connection Errors

Check that:
1. Ollama is running (`ollama serve`)
2. The URL is correct (default: `http://localhost:11434`)
3. No firewall is blocking the connection

### Import Errors

```bash
# Make sure dependencies are installed
pip install -r requirements.txt

# Or reinstall the package
pip install -e .
```

## Getting Help

- Check the examples in `agents/examples/`
- Read the full README: `agents/README.md`
- Review the source code (it's well-documented!)
- Open an issue on GitHub

## What's Next?

Now that you have the basics, try:

1. **Building a custom agent** for your specific use case
2. **Chaining multiple agents** together in a workflow
3. **Using embeddings** for semantic search
4. **Setting up distributed mode** with SOLLOL

Happy coding! 🚀
