# agents

A lightweight, modular framework for building agents with Ollama models. Designed for minimal overhead while supporting both single-node and distributed deployments.

## Features

- **Minimal Overhead**: Direct API calls with clean abstractions
- **Modular Design**: Easy to extend and customize
- **Dual Deployment Support**: Works with single Ollama instances or distributed clusters
- **Built-in Agents**: Pre-configured agents for common tasks
- **Streaming Support**: Native async streaming for all operations
- **SOLLOL Integration**: Optional intelligent load balancing across multiple nodes
- **Type-Safe**: Full type hints for better IDE support

## Installation

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd agents

# Install in development mode
pip install -e .
```

### With SOLLOL Support (Distributed Mode)

```bash
pip install -e .
pip install sollol
```

## Quick Start

### Basic Chat

```python
import asyncio
from agents import LocalOllamaClient, ChatAgent

async def main():
    # Create a client
    client = LocalOllamaClient(
        model_name="llama3:latest",
        api_base="http://localhost:11434"
    )

    # Create an agent
    agent = ChatAgent(client)

    # Get a response
    response = agent.get_full_response([
        {"role": "user", "content": "What is machine learning?"}
    ])
    print(response)

    await client.close()

asyncio.run(main())
```

### Streaming Responses

```python
async def stream_example():
    client = LocalOllamaClient("llama3:latest", "http://localhost:11434")
    agent = ChatAgent(client)

    messages = [{"role": "user", "content": "Tell me a story"}]

    async for chunk in agent.run(messages):
        print(chunk.content, end="", flush=True)

    await client.close()
```

## Architecture

### Clients

The framework provides three client types:

#### 1. BaseOllamaClient (Abstract)

Base interface defining common methods for all clients.

```python
class BaseOllamaClient(abc.ABC):
    @abc.abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        ...

    @abc.abstractmethod
    async def chat(self, messages, **options) -> AsyncGenerator[ChatResponse, None]:
        ...
```

#### 2. LocalOllamaClient

For single-node deployments. Uses `aiohttp` for HTTP requests.

```python
client = LocalOllamaClient(
    model_name="llama3:latest",
    api_base="http://localhost:11434",
    embed_model="mxbai-embed-large",  # Optional separate embedding model
    keep_alive="5m"  # Keep model in memory
)
```

#### 3. DistributedOllamaClient

For multi-node deployments with SOLLOL integration.

```python
client = DistributedOllamaClient(
    model_name="llama3:latest",
    nodes=["http://node1:11434", "http://node2:11434"],
    enable_intelligent_routing=True,
    health_check_interval=120
)
```

### Agents

Agents wrap clients with specialized behavior and prompts.

#### Built-in Agents

- **ChatAgent**: Basic conversational agent
- **CodingAgent**: Specialized for code generation
- **ReasoningAgent**: Optimized for analytical tasks
- **ResearchAgent**: Information synthesis and research
- **SummarizationAgent**: Text summarization
- **EmbeddingAgent**: Generate text embeddings

#### Using Agents

```python
from agents import CodingAgent

agent = CodingAgent(client)
code = agent.get_full_response([
    {"role": "user", "content": "Write a binary search function in Python"}
])
```

### Custom Agents

Create custom agents by extending `BaseAgent`:

```python
from agents import BaseAgent

class SQLAgent(BaseAgent):
    system_prompt = """You are an expert SQL developer.
    Generate clean, efficient SQL queries."""

    # Optionally override run() for custom behavior
    async def run(self, messages, **options):
        # Custom preprocessing
        ...
        # Call parent
        async for chunk in super().run(messages, **options):
            yield chunk
```

## Examples

### Code Generation

```python
from agents import LocalOllamaClient, CodingAgent

client = LocalOllamaClient("codellama:latest", "http://localhost:11434")
agent = CodingAgent(client)

code = agent.get_full_response([{
    "role": "user",
    "content": "Write a function to find prime numbers using the Sieve of Eratosthenes"
}])
print(code)
```

### Multi-Agent Workflow

```python
from agents import ResearchAgent, ReasoningAgent, SummarizationAgent

# Stage 1: Research
researcher = ResearchAgent(client)
research = researcher.get_full_response([
    {"role": "user", "content": "Research quantum computing applications"}
])

# Stage 2: Analysis
analyzer = ReasoningAgent(client)
analysis = analyzer.get_full_response([
    {"role": "user", "content": f"Analyze: {research}"}
])

# Stage 3: Summary
summarizer = SummarizationAgent(client)
summary = summarizer.get_full_response([
    {"role": "user", "content": f"Summarize: {analysis}"}
])
```

### Embeddings and Similarity

```python
from agents import EmbeddingAgent
import numpy as np

agent = EmbeddingAgent(client)

# Generate embeddings
texts = ["Machine learning is AI", "Deep learning uses neural networks"]
embeddings = []

for text in texts:
    responses = agent.run_sync([{"role": "user", "content": text}])
    embeddings.append(responses[0].metadata["embedding"])

# Calculate similarity
similarity = np.dot(embeddings[0], embeddings[1]) / (
    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
)
print(f"Similarity: {similarity}")
```

### Distributed Deployment

```python
from agents import DistributedOllamaClient, ChatAgent

# SOLLOL handles node discovery and load balancing
client = DistributedOllamaClient(
    model_name="llama3:latest",
    nodes=["http://192.168.1.100:11434", "http://192.168.1.101:11434"]
)

agent = ChatAgent(client)

# Requests are automatically distributed across nodes
for i in range(10):
    response = agent.get_full_response([
        {"role": "user", "content": f"Question {i}"}
    ])
    print(response)
```

## Utilities

The framework includes helpful utilities:

```python
from agents import (
    parse_json_response,      # Extract JSON from LLM responses
    check_node_health,        # Health check for Ollama nodes
    retry_with_backoff,       # Retry logic with exponential backoff
    format_chat_history,      # Format messages for display
    estimate_tokens,          # Rough token estimation
    truncate_to_token_limit,  # Truncate text to fit token limit
    merge_response_chunks,    # Merge streaming chunks
)

# Example: Parse JSON from model output
result = parse_json_response(model_output, strict=True)

# Example: Health check
is_healthy = check_node_health("http://localhost:11434")

# Example: Retry with backoff
result = retry_with_backoff(
    lambda: risky_operation(),
    max_retries=3,
    initial_delay=1.0
)
```

## Patterns from Related Projects

This framework incorporates proven patterns from:

- **SynapticLlamas**: Agent abstraction, node health checking, metrics tracking
- **Hydra**: Async streaming, reasoning modes, SOLLOL integration
- **FlockParser**: Keep-alive configuration, embedding pipelines

## API Reference

### ChatMessage

```python
@dataclass
class ChatMessage:
    role: str       # "user", "assistant", "system"
    content: str    # Message content
```

### ChatResponse

```python
@dataclass
class ChatResponse:
    content: str              # Response content
    role: str = "assistant"   # Role
    done: bool = False        # End of stream flag
    metadata: dict = None     # Additional data (tokens, timing, etc.)
```

## Configuration

### Keep-Alive

Control how long models stay in memory:

```python
client = LocalOllamaClient(
    model_name="llama3:latest",
    api_base="http://localhost:11434",
    keep_alive="5m"  # Keep for 5 minutes (or "1h", "30s", etc.)
)
```

### Model Options

Pass options to control generation:

```python
async for chunk in agent.run(
    messages,
    temperature=0.7,    # Control randomness
    top_p=0.9,          # Nucleus sampling
    max_tokens=4096,    # Maximum tokens to generate
    repeat_penalty=1.1  # Penalize repetition
):
    print(chunk.content)
```

## Performance Tips

1. **Reuse Clients**: Create one client and reuse it across multiple agents
2. **Use Streaming**: Stream responses for better perceived performance
3. **Keep-Alive**: Set appropriate keep-alive to avoid model reload delays
4. **Distributed Mode**: Use SOLLOL for automatic load balancing across nodes
5. **Connection Pooling**: `aiohttp` handles connection pooling automatically

## Contributing

Contributions are welcome! Areas for improvement:

- Additional specialized agents
- More utility functions
- Better error handling
- Performance optimizations
- Documentation improvements

## License

MIT License - See LICENSE file for details

## Related Projects

- [Hydra](../hydra) - Advanced reasoning engine with SOLLOL integration
- [SynapticLlamas](../SynapticLlamas) - Multi-agent orchestration framework
- [FlockParser](../FlockParser) - Document processing with RAG
- [SOLLOL](https://github.com/yourusername/sollol) - Intelligent Ollama load balancer

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing examples in `examples/`
- Review the source code documentation
