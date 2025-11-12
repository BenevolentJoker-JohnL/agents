# agents Architecture

## Overview

agents is designed with a clean, layered architecture that separates concerns and maximizes modularity while minimizing overhead.

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│              (Your custom agents and logic)              │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                      Agent Layer                         │
│  BaseAgent, ChatAgent, CodingAgent, ReasoningAgent, etc.│
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     Client Layer                         │
│   LocalOllamaClient  │  DistributedOllamaClient         │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Transport Layer                        │
│         aiohttp (HTTP)  │  SOLLOL (Distributed)          │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Ollama Server(s)                      │
│              Single Node  │  Multi-Node Cluster          │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Client Layer

The client layer provides the interface to Ollama servers.

#### BaseOllamaClient (Abstract)

Defines the contract that all clients must implement:

```python
class BaseOllamaClient(abc.ABC):
    @abc.abstractmethod
    async def generate_embedding(text: str) -> List[float]

    @abc.abstractmethod
    async def chat(messages, **options) -> AsyncGenerator[ChatResponse]
```

**Design Goals:**
- Minimal interface
- Async-first design
- No unnecessary abstraction layers

#### LocalOllamaClient

Single-node implementation using `aiohttp`:

```python
┌──────────────────────────────────────────┐
│       LocalOllamaClient                  │
├──────────────────────────────────────────┤
│ + model_name: str                        │
│ + api_base: str                          │
│ + embed_model: str                       │
│ + keep_alive: str                        │
│ - _session: ClientSession                │
├──────────────────────────────────────────┤
│ + generate_embedding(text)               │
│ + chat(messages, **options)              │
│ + close()                                │
└──────────────────────────────────────────┘
```

**Key Features:**
- Connection pooling via aiohttp
- Streaming support
- Keep-alive configuration
- Minimal overhead (direct HTTP calls)

#### DistributedOllamaClient

Multi-node implementation with SOLLOL integration:

```python
┌──────────────────────────────────────────┐
│     DistributedOllamaClient              │
├──────────────────────────────────────────┤
│ + model_name: str                        │
│ + pool: OllamaPool                       │
├──────────────────────────────────────────┤
│ + generate_embedding(text)               │
│ + chat(messages, **options)              │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│          SOLLOL OllamaPool               │
├──────────────────────────────────────────┤
│ - Auto-discovery                         │
│ - Intelligent routing                    │
│ - Health checking                        │
│ - Load balancing                         │
└──────────────────────────────────────────┘
```

**Key Features:**
- Automatic node discovery
- Intelligent routing based on load and latency
- Health monitoring
- Failover support

### 2. Agent Layer

The agent layer provides reusable, task-specific behaviors.

```python
┌──────────────────────────────────────────┐
│          BaseAgent (Abstract)            │
├──────────────────────────────────────────┤
│ + client: BaseOllamaClient               │
│ + system_prompt: str                     │
├──────────────────────────────────────────┤
│ + run(messages, **options)               │
│ + run_sync(messages, **options)          │
│ + get_full_response(messages)            │
└──────────────────────────────────────────┘
             │
             ├─────────────────────────────┐
             │                             │
┌────────────▼────────┐      ┌────────────▼────────┐
│    ChatAgent        │      │   CodingAgent       │
├─────────────────────┤      ├─────────────────────┤
│ system_prompt: None │      │ system_prompt: ...  │
└─────────────────────┘      └─────────────────────┘
             │                             │
┌────────────▼────────┐      ┌────────────▼────────┐
│  ReasoningAgent     │      │  ResearchAgent      │
├─────────────────────┤      ├─────────────────────┤
│ system_prompt: ...  │      │ system_prompt: ...  │
└─────────────────────┘      └─────────────────────┘
```

**Design Pattern:**
- Template Method Pattern
- Strategy Pattern (via system prompts)
- Composition over inheritance

### 3. Data Flow

#### Request Flow (Streaming)

```
User Code
   │
   │ agent.run(messages)
   ▼
BaseAgent
   │
   │ + Prepend system prompt
   │ + Normalize messages
   ▼
BaseOllamaClient
   │
   ├─► LocalOllamaClient ──► aiohttp ──► Ollama Server
   │
   └─► DistributedOllamaClient ──► SOLLOL Pool ──► Ollama Cluster
                                      │
                                      ├─► Node selection
                                      ├─► Health check
                                      └─► Load balancing
                                              │
                                              ▼
                                      HTTP Request to selected node
                                              │
                                              ▼
                                      Streaming Response
                                              │
                                              ▼
                                      yield ChatResponse chunks
                                              │
                                              ▼
                                      Back to User Code
```

#### Embedding Flow

```
User Code
   │
   │ client.generate_embedding(text)
   ▼
BaseOllamaClient
   │
   ▼
POST /api/embeddings
   │
   ▼
Ollama Server
   │
   ▼
Return embedding vector
```

## Design Principles

### 1. Minimal Overhead

- Direct API calls without unnecessary abstraction
- No heavy dependencies (only aiohttp and numpy)
- Optional features (SOLLOL) via extras_require

### 2. Modularity

- Clean separation of concerns
- Composable components
- Easy to extend and customize

### 3. Async-First

- Native async/await support
- Streaming by default
- Sync wrappers provided for convenience

### 4. Type Safety

- Full type hints
- Clear contracts via abstract base classes
- Better IDE support and early error detection

### 5. Flexibility

- Support both local and distributed deployments
- Easy to switch between modes
- No vendor lock-in

## Patterns from Related Projects

### From SynapticLlamas

- **BaseAgent abstraction**: Clean agent interface with abstract `process()` method
- **OllamaNode pattern**: Health checking, metrics tracking
- **SOLLOL integration**: Intelligent routing and load balancing

### From Hydra

- **Async streaming**: Generator-based streaming without buffering
- **Reasoning modes**: Specialized agents for different tasks
- **No time-based timeouts**: Let models take as long as needed

### From FlockParser

- **Keep-alive configuration**: Control model memory retention
- **Embedding pipelines**: Separate embedding model support
- **RAG patterns**: Citation support and document processing

## Extension Points

### Custom Clients

Implement `BaseOllamaClient` for custom backends:

```python
class CustomClient(BaseOllamaClient):
    async def generate_embedding(self, text: str):
        # Custom implementation
        ...

    async def chat(self, messages, **options):
        # Custom implementation
        ...
```

### Custom Agents

Extend `BaseAgent` for specialized behaviors:

```python
class CustomAgent(BaseAgent):
    system_prompt = "..."

    async def run(self, messages, **options):
        # Preprocessing
        ...
        # Call parent
        async for chunk in super().run(messages, **options):
            # Postprocessing
            yield modified_chunk
```

### Custom Utilities

Add to `utils.py`:

```python
def custom_parser(text: str):
    # Custom parsing logic
    ...
```

## Performance Characteristics

### LocalOllamaClient

- **Latency**: Direct HTTP call overhead (~1-5ms)
- **Throughput**: Limited by single server capacity
- **Failure Mode**: Fails if server is down
- **Best For**: Development, single-server deployments

### DistributedOllamaClient

- **Latency**: SOLLOL routing overhead (~5-10ms)
- **Throughput**: Scales with number of nodes
- **Failure Mode**: Automatic failover to healthy nodes
- **Best For**: Production, high-availability deployments

## Testing Strategy

### Unit Tests

Test individual components in isolation:
- Client methods
- Agent behaviors
- Utility functions

### Integration Tests

Test component interactions:
- Agent + Client
- Streaming end-to-end
- Error handling

### System Tests

Test with real Ollama servers:
- Health checking
- Load balancing
- Failover scenarios

## Future Enhancements

Potential areas for improvement:

1. **Caching Layer**: Cache embeddings and common queries
2. **Retry Logic**: Configurable retry strategies
3. **Metrics**: Built-in performance monitoring
4. **Batching**: Batch multiple requests for efficiency
5. **Tool Use**: Function calling support
6. **Multi-modal**: Image and audio support
7. **Async Context Managers**: Better resource management

## Conclusion

The agents architecture prioritizes:

- **Simplicity**: Easy to understand and extend
- **Performance**: Minimal overhead, efficient streaming
- **Flexibility**: Works in many deployment scenarios
- **Reliability**: Built-in health checking and failover

This makes it suitable for both rapid prototyping and production deployments.
