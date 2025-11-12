# Integration Guide - Using agents in Your Applications

agents is designed to be easily integrated into any Python application. This guide shows you how.

## Installation Methods

### Method 1: Install as a Package (Recommended for Production)

```bash
# From the OllamaAgent directory
pip install -e .

# Or install from a specific location
pip install -e /path/to/OllamaAgent

# With distributed support
pip install -e ".[distributed]"
```

After installation, import from anywhere:

```python
from agents import LocalOllamaClient, ChatAgent
```

### Method 2: Add to PYTHONPATH (Development)

```bash
export PYTHONPATH="/path/to/OllamaAgent:$PYTHONPATH"
```

Or in Python:

```python
import sys
sys.path.insert(0, '/path/to/OllamaAgent')
from agents import LocalOllamaClient, ChatAgent
```

### Method 3: Copy Into Your Project

```bash
cp -r /path/to/OllamaAgent/agents /path/to/your/project/
```

Then import as a local module:

```python
from agents import LocalOllamaClient, ChatAgent
```

## Integration Examples

### 1. Flask Web Application

```python
from flask import Flask, request, jsonify, Response
from agents import LocalOllamaClient, ChatAgent
import asyncio

app = Flask(__name__)

# Initialize client once at startup
client = LocalOllamaClient(
    model_name="llama3:latest",
    api_base="http://localhost:11434",
    keep_alive="10m"
)
agent = ChatAgent(client)

@app.route('/chat', methods=['POST'])
def chat():
    """Simple chat endpoint."""
    data = request.json
    message = data.get('message', '')

    # Use sync wrapper for Flask
    response = agent.get_full_response([
        {"role": "user", "content": message}
    ])

    return jsonify({"response": response})

@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint."""
    data = request.json
    message = data.get('message', '')

    async def generate():
        async for chunk in agent.run([{"role": "user", "content": message}]):
            yield f"data: {chunk.content}\n\n"

    # Run async generator in sync context
    def sync_generate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async_gen = generate()
        try:
            while True:
                try:
                    chunk = loop.run_until_complete(async_gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    return Response(sync_generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### 2. FastAPI Application (Async-Native)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agents import LocalOllamaClient, ChatAgent
import asyncio

app = FastAPI()

# Initialize client
client = LocalOllamaClient(
    model_name="llama3:latest",
    api_base="http://localhost:11434"
)
agent = ChatAgent(client)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    """Async chat endpoint."""
    messages = [{"role": "user", "content": request.message}]

    # Collect full response
    chunks = []
    async for chunk in agent.run(messages):
        chunks.append(chunk.content)

    return {"response": "".join(chunks)}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming response."""
    messages = [{"role": "user", "content": request.message}]

    async def generate():
        async for chunk in agent.run(messages):
            yield f"data: {chunk.content}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.on_event("shutdown")
async def shutdown():
    """Clean up on shutdown."""
    await client.close()
```

### 3. Django Application

```python
# views.py
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from agents import LocalOllamaClient, ChatAgent
import json
import asyncio

# Initialize once (consider using Django's app config)
client = LocalOllamaClient(
    model_name="llama3:latest",
    api_base="http://localhost:11434"
)
agent = ChatAgent(client)

@csrf_exempt
def chat_view(request):
    """Chat endpoint for Django."""
    if request.method == 'POST':
        data = json.loads(request.body)
        message = data.get('message', '')

        # Use sync wrapper
        response = agent.get_full_response([
            {"role": "user", "content": message}
        ])

        return JsonResponse({"response": response})
```

### 4. Streamlit Application

```python
import streamlit as st
from agents import LocalOllamaClient, ChatAgent

# Initialize in session state
if 'client' not in st.session_state:
    st.session_state.client = LocalOllamaClient(
        model_name="llama3:latest",
        api_base="http://localhost:11434"
    )
    st.session_state.agent = ChatAgent(st.session_state.client)
    st.session_state.messages = []

st.title("Chat with Ollama")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Your message"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Stream response
        import asyncio

        async def get_response():
            full = ""
            async for chunk in st.session_state.agent.run(
                st.session_state.messages
            ):
                full += chunk.content
                message_placeholder.write(full + "▌")
            message_placeholder.write(full)
            return full

        full_response = asyncio.run(get_response())

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })
```

### 5. Command-Line Tool

```python
#!/usr/bin/env python3
"""
Simple CLI tool using agents
"""
import argparse
import asyncio
from agents import LocalOllamaClient, ChatAgent, CodingAgent

async def main():
    parser = argparse.ArgumentParser(description='Chat with Ollama')
    parser.add_argument('message', help='Message to send')
    parser.add_argument('--model', default='llama3:latest', help='Model to use')
    parser.add_argument('--url', default='http://localhost:11434', help='Ollama URL')
    parser.add_argument('--code', action='store_true', help='Use coding agent')

    args = parser.parse_args()

    # Create client
    client = LocalOllamaClient(
        model_name=args.model,
        api_base=args.url
    )

    # Create agent
    agent = CodingAgent(client) if args.code else ChatAgent(client)

    # Get response
    print("Response:")
    async for chunk in agent.run([{"role": "user", "content": args.message}]):
        print(chunk.content, end="", flush=True)
    print("\n")

    await client.close()

if __name__ == '__main__':
    asyncio.run(main())
```

### 6. Background Task/Worker

```python
from celery import Celery
from agents import LocalOllamaClient, ResearchAgent

app = Celery('tasks', broker='redis://localhost:6379')

# Global client (initialized per worker)
client = None
agent = None

@app.task
def research_task(topic):
    """Background research task."""
    global client, agent

    if client is None:
        client = LocalOllamaClient(
            model_name="llama3:latest",
            api_base="http://localhost:11434"
        )
        agent = ResearchAgent(client)

    result = agent.get_full_response([
        {"role": "user", "content": f"Research: {topic}"}
    ])

    return result
```

### 7. Jupyter Notebook

```python
# Cell 1: Import and setup
from agents import LocalOllamaClient, ChatAgent, CodingAgent
import asyncio

client = LocalOllamaClient(
    model_name="llama3:latest",
    api_base="http://localhost:11434"
)

# Cell 2: Use the agent
agent = ChatAgent(client)

response = agent.get_full_response([
    {"role": "user", "content": "Explain quantum computing"}
])

print(response)

# Cell 3: Streaming example
async def stream_example():
    async for chunk in agent.run([
        {"role": "user", "content": "Tell me a joke"}
    ]):
        print(chunk.content, end="", flush=True)

await stream_example()
```

### 8. Multi-Process Application

```python
from multiprocessing import Process, Queue
from agents import LocalOllamaClient, ChatAgent

def worker(task_queue, result_queue):
    """Worker process."""
    client = LocalOllamaClient(
        model_name="llama3:latest",
        api_base="http://localhost:11434"
    )
    agent = ChatAgent(client)

    while True:
        task = task_queue.get()
        if task is None:
            break

        result = agent.get_full_response([
            {"role": "user", "content": task}
        ])
        result_queue.put(result)

# Main process
if __name__ == '__main__':
    task_queue = Queue()
    result_queue = Queue()

    # Start workers
    workers = [
        Process(target=worker, args=(task_queue, result_queue))
        for _ in range(4)
    ]

    for w in workers:
        w.start()

    # Submit tasks
    tasks = ["Task 1", "Task 2", "Task 3", "Task 4"]
    for task in tasks:
        task_queue.put(task)

    # Get results
    results = [result_queue.get() for _ in tasks]

    # Stop workers
    for _ in workers:
        task_queue.put(None)

    for w in workers:
        w.join()
```

## Best Practices

### 1. Client Reuse

**DO**: Create one client and reuse it

```python
# Good
client = LocalOllamaClient("llama3:latest", "http://localhost:11434")
agent1 = ChatAgent(client)
agent2 = CodingAgent(client)
```

**DON'T**: Create multiple clients unnecessarily

```python
# Bad - wastes connections
agent1 = ChatAgent(LocalOllamaClient(...))
agent2 = CodingAgent(LocalOllamaClient(...))
```

### 2. Resource Cleanup

Always close clients when done:

```python
try:
    client = LocalOllamaClient(...)
    # Use client
finally:
    await client.close()
```

Or use context managers (if you add them):

```python
async with LocalOllamaClient(...) as client:
    agent = ChatAgent(client)
    # Use agent
```

### 3. Error Handling

```python
from agents import LocalOllamaClient, ChatAgent

client = LocalOllamaClient("llama3:latest", "http://localhost:11434")
agent = ChatAgent(client)

try:
    response = agent.get_full_response([
        {"role": "user", "content": "Hello"}
    ])
except RuntimeError as e:
    print(f"Model error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 4. Configuration Management

```python
import os
from agents import LocalOllamaClient

# Use environment variables
client = LocalOllamaClient(
    model_name=os.getenv("OLLAMA_MODEL", "llama3:latest"),
    api_base=os.getenv("OLLAMA_URL", "http://localhost:11434"),
    keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "5m")
)
```

## Common Integration Patterns

### Singleton Pattern (Web Apps)

```python
# app/ollama_service.py
from agents import LocalOllamaClient, ChatAgent

class OllamaService:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_client(self):
        if self._client is None:
            self._client = LocalOllamaClient(
                model_name="llama3:latest",
                api_base="http://localhost:11434"
            )
        return self._client

    def get_agent(self, agent_class):
        return agent_class(self.get_client())

# Usage
service = OllamaService()
agent = service.get_agent(ChatAgent)
```

### Factory Pattern

```python
from agents import LocalOllamaClient, ChatAgent, CodingAgent

class AgentFactory:
    def __init__(self, client):
        self.client = client

    def create_agent(self, agent_type):
        agents = {
            'chat': ChatAgent,
            'code': CodingAgent,
            # ... more
        }
        agent_class = agents.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return agent_class(self.client)

# Usage
client = LocalOllamaClient("llama3:latest", "http://localhost:11434")
factory = AgentFactory(client)
chat_agent = factory.create_agent('chat')
code_agent = factory.create_agent('code')
```

## Troubleshooting Integration Issues

### Import Errors

```bash
# Verify installation
pip list | grep ollama

# Reinstall
pip install -e /path/to/OllamaAgent

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Async in Sync Contexts

If you need to call async code from sync context:

```python
import asyncio

def sync_function():
    async def async_work():
        async for chunk in agent.run(messages):
            yield chunk

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(collect_async(async_work()))
    finally:
        loop.close()

def collect_async(async_gen):
    result = []
    async def collect():
        async for item in async_gen:
            result.append(item)
        return result
    return collect()
```

## Performance Tips

1. **Connection Pooling**: aiohttp handles this automatically
2. **Keep-Alive**: Set appropriate keep-alive to avoid model reload
3. **Client Reuse**: One client for multiple agents
4. **Streaming**: Use streaming for better perceived performance
5. **Distributed Mode**: Use SOLLOL for horizontal scaling

## Support

Need help integrating? Check:
- Examples in `agents/examples/`
- Architecture docs in `ARCHITECTURE.md`
- Source code (well-documented)
