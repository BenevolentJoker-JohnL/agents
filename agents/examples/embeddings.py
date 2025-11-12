"""
Embeddings Example
==================

Demonstrates generating embeddings for semantic similarity.
"""

import asyncio
import numpy as np
from agents import LocalOllamaClient, EmbeddingAgent


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


async def main():
    # Create a local client with an embedding model
    client = LocalOllamaClient(
        model_name="llama3:latest",
        api_base="http://localhost:11434",
        embed_model="mxbai-embed-large"  # Specialized embedding model
    )

    # Create an embedding agent
    agent = EmbeddingAgent(client)

    print("=== Embeddings Example ===\n")

    # Generate embeddings for multiple texts
    texts = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "The dog played in the park.",
        "Python is a programming language.",
    ]

    embeddings = []
    for text in texts:
        messages = [{"role": "user", "content": text}]
        responses = agent.run_sync(messages)
        embedding = responses[0].metadata["embedding"]
        embeddings.append(embedding)
        print(f"Generated embedding for: '{text}'")
        print(f"  Dimension: {len(embedding)}\n")

    # Calculate similarities
    print("\nSemantic Similarities:")
    print("-" * 60)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            print(f"'{texts[i]}' <-> '{texts[j]}'")
            print(f"  Similarity: {similarity:.4f}\n")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
