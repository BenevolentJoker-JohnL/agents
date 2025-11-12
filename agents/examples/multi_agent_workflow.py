"""
Multi-Agent Workflow Example
============================

Demonstrates coordinating multiple specialized agents to solve a complex task.
"""

import asyncio
from agents import (
    LocalOllamaClient,
    ResearchAgent,
    ReasoningAgent,
    SummarizationAgent,
)


async def research_and_summarize(topic: str):
    """
    Multi-stage workflow:
    1. Research agent gathers information
    2. Reasoning agent analyzes the information
    3. Summarization agent creates a concise summary
    """
    # Create a shared client
    client = LocalOllamaClient(
        model_name="llama3:latest",
        api_base="http://localhost:11434"
    )

    print(f"=== Multi-Agent Workflow: {topic} ===\n")

    # Stage 1: Research
    print("Stage 1: Research")
    print("-" * 50)
    researcher = ResearchAgent(client)
    research_messages = [
        {"role": "user", "content": f"Research the topic: {topic}"}
    ]
    research_result = researcher.get_full_response(research_messages)
    print(research_result[:500] + "...\n")  # Print first 500 chars

    # Stage 2: Analysis
    print("\nStage 2: Analysis")
    print("-" * 50)
    analyzer = ReasoningAgent(client)
    analysis_messages = [
        {
            "role": "user",
            "content": f"Analyze this research and identify key insights:\n\n{research_result}"
        }
    ]
    analysis_result = analyzer.get_full_response(analysis_messages)
    print(analysis_result[:500] + "...\n")

    # Stage 3: Summarization
    print("\nStage 3: Summary")
    print("-" * 50)
    summarizer = SummarizationAgent(client)
    summary_messages = [
        {
            "role": "user",
            "content": f"Create a concise summary of these insights:\n\n{analysis_result}"
        }
    ]
    summary = summarizer.get_full_response(summary_messages)
    print(summary)

    await client.close()
    return summary


async def main():
    topic = "The impact of quantum computing on cryptography"
    await research_and_summarize(topic)


if __name__ == "__main__":
    asyncio.run(main())
