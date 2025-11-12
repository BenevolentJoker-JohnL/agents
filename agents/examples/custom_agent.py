"""
Custom Agent Example
====================

Demonstrates creating a custom agent with specialized behavior.
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, Iterable, Union
from agents import LocalOllamaClient, BaseAgent, ChatResponse, ChatMessage


class SQLAgent(BaseAgent):
    """Custom agent specialized for SQL query generation."""

    system_prompt = """You are an expert SQL developer. Generate SQL queries based on
natural language descriptions. Follow these rules:
1. Use PostgreSQL syntax
2. Include comments explaining complex queries
3. Use proper formatting and indentation
4. Only output the SQL query, nothing else"""


class DataAnalystAgent(BaseAgent):
    """Custom agent that acts as a data analyst."""

    system_prompt = """You are a professional data analyst. When analyzing data:
1. Identify patterns and trends
2. Calculate relevant statistics
3. Provide actionable insights
4. Present findings in a clear, structured format"""

    async def run(
        self,
        messages: Iterable[Union[ChatMessage, Dict[str, Any]]],
        **options: Any,
    ) -> AsyncGenerator[ChatResponse, None]:
        """Override run to add custom preprocessing."""
        message_list = list(messages)

        # Add analysis framework to user messages
        for msg in message_list:
            if isinstance(msg, dict) and msg.get("role") == "user":
                original_content = msg["content"]
                msg["content"] = f"""Analyze the following data:

{original_content}

Provide:
1. Summary statistics
2. Key patterns
3. Insights and recommendations
"""

        # Call parent run with modified messages
        async for chunk in super().run(message_list, **options):
            yield chunk


async def main():
    # Create client
    client = LocalOllamaClient(
        model_name="llama3:latest",
        api_base="http://localhost:11434"
    )

    print("=== Custom Agent Example ===\n")

    # Example 1: SQL Agent
    print("Example 1: SQL Agent")
    print("-" * 50)
    sql_agent = SQLAgent(client)

    sql_request = """Create a query to find the top 10 customers by total purchase amount
from orders and customers tables, including customer name and email."""

    sql_query = sql_agent.get_full_response([
        {"role": "user", "content": sql_request}
    ])
    print(sql_query)

    # Example 2: Data Analyst Agent
    print("\n\nExample 2: Data Analyst Agent")
    print("-" * 50)
    analyst = DataAnalystAgent(client)

    data = """Sales data for Q1 2024:
January: $45,000
February: $52,000
March: $61,000

Customer acquisition:
January: 120 customers
February: 145 customers
March: 178 customers"""

    async for chunk in analyst.run([{"role": "user", "content": data}]):
        print(chunk.content, end="", flush=True)

    print("\n")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
