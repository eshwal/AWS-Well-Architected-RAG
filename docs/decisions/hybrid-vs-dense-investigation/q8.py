# check_q8_const_sensitivity.py
# As we are getting correct doc but not at correct position
import asyncio
from src.service.rag import query_compliance_platform

async def check():
    query = "How can I implement model cascading for cost-performance optimization in Amazon Bedrock agents, and what are the key components required?"
    for const in [40, 60, 100, 150, 200, 300]:
        res = await query_compliance_platform(query, flags={"search_mode": "hybrid", "rrf_const": const})
        top_source = res["referenced_metadata"][0]["source"]
        print(f"rrf_const={const:<4} → rank0 = {top_source}")

asyncio.run(check())