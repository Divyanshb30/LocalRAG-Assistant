from llama_cpp import Llama
from typing import Dict, Any
import json


class RAGAgent:
    """AI Agent with intent detection and tool routing"""
    
    def __init__(self, llm: Llama, tools: Dict):
        self.llm = llm
        self.tools = tools
    
    def detect_intent(self, query: str) -> str:
        """Detect which tool to use based on query"""
        query_lower = query.lower()
        
        # Priority-based routing (order matters!)
        
        # 1. Task breakdown (highest priority for action words)
        if any(word in query_lower for word in ["task", "tasks", "breakdown", "tickets", "implement", "build", "create tasks", "generate tasks"]):
            return "task_breakdown"
        
        # 2. Summarizer (for summary requests)
        if any(word in query_lower for word in ["summary", "summarize", "overview", "tl;dr", "brief", "give me a summary"]):
            return "summarizer"
        
        # 3. CSV Analyzer (only for explicit data/analytics queries with numeric operations)
        if any(word in query_lower for word in ["average", "sum", "total", "maximum", "minimum", "calculate", "count rows", "statistics"]) and \
        any(word in query_lower for word in ["csv", "table", "data", "numbers", "values"]):
            return "csv_analyzer"
        
        # 4. Default: Document search (for comparisons, questions, general queries)
        return "document_search"

    
    def execute(self, query: str, csv_path: str = None) -> Dict[str, Any]:
        """Execute query through appropriate tool"""
        
        # Detect intent
        tool_name = self.detect_intent(query)
        
        # Override if CSV path provided
        if csv_path:
            tool_name = "csv_analyzer"
        
        # Execute tool
        tool = self.tools.get(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}
        
        # Call appropriate tool
        if tool_name == "csv_analyzer" and csv_path:
            tool_output = tool.execute(csv_path, query)
        elif tool_name in ["document_search", "summarizer"]:
            tool_output = tool.execute(query)
        elif tool_name == "task_breakdown":
            tool_output = tool.execute(query)
        else:
            tool_output = {"success": False, "error": "Unknown tool"}
        
        # Generate natural language response with LLM
        if tool_output.get("success"):
            answer = self._generate_answer(query, tool_output, tool_name)
        else:
            answer = f"Error: {tool_output.get('error', 'Unknown error')}"
        
        return {
            "answer": answer,
            "tool_used": tool_name,
            "raw_output": tool_output,
            "success": tool_output.get("success", False)
        }
    
    def _generate_answer(self, query: str, tool_output: Dict, tool_name: str) -> str:
        """Generate natural language answer from tool output"""
        
        # Create prompt based on tool
        if tool_name == "document_search":
            context = tool_output.get("context", "")
            context = context[:2000]
            prompt = f"""Based on the following document excerpts, answer the question concisely.

Question: {query}

Document Excerpts:
{context}

Answer:"""
        
        elif tool_name == "summarizer":
            text = tool_output.get("full_text", "")[:2000]
            prompt = f"""Summarize the following text in 3-5 clear sentences:

{text}

Summary:"""
        
        elif tool_name == "task_breakdown":
            tasks = tool_output.get("tasks", [])
            task_list = "\n".join([f"{i+1}. {t['title']}" for i, t in enumerate(tasks)])
            prompt = f"""The following tasks have been created:

{task_list}

Briefly explain these tasks in 2-3 sentences.

Explanation:"""
        
        else:  # csv_analyzer
            results = str(tool_output.get("results", {}))[:1000]
            prompt = f"""Based on this data analysis: {results}

Answer the question: {query}

Answer:"""
        
        # Generate with LLM - INCREASED TOKEN LIMIT
        try:
            response = self.llm(
                prompt,
                max_tokens=512,  # INCREASED from 200 to 512
                temperature=0.7,
                stop=["Question:", "Query:", "\n\n\n"]
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            # Fallback to template
            print(f"LLM generation failed: {e}")
            return self._fallback_answer(tool_output, tool_name)
    
    def _fallback_answer(self, tool_output: Dict, tool_name: str) -> str:
        """Fallback template-based answer"""
        if tool_name == "document_search":
            chunks = tool_output.get("chunks", [])
            if chunks:
                return "Based on the documents: " + chunks[0].get("text", "No information found.")[:300]
            return "No relevant information found."
        elif tool_name == "summarizer":
            highlights = tool_output.get("highlights", [])
            if highlights:
                return "Summary: " + "; ".join(highlights[:3])
            return "No highlights available for summary."
        elif tool_name == "task_breakdown":
            count = tool_output.get("total_tasks", 0)
            tasks = tool_output.get("tasks", [])
            if tasks:
                task_titles = ", ".join([t['title'] for t in tasks[:3]])
                return f"Created {count} tasks including: {task_titles}"
            return f"Created {count} tasks for implementation."
        else:
            return f"Analysis complete. Results: {str(tool_output.get('results', {}))[:200]}"
