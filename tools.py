import os
import json
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

class DocumentSearchTool:
    """Tool 1: RAG-based document search and QA"""
    
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
    
    def execute(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search documents and generate answer"""
        try:
            results = self.rag.retrieve(query, top_k=top_k)
            
            # Extract chunks and create context
            chunks = []
            context = ""
            for i, result in enumerate(results):
                chunks.append({
                    "chunk_id": i,
                    "text": result["text"],
                    "score": result["score"]
                })
                context += f"\n[Chunk {i+1}]: {result['text']}\n"
            
            return {
                "success": True,
                "chunks": chunks,
                "context": context,
                "query": query
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class SummarizerTool:
    """Tool 2: Summarize retrieved content"""
    
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
    
    def execute(self, query: str, style: str = "brief") -> Dict[str, Any]:
        """Summarize document content"""
        try:
            # Retrieve relevant chunks
            results = self.rag.retrieve(query, top_k=5)
            
            # Combine text
            combined_text = "\n\n".join([r["text"] for r in results])
            
            # Extract key points (simple keyword extraction for demo)
            highlights = self._extract_highlights(combined_text)
            
            return {
                "success": True,
                "full_text": combined_text,
                "highlights": highlights,
                "chunk_count": len(results),
                "style": style
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_highlights(self, text: str) -> List[str]:
        """Extract bullet points from text"""
        lines = text.split('\n')
        highlights = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                highlights.append(line.lstrip('-•* '))
            elif len(line) > 20 and len(highlights) < 5:
                # Add first sentence as highlight if no bullets
                highlights.append(line.split('.')[0] + '.')
        
        return highlights[:5]


class TaskBreakdownTool:
    """Tool 3: Break requirements into developer tasks"""
    
    def execute(self, change_description: str, project_context: str = "") -> Dict[str, Any]:
        """Generate structured tasks from requirements"""
        try:
            # Simple task generation logic (you'll enhance with LLM later)
            tasks = self._generate_tasks(change_description)
            
            return {
                "success": True,
                "tasks": tasks,
                "total_tasks": len(tasks),
                "change_description": change_description
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_tasks(self, description: str) -> List[Dict]:
        """Generate task breakdown (template-based for now)"""
        # Keywords for task detection
        keywords = {
            "api": "Backend API Development",
            "ui": "Frontend UI Development",
            "database": "Database Schema Design",
            "test": "Testing & QA",
            "deploy": "Deployment & DevOps"
        }
        
        tasks = []
        task_id = 1
        
        for keyword, component in keywords.items():
            if keyword in description.lower():
                tasks.append({
                    "task_id": f"TASK-{task_id}",
                    "title": f"Implement {keyword.upper()} changes",
                    "description": f"Implement changes related to {keyword}",
                    "component": component,
                    "acceptance_criteria": [
                        f"{keyword.capitalize()} functionality working",
                        "Code reviewed and tested",
                        "Documentation updated"
                    ],
                    "estimated_effort": "3-5 hours"
                })
                task_id += 1
        
        # Default task if no keywords matched
        if not tasks:
            tasks.append({
                "task_id": "TASK-1",
                "title": "Implement requested changes",
                "description": description,
                "component": "General Development",
                "acceptance_criteria": ["Changes implemented", "Tested and verified"],
                "estimated_effort": "2-4 hours"
            })
        
        return tasks


class CSVAnalyzerTool:
    """Tool 4: Analyze CSV files with pandas"""
    
    def execute(self, csv_file_path: str, question: str = "") -> Dict[str, Any]:
        """Analyze CSV and answer questions"""
        try:
            if not os.path.exists(csv_file_path):
                return {"success": False, "error": f"File not found: {csv_file_path}"}
            
            # Load CSV
            df = pd.read_csv(csv_file_path)
            
            # Generate metadata
            metadata = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "data_types": df.dtypes.astype(str).to_dict()
            }
            
            # Analyze based on question
            results = self._analyze_data(df, question)
            
            return {
                "success": True,
                "metadata": metadata,
                "results": results,
                "preview": df.head(5).to_dict('records'),
                "question": question
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _analyze_data(self, df: pd.DataFrame, question: str) -> Dict:
        """Perform analysis based on question"""
        question_lower = question.lower()
        
        results = {}
        
        # Basic statistics
        if "average" in question_lower or "mean" in question_lower:
            numeric_cols = df.select_dtypes(include=['number']).columns
            results["averages"] = df[numeric_cols].mean().to_dict()
        
        if "sum" in question_lower or "total" in question_lower:
            numeric_cols = df.select_dtypes(include=['number']).columns
            results["totals"] = df[numeric_cols].sum().to_dict()
        
        if "max" in question_lower or "maximum" in question_lower:
            numeric_cols = df.select_dtypes(include=['number']).columns
            results["maximums"] = df[numeric_cols].max().to_dict()
        
        if "min" in question_lower or "minimum" in question_lower:
            numeric_cols = df.select_dtypes(include=['number']).columns
            results["minimums"] = df[numeric_cols].min().to_dict()
        
        # Default: describe
        if not results:
            results["description"] = df.describe().to_dict()
        
        return results


# Tool registry
def get_tools(rag_pipeline):
    """Initialize all tools with RAG pipeline"""
    return {
        "document_search": DocumentSearchTool(rag_pipeline),
        "summarizer": SummarizerTool(rag_pipeline),
        "task_breakdown": TaskBreakdownTool(),
        "csv_analyzer": CSVAnalyzerTool()
    }
