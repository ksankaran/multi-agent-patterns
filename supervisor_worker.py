# Supervisor-Worker Pattern
# Full code for: https://medium.com/@v31u/stop-overloading-your-ai-agent-build-a-team-instead-256fb0097eb7
#
# A supervisor agent analyzes requests, delegates to specialist workers,
# and synthesizes results. This mirrors how human organizations work.

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# =============================================================================
# STATE
# =============================================================================

class WritingState(TypedDict):
    request: str          # User's writing request
    task_type: str        # Category determined by supervisor
    draft: str            # Output from specialist worker
    final_output: str     # Polished final output


# =============================================================================
# SUPERVISOR
# =============================================================================

def supervisor(state: WritingState) -> dict:
    """
    Analyzes the request and decides which specialist should handle it.
    
    The supervisor is an LLM too - it uses reasoning to make routing
    decisions, not hardcoded rules. This makes the system adaptive.
    """
    prompt = f"""Analyze this writing request and categorize it:

Request: {state['request']}

Categories:
- email: Professional correspondence, formal messages, business communication
- blog: Articles, posts, educational content, thought pieces
- summary: Condensing information, briefs, executive summaries

Reply with just the category name (email, blog, or summary)."""
    
    response = llm.invoke(prompt)
    task_type = response.content.strip().lower()
    
    # Normalize the response to handle variations
    if "email" in task_type:
        task_type = "email"
    elif "blog" in task_type:
        task_type = "blog"
    else:
        task_type = "summary"
    
    print(f"[Supervisor] Routing to: {task_type}")
    return {"task_type": task_type}


# =============================================================================
# SPECIALIST WORKERS
# =============================================================================

def email_writer(state: WritingState) -> dict:
    """Specialist for professional emails."""
    prompt = f"""Write a professional email for this request:

{state['request']}

Use proper email format with:
- Clear subject line
- Professional greeting
- Concise body paragraphs
- Appropriate sign-off"""
    
    response = llm.invoke(prompt)
    print("[Email Writer] Draft complete")
    return {"draft": response.content}


def blog_writer(state: WritingState) -> dict:
    """Specialist for blog content."""
    prompt = f"""Write an engaging blog post for this request:

{state['request']}

Include:
- Attention-grabbing headline
- Hook in the opening paragraph
- Clear sections with subheadings
- Actionable takeaways
- Conversational but authoritative tone"""
    
    response = llm.invoke(prompt)
    print("[Blog Writer] Draft complete")
    return {"draft": response.content}


def summary_writer(state: WritingState) -> dict:
    """Specialist for summaries and briefs."""
    prompt = f"""Write a clear, concise summary for this request:

{state['request']}

Guidelines:
- Lead with the most important information
- Use bullet points for key facts
- Keep it scannable
- No fluff or filler"""
    
    response = llm.invoke(prompt)
    print("[Summary Writer] Draft complete")
    return {"draft": response.content}


# =============================================================================
# FINALIZER
# =============================================================================

def finalizer(state: WritingState) -> dict:
    """Polishes the draft into final output."""
    prompt = f"""Review and polish this draft. Fix any issues with:
- Grammar and spelling
- Clarity and flow
- Tone consistency

Draft:
{state['draft']}

Return the polished version."""
    
    response = llm.invoke(prompt)
    print("[Finalizer] Output polished")
    return {"final_output": response.content}


# =============================================================================
# ROUTING
# =============================================================================

def route_to_worker(state: WritingState) -> Literal["email", "blog", "summary"]:
    """Routes to the appropriate specialist based on supervisor's decision."""
    return state["task_type"]


# =============================================================================
# BUILD WORKFLOW
# =============================================================================

workflow = StateGraph(WritingState)

# Add nodes
workflow.add_node("supervisor", supervisor)
workflow.add_node("email", email_writer)
workflow.add_node("blog", blog_writer)
workflow.add_node("summary", summary_writer)
workflow.add_node("finalizer", finalizer)

# Supervisor analyzes first
workflow.add_edge(START, "supervisor")

# Route to appropriate specialist
workflow.add_conditional_edges(
    "supervisor",
    route_to_worker,
    {
        "email": "email",
        "blog": "blog",
        "summary": "summary"
    }
)

# All specialists go to finalizer
workflow.add_edge("email", "finalizer")
workflow.add_edge("blog", "finalizer")
workflow.add_edge("summary", "finalizer")

workflow.add_edge("finalizer", END)

app = workflow.compile()


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    # Test with different request types
    test_requests = [
        "Write a message to my team announcing that we're switching to a new project management tool next month",
        "Create content about why startups should invest in AI automation early",
        "Condense our Q3 results: revenue up 23%, new customers 145, churn down to 2.1%, launched 3 new features"
    ]
    
    for request in test_requests:
        print("\n" + "=" * 60)
        print(f"REQUEST: {request[:50]}...")
        print("=" * 60)
        
        result = app.invoke({
            "request": request,
            "task_type": "",
            "draft": "",
            "final_output": ""
        })
        
        print(f"\nFINAL OUTPUT:\n{result['final_output']}")