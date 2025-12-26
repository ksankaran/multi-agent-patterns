# Critique and Refine Pattern
# Full code for: https://medium.com/@ksankaran/stop-overloading-your-ai-agent
#
# One agent creates, another critiques, and they iterate until quality
# is acceptable. This mimics how professional editing actually works.

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# =============================================================================
# STATE
# =============================================================================

class CritiqueState(TypedDict):
    task: str               # The task to accomplish
    current_work: str       # Current version of the work
    critique: str           # Feedback from the critic
    revision_count: int     # Number of revisions made
    max_revisions: int      # Maximum allowed revisions
    is_approved: bool       # Whether the critic approved
    final_work: str         # Final approved version


# =============================================================================
# CREATOR
# =============================================================================

def creator(state: CritiqueState) -> dict:
    """
    Creates initial work or revises based on feedback.
    
    The creator doesn't need to be perfect on the first try.
    It just needs to get something on the page that the critic
    can work with.
    """
    revision_count = state.get("revision_count", 0)
    
    if revision_count == 0:
        # Initial creation
        prompt = f"""Create a response for this task:

TASK: {state['task']}

Do your best work. Be thorough but concise."""
        
        print("[Creator] Generating initial draft...")
    else:
        # Revision based on feedback
        prompt = f"""Revise your work based on this critique:

TASK: {state['task']}

YOUR PREVIOUS WORK:
{state['current_work']}

CRITIC'S FEEDBACK:
{state['critique']}

Address the specific concerns raised while keeping what works well.
Don't over-correct - fix the issues mentioned, not everything."""
        
        print(f"[Creator] Revision {revision_count} based on feedback...")
    
    response = llm.invoke(prompt)
    
    return {
        "current_work": response.content,
        "revision_count": revision_count + 1
    }


# =============================================================================
# CRITIC
# =============================================================================

def critic(state: CritiqueState) -> dict:
    """
    Evaluates work and provides specific, actionable feedback.
    
    The critic is harder to satisfy than the creator is to produce.
    This asymmetry is the engine of improvement.
    """
    prompt = f"""Critically evaluate this work:

TASK: {state['task']}

WORK TO EVALUATE:
{state['current_work']}

Evaluate on:
1. Completeness - Does it fully address the task?
2. Accuracy - Is the information correct?
3. Clarity - Is it easy to understand?
4. Quality - Is it well-written/well-structured?

If the work meets a high quality bar (8/10 or better), respond with:
APPROVED

If the work needs improvement, respond with:
REVISE: [specific, actionable feedback]

Be specific. Don't say "make it better" - say exactly what needs to change and why.
But also be reasonable - don't demand perfection."""
    
    response = llm.invoke(prompt)
    content = response.content
    
    is_approved = "APPROVED" in content.upper() and "REVISE" not in content.upper()
    
    if is_approved:
        print("[Critic] APPROVED")
    else:
        print(f"[Critic] Revision requested")
    
    return {
        "critique": content,
        "is_approved": is_approved
    }


# =============================================================================
# FINALIZER
# =============================================================================

def finalizer(state: CritiqueState) -> dict:
    """Packages the approved work as final output."""
    return {"final_work": state["current_work"]}


# =============================================================================
# ROUTING
# =============================================================================

def should_continue(state: CritiqueState) -> Literal["revise", "finalize"]:
    """
    Decides whether to continue revising or finalize.
    
    Stops if:
    - Work is approved by critic
    - Maximum revisions reached (prevents infinite loops)
    """
    if state.get("is_approved", False):
        print(f"[Router] Work approved after {state['revision_count']} revision(s) -> Finalize")
        return "finalize"
    
    if state.get("revision_count", 0) >= state.get("max_revisions", 3):
        print(f"[Router] Max revisions reached -> Finalize (best effort)")
        return "finalize"
    
    print(f"[Router] Revision {state['revision_count']} not approved -> Revise")
    return "revise"


# =============================================================================
# BUILD WORKFLOW
# =============================================================================

workflow = StateGraph(CritiqueState)

# Add nodes
workflow.add_node("creator", creator)
workflow.add_node("critic", critic)
workflow.add_node("finalizer", finalizer)

# Creator produces work, critic evaluates
workflow.add_edge(START, "creator")
workflow.add_edge("creator", "critic")

# Critic decides: revise or finalize
workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "revise": "creator",
        "finalize": "finalizer"
    }
)

workflow.add_edge("finalizer", END)

app = workflow.compile()


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    # Test with a writing task that benefits from iteration
    task = """
    Write a product announcement email for our new AI-powered code review tool.
    
    Key features:
    - Catches bugs before they reach production
    - Learns your team's coding standards
    - Integrates with GitHub, GitLab, and Bitbucket
    - Free tier available for small teams
    
    Target audience: Engineering managers at mid-size companies (100-500 employees)
    
    Tone: Professional but not stuffy, confident but not arrogant
    """
    
    print("=" * 60)
    print("TASK")
    print("=" * 60)
    print(task)
    print("=" * 60)
    
    result = app.invoke({
        "task": task,
        "current_work": "",
        "critique": "",
        "revision_count": 0,
        "max_revisions": 3,
        "is_approved": False,
        "final_work": ""
    })
    
    print("\n" + "=" * 60)
    print(f"FINAL WORK (after {result['revision_count']} revision(s))")
    print("=" * 60)
    print(result["final_work"])