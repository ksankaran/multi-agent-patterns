# Ensemble Pattern (Parallel Solvers)
# Full code for: https://medium.com/@ksankaran/stop-overloading-your-ai-agent
#
# Multiple agents tackle the same problem with different thinking styles,
# then a merger synthesizes their solutions. Like getting second opinions
# from multiple doctors.

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# =============================================================================
# STATE
# =============================================================================

class EnsembleState(TypedDict):
    problem: str                # The problem to solve
    solution_creative: str      # Creative solver's approach
    solution_analytical: str    # Analytical solver's approach
    solution_practical: str     # Practical solver's approach
    merged_solution: str        # Final synthesized solution


# =============================================================================
# PARALLEL SOLVERS
# Each solver approaches the problem with a different thinking style.
# They run in parallel and are unaware of each other's solutions.
# =============================================================================

def creative_solver(state: EnsembleState) -> dict:
    """
    Thinks outside the box.
    
    Prioritizes novel, unconventional ideas over safe solutions.
    May suggest things that seem impractical but spark insight.
    """
    prompt = f"""Solve this problem CREATIVELY. Think outside the box.

PROBLEM: {state['problem']}

Your approach should:
- Challenge conventional assumptions
- Propose unconventional solutions others might dismiss
- Prioritize innovation over safety
- Use analogies from unexpected domains

Be bold. The practical constraints will be handled by others."""
    
    response = llm.invoke(prompt)
    print("[Creative Solver] Solution ready")
    return {"solution_creative": response.content}


def analytical_solver(state: EnsembleState) -> dict:
    """
    Uses data and logic.
    
    Breaks down the problem systematically, considers trade-offs,
    and provides clear reasoning for recommendations.
    """
    prompt = f"""Solve this problem ANALYTICALLY. Use logic and structure.

PROBLEM: {state['problem']}

Your approach should:
- Break the problem into components
- Identify key variables and constraints
- Consider trade-offs explicitly
- Provide clear reasoning for each recommendation
- Acknowledge uncertainties and assumptions

Be rigorous. Show your work."""
    
    response = llm.invoke(prompt)
    print("[Analytical Solver] Solution ready")
    return {"solution_analytical": response.content}


def practical_solver(state: EnsembleState) -> dict:
    """
    Focuses on what's actionable.
    
    Prioritizes implementability, quick wins, and realistic constraints.
    Less interested in elegance, more interested in getting things done.
    """
    prompt = f"""Solve this problem PRACTICALLY. Focus on what's actionable.

PROBLEM: {state['problem']}

Your approach should:
- Prioritize solutions that can be implemented quickly
- Consider resource constraints (time, money, people)
- Identify the simplest path to results
- Break recommendations into concrete next steps
- Anticipate implementation challenges

Be realistic. Perfect is the enemy of good."""
    
    response = llm.invoke(prompt)
    print("[Practical Solver] Solution ready")
    return {"solution_practical": response.content}


# =============================================================================
# MERGER
# =============================================================================

def solution_merger(state: EnsembleState) -> dict:
    """
    Synthesizes all three approaches into one coherent solution.
    
    The merger's job is to find the best insights from each approach
    and combine them into something better than any individual solution.
    """
    prompt = f"""You have three different approaches to this problem:

PROBLEM: {state['problem']}

CREATIVE APPROACH:
{state['solution_creative']}

ANALYTICAL APPROACH:
{state['solution_analytical']}

PRACTICAL APPROACH:
{state['solution_practical']}

Synthesize these into ONE comprehensive solution that:
1. Takes the most innovative insights from the creative approach
2. Incorporates the rigorous analysis and trade-off thinking from the analytical approach
3. Grounds everything in the actionable, realistic framing of the practical approach

Don't just summarize each approach. Create something new that's better than any individual solution.

Structure your response as:
- RECOMMENDED APPROACH: The synthesized solution
- KEY INSIGHTS COMBINED: What each perspective contributed
- IMPLEMENTATION PRIORITY: What to do first, second, third"""
    
    response = llm.invoke(prompt)
    print("[Merger] Synthesis complete")
    return {"merged_solution": response.content}


# =============================================================================
# BUILD WORKFLOW
# =============================================================================

workflow = StateGraph(EnsembleState)

# Add nodes
workflow.add_node("creative", creative_solver)
workflow.add_node("analytical", analytical_solver)
workflow.add_node("practical", practical_solver)
workflow.add_node("merger", solution_merger)

# All three solvers start in parallel from START
workflow.add_edge(START, "creative")
workflow.add_edge(START, "analytical")
workflow.add_edge(START, "practical")

# All three feed into the merger
# LangGraph waits for all incoming edges before executing a node
workflow.add_edge("creative", "merger")
workflow.add_edge("analytical", "merger")
workflow.add_edge("practical", "merger")

workflow.add_edge("merger", END)

app = workflow.compile()


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    # Test with a complex business problem
    problem = """
    Our B2B SaaS startup has strong product-market fit (NPS 65, low churn) but 
    is struggling to scale sales. We have 2 sales reps closing $50k/month each, 
    but adding more reps hasn't proportionally increased revenue. Our sales cycle 
    is 45 days and requires significant pre-sales engineering support.
    
    How should we scale revenue from $1.2M to $5M ARR in the next 18 months?
    """
    
    print("=" * 60)
    print("PROBLEM")
    print("=" * 60)
    print(problem)
    
    result = app.invoke({
        "problem": problem,
        "solution_creative": "",
        "solution_analytical": "",
        "solution_practical": "",
        "merged_solution": ""
    })
    
    print("\n" + "=" * 60)
    print("MERGED SOLUTION")
    print("=" * 60)
    print(result["merged_solution"])