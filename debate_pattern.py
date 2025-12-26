# Debate Pattern
# Full code for: https://medium.com/@ksankaran/stop-overloading-your-ai-agent
#
# Two agents argue opposing positions, and a judge synthesizes their
# arguments into a balanced conclusion. This surfaces trade-offs that
# a single agent might gloss over.

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import operator

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# =============================================================================
# STATE
# =============================================================================

class DebateState(TypedDict):
    topic: str                                              # The debate topic
    pro_arguments: Annotated[list[str], operator.add]       # Arguments in favor
    con_arguments: Annotated[list[str], operator.add]       # Arguments against
    current_round: int                                      # Current debate round
    max_rounds: int                                         # Maximum rounds
    synthesis: str                                          # Judge's final synthesis


# =============================================================================
# DEBATERS
# =============================================================================

def pro_debater(state: DebateState) -> dict:
    """
    Argues in favor of the topic.
    
    Responds to opposing arguments to create genuine intellectual exchange,
    not just isolated talking points.
    """
    round_num = state.get("current_round", 1)
    existing_con = state.get("con_arguments", [])
    
    if existing_con:
        context = f"The opposition has argued: {existing_con[-1]}\n\nRespond to their points while advancing your position."
    else:
        context = "You are opening the debate. Make your strongest opening argument."
    
    prompt = f"""You are arguing IN FAVOR of: {state['topic']}

Round {round_num} of {state.get('max_rounds', 3)}.

{context}

Provide ONE compelling argument. Be concise (2-3 paragraphs) but persuasive.
Use evidence and logic, not just assertions."""
    
    response = llm.invoke(prompt)
    print(f"[Pro] Round {round_num} argument delivered")
    return {"pro_arguments": [f"[Round {round_num}] {response.content}"]}


def con_debater(state: DebateState) -> dict:
    """
    Argues against the topic.
    
    Must engage with the pro arguments, not just present independent points.
    """
    round_num = state.get("current_round", 1)
    existing_pro = state.get("pro_arguments", [])
    
    prompt = f"""You are arguing AGAINST: {state['topic']}

Round {round_num} of {state.get('max_rounds', 3)}.

The proponent has argued: {existing_pro[-1] if existing_pro else 'Nothing yet'}

Counter their argument while advancing your position. Find weaknesses in their reasoning.

Provide ONE compelling counter-argument. Be concise (2-3 paragraphs) but persuasive.
Use evidence and logic, not just assertions."""
    
    response = llm.invoke(prompt)
    print(f"[Con] Round {round_num} argument delivered")
    return {"con_arguments": [f"[Round {round_num}] {response.content}"]}


# =============================================================================
# ROUND COORDINATOR
# =============================================================================

def round_coordinator(state: DebateState) -> dict:
    """Advances the round counter."""
    current = state.get("current_round", 1)
    return {"current_round": current + 1}


def should_continue(state: DebateState) -> Literal["continue", "judge"]:
    """Decides whether to continue debating or move to judgment."""
    current_round = state.get("current_round", 1)
    max_rounds = state.get("max_rounds", 3)
    
    if current_round >= max_rounds:
        print(f"[Coordinator] Debate complete after {current_round} rounds -> Judge")
        return "judge"
    
    print(f"[Coordinator] Round {current_round} complete -> Continue to round {current_round + 1}")
    return "continue"


# =============================================================================
# JUDGE
# =============================================================================

def judge(state: DebateState) -> dict:
    """
    Synthesizes the debate into a balanced conclusion.
    
    The judge is impartial - they identify the strongest points from
    each side and provide a nuanced conclusion.
    """
    pro_args = "\n\n".join(state["pro_arguments"])
    con_args = "\n\n".join(state["con_arguments"])
    
    prompt = f"""As an impartial judge, synthesize this debate:

TOPIC: {state['topic']}

ARGUMENTS IN FAVOR:
{pro_args}

ARGUMENTS AGAINST:
{con_args}

Provide:
1. The single strongest point from the PRO side and why it's compelling
2. The single strongest point from the CON side and why it's compelling
3. Where the two sides actually agree (if anywhere)
4. A balanced conclusion that acknowledges the complexity
5. What additional information would help resolve this debate

Be fair to both sides. Avoid false balance - if one side genuinely has stronger arguments, say so."""
    
    response = llm.invoke(prompt)
    print("[Judge] Synthesis complete")
    return {"synthesis": response.content}


# =============================================================================
# BUILD WORKFLOW
# =============================================================================

workflow = StateGraph(DebateState)

# Add nodes
workflow.add_node("pro", pro_debater)
workflow.add_node("con", con_debater)
workflow.add_node("coordinator", round_coordinator)
workflow.add_node("judge", judge)

# Debate flow: pro argues, con responds, coordinator checks if we continue
workflow.add_edge(START, "pro")
workflow.add_edge("pro", "con")
workflow.add_edge("con", "coordinator")

# Either continue to next round or move to judge
workflow.add_conditional_edges(
    "coordinator",
    should_continue,
    {
        "continue": "pro",
        "judge": "judge"
    }
)

workflow.add_edge("judge", END)

app = workflow.compile()


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    # Test with a debatable topic
    topic = "Remote work should be the default for knowledge workers"
    
    print("=" * 60)
    print(f"DEBATE TOPIC: {topic}")
    print("=" * 60)
    
    result = app.invoke({
        "topic": topic,
        "pro_arguments": [],
        "con_arguments": [],
        "current_round": 1,
        "max_rounds": 2,
        "synthesis": ""
    })
    
    print("\n" + "=" * 60)
    print("JUDGE'S SYNTHESIS")
    print("=" * 60)
    print(result["synthesis"])