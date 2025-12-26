# Multi-Agent Patterns

Code examples for the blog post: **[Stop Overloading Your AI Agent — Build a Team Instead](https://medium.com/@v31u/stop-overloading-your-ai-agent-build-a-team-instead-256fb0097eb7)**

Four proven patterns for building multi-agent systems where specialized AI agents collaborate to solve problems too complex for any single agent.

## Patterns

| Pattern | File | Description |
|---------|------|-------------|
| **Supervisor-Worker** | `supervisor_worker.py` | A supervisor routes requests to specialist workers |
| **Debate** | `debate_pattern.py` | Agents argue opposing positions, judge synthesizes |
| **Ensemble** | `ensemble_pattern.py` | Parallel solvers with different thinking styles |
| **Critique & Refine** | `critique_refine.py` | Creator and critic iterate until quality bar is met |

## Setup

1. Clone the repo:
```bash
git clone https://github.com/ksankaran/multi-agent-patterns.git
cd multi-agent-patterns
```

2. Create and activate the virtual environment:
```bash
uv venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv sync
```

4. Set up your OpenAI API key:
```bash
cp .env.example .env
```

Edit `.env` and replace `sk-your-openai-api-key-here` with your actual OpenAI API key. You can get one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

5. Run any pattern:
```bash
python supervisor_worker.py
python debate_pattern.py
python ensemble_pattern.py
python critique_refine.py
```

## When to Use Each Pattern

| Pattern | Best For |
|---------|----------|
| **Supervisor-Worker** | Clear task categories, routing, delegation |
| **Debate** | Complex decisions, surfacing trade-offs, stress-testing ideas |
| **Ensemble** | Important decisions, reducing bias, diverse perspectives |
| **Critique & Refine** | Quality-sensitive work, iterative improvement |

## License

MIT