---
title: Drug Repurposing Environment
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - biology
---

# Drug Repurposing Environment

## Description & Motivation
This is a biomedical reinforcement learning environment that simulates a real-world task: **drug repurposing**.
Discovering new uses for existing FDA-approved drugs is a critical task in modern pharmacology. It saves years of clinical trials and billions of dollars in research.
In this environment, an AI agent navigates a biomedical knowledge graph containing diseases, drugs, protein targets, and biological pathways. The agent's goal is to explore these biomedical connections to discover underlying mechanistic overlap (i.e., shared pathways) and propose a valid, biologically plausible drug repurposing candidate for a target disease over a multi-step episode. 

## Action Space
The environment uses a typed action space. Actions are structured as JSON/Pydantic models with the following fields:
- `action_type` (str): One of `explore_drug`, `explore_target`, `explore_pathway`, `explore_disease`, or `propose_repurposing`.
- `node_id` (str): Flow control identifier for the destination node (must be present in `available_actions` or `candidate_drugs`).
- `reasoning` (str): An explanation of the biomedical reasoning used, scored heavily when grading the final proposal.

## Observation Space
The environment uses a typed observation space returning relevant state to the agent:
- `current_disease` & `current_disease_name` (str): The target condition for which a drug must be repurposed.
- `current_node_id`, `current_node_type`, `current_node_name` (str): The agent's current position in the knowledge graph.
- `available_actions` (array): Immediate neighbor nodes reachable from the current node.
- `visited_nodes` (array): All node IDs visited in the current episode.
- `candidate_drugs` (array): Discovered drugs with their respective `pathway_overlap_score`.
- `last_action_result` (str): A human-readable step result description.
- `exploration_score` & `reward` (float): The current total shaped reward and the reward for the previous step.
- `steps_remaining` (int), `done` (bool): Episode control indicators.

## Tasks and Difficulty
The environment defines three progressively challenging tasks to test LLM reasoning capabilities:

1. **explore (Difficulty: Easy)**
   - **Goal:** Traverse the knowledge graph to visit at least 8 unique nodes.
   - **Difficulty:** Evaluates basic step-by-step API interaction and neighbor selection without an end outcome requirement.
   - **Baseline Score:** 0.42

2. **find_target (Difficulty: Medium)**
   - **Goal:** Navigate specifically to a protein target or biological pathway directly linked to the target disease within 15 steps.
   - **Difficulty:** Requires directed navigation instead of random sampling.
   - **Baseline Score:** 0.48

3. **repurpose (Difficulty: Hard)**
   - **Goal:** Systematically collect evidence and formally propose an optimal drug for repurposing based on shared biological pathways and mechanistic reasoning.
   - **Difficulty:** The final response is rigorously graded across biological plausibility, novelty, reasoning specificity, and existing literature support. Requires sophisticated synthesis.
   - **Baseline Score:** 0.38

## Setup and Usage

**Running Locally via Docker:**
```bash
docker build -t meta-drug-env -f Dockerfile .
docker run -p 8000:8000 meta-drug-env
```

**Running the Baseline Agent:**
Configure your LLM connection, ensuring the variables match standard OpenAI-compatible endpoints:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hf_token_here"
```
Execute the baseline inference script:
```bash
python inference.py
```
