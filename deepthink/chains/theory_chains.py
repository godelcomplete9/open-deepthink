"""
Theory Synthesis chain factories.

One swarm member is one QNN, modified for problem solving. The chains follow
the run order:

  0. Brief (documents only)            get_theory_brief_chain
  1. Sizing (auto)                     get_theory_sizer_chain
  2. Seeds (verbs + nouns)             get_theory_seed_chain
  3. Personas (one call per QNN)       get_theory_member_spanner_chain
  4. Descent: split cells until easy   get_theory_decomposer_chain
  5. Ascent: solve leaves, synthesize  get_theory_cell_chain (solver and synthesizer prompts)
  6. Verification of each proposal     get_theory_verifier_chain
  7. Consistency of the body of work   get_theory_arbiter_chain
  8. Dissent: evolve, reframe          get_theory_persona_evolver_chain, get_theory_reframer_chain
  9. Optional write-up                 get_theory_report_chain

Every prompt that reads the shared body of work receives it as ``{ledger}``.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def get_theory_brief_chain(llm):
    """Step 0: extract the exact problem from attached documents."""
    prompt = ChatPromptTemplate.from_template("""
You are the Theory Brief writer. Extract from the request and the attached documents everything a
solver needs: the exact problem statement, definitions, notation, given data and constraints.
Do not solve it and do not drop any condition.

Request:
---
{user_input}
---

Documents:
---
{document_context}
---

Theory Brief:
""")
    return prompt | llm | StrOutputParser()


def get_theory_sizer_chain(llm):
    """Step 1 (auto): classify the problem and size one QNN member."""
    prompt = ChatPromptTemplate.from_template("""
You size a Theory Synthesis swarm member (one QNN) for a single problem. Classify the problem and
recommend how deep to decompose it.

Problem:
---
{problem}
---

Document context (optional):
---
{document_context}
---

Domain hint from the user: {domain_hint}

Domains: math (proofs, computations), olympiad (one IMO-style problem instance), algorithms
(design and analysis), physics (models, derivations, estimates).
Answer types: number, expression, set, tuple, yesno, interval, formula, complexity, construction,
proof, other. Use "proof" only when the problem asks to prove a statement and nothing else.

Respond with JSON only:
{{
  "domain": "<math|olympiad|algorithms|physics>",
  "answer_type": "<one of the answer types>",
  "difficulty": <1-10>,
  "recommended_depth": <1-4>,
  "recommended_branching": <2-5>,
  "recommended_layers": <2-4>,
  "recommended_width": <2-5>,
  "reasoning": "<one or two sentences>"
}}
""")
    return prompt | llm | StrOutputParser()


def get_theory_seed_chain(llm):
    """Step 2: seed verbs and nouns; they become each column's word-vector."""
    prompt = ChatPromptTemplate.from_template("""
You are the Theory Seed Generator. Generate exactly {word_count} unique single-word seeds. They
become the guiding words (word-vectors) that shape solver personas.

Domain: {domain_label}

Problem:
---
{problem}
---

Requirements:
1. About half are verbs: proof moves and techniques (for example {seed_verbs}).
2. About half are nouns: objects and structures (for example {seed_nouns}).
3. Mix words tightly related to the problem with words from far fields, so personas cover
   unexpected methods.
4. Single tokens only. No filler words.

Output ONLY a single space-separated string of words.
""")
    return prompt | llm | StrOutputParser()


def get_theory_member_spanner_chain(llm):
    """Step 3: span all personas of one QNN member in a single call."""
    prompt = ChatPromptTemplate.from_template("""
You are the QNN Node Generator for a Theory Synthesis member. One member is one small QNN: a grid
of {layers} layers × {width} columns. Write every agent's persona in one pass.

Domain: {domain_label}

Problem (the personas must fit it; do not solve it):
---
{problem}
---

Roles:
- Layer 0 agents are SOLVERS. Each solves small cells directly, so its skills must be concrete
  techniques (for example {skill_examples}).
- Layers 1+ are SYNTHESIZERS. They collapse solved subcells into the parent cell's solution and
  check that the pieces fit: every case covered, no hidden assumption, the answer to the question
  actually asked.

Each column has its own guiding words (a word-vector of verbs and nouns). Blend them into every
agent of that column: a career and attributes shaped by the guiding words, skills that extend them.

Columns:
{columns}

Agent ids to fill (exactly these):
{agent_ids}

Respond with JSON only:
{{
  "agents": [
    {{"id": "agent_0_0", "name": "<name>", "specialty": "<niche specialty>", "skills": ["<4-6 techniques>"], "system_prompt": "<second person, 3-6 sentences: who they are, how the guiding words shape their thinking, their role (solver or synthesizer), and that every step must be justified>"}}
  ]
}}
""")
    return prompt | llm | StrOutputParser()


def get_theory_decomposer_chain(llm):
    """Step 4: the descent. Rate a cell's easiness for the best solver; split it if too hard."""
    prompt = ChatPromptTemplate.from_template("""
You are the Cell Decomposer of a Theory Synthesis QNN. This is the descent step: you split a cell
until every piece is easy for one of the solvers below. You do not solve anything.

Domain: {domain_label}
{domain_brief}

Original problem (ground truth; never change it):
---
{problem}
---
{framing}
Path from the root to this cell:
---
{path}
---

THIS CELL (id {cell_id}, depth {depth} of at most {max_depth}):
---
{statement}
---

Solvers available (layer 0 of this QNN) and their skills:
{solvers}

{ledger}

Steps:
1. Pick the solver whose skills fit this cell best.
2. Rate easiness from 0 to 1: the probability that this solver, working alone in one pass,
   produces a complete and correct result for this cell.
3. If easiness is at least {threshold}, or the cell cannot be split into strictly easier parts,
   set "atomic": true and return no subcells.
4. Otherwise split it into 2 to {branching} subcells. Each subcell must be strictly easier than
   this cell and self-contained (restate the notation it needs). Kinds: lemma, case,
   computation, construction, bound, reduction, check. Cases must cover every possibility. The
   subcell results must combine into a result for this cell; say how in "plan". Use
   "depends_on" for earlier subcell indices (0-based) whose results a subcell needs.
5. If the body of work already establishes a subcell (a verified, corroborated or proposed
   entry), set "reuse" to that entry id instead of asking for a new solution.
6. Rate each subcell's easiness for its best solver.

Respond with JSON only:
{{
  "best_solver": "<agent id>",
  "easiness": <0.0-1.0>,
  "atomic": <true|false>,
  "plan": "<how the subcell results combine into this cell's result>",
  "subcells": [
    {{"statement": "<precise and self-contained>", "kind": "<lemma|case|computation|construction|bound|reduction|check>", "depends_on": [], "best_solver": "<agent id>", "easiness": <0.0-1.0>, "reuse": "<entry id or empty>"}}
  ]
}}
""")
    return prompt | llm | StrOutputParser()


def get_theory_cell_chain(llm):
    """Step 5: one cell of the ascent. The pipeline renders the solver or synthesizer prompt."""
    prompt = ChatPromptTemplate.from_template("{input}")
    return prompt | llm | StrOutputParser()


SOLVER_PROMPT = """{persona}

You are a Theory Synthesis solver working on one small cell of a larger problem.

Domain: {domain_label}
{domain_brief}

Original problem (ground truth; context only, solve only YOUR cell):
---
{problem}
---
{framing}
Path from the root to your cell:
---
{path}
---

Results you may use (the cells this one depends on):
---
{dependencies}
---

{ledger}

YOUR CELL (id {cell_id}, kind {kind}):
---
{statement}
---

Rules:
- Solve exactly this cell. Show every step and justify each one.
- Use only what is given, the results listed above, and body-of-work entries you cite by id.
- If the cell is false as stated, say so and give a counterexample.
- Run the checks that fit: {checks}.

Respond with JSON only:
{{
  "result": "<complete derivation or proof of this cell>",
  "claim": "<one sentence: exactly what you established>",
  "final_answer": "<the cell's answer in its shortest form (number, expression, set, yes/no), or proof>",
  "confidence": <0.0-1.0>,
  "checks": "<checks you ran and what they showed>",
  "uses": ["<body-of-work ids you relied on>"],
  "same_as": "<id of an entry that asks exactly what this cell asks, or empty>",
  "contradicts": ["<ids of entries your result shows are wrong>"]
}}
"""

SYNTHESIZER_PROMPT = """{persona}

You are a Theory Synthesis synthesizer (layer {layer}). Solvers and lower synthesizers solved the
subcells of the cell below. Collapse their results into a solution of THIS cell.

Domain: {domain_label}
{domain_brief}

Original problem (ground truth; never change it):
---
{problem}
---
{framing}
Path from the root to this cell:
---
{path}
---

THIS CELL (id {cell_id}, kind {kind}):
---
{statement}
---

Decomposition plan: {plan}

Subcell results:
---
{children}
---

Results this cell depends on:
---
{dependencies}
---

{ledger}

Rules:
- Check that the subcell claims really combine into this cell's result. Fill any missing step.
  If a subcell result is wrong, fix it, say so, and list its body-of-work id in "contradicts".
- Do not drop cases. Keep the argument complete; do not just summarize.
{root_rule}

Respond with JSON only:
{{
  "result": "<the complete solution of this cell>",
  "claim": "<one sentence: exactly what this cell establishes>",
  "final_answer": "<the answer in its shortest form, or proof>",
  "confidence": <0.0-1.0>,
  "gaps": ["<anything still unproven>"],
  "uses": ["<body-of-work ids you relied on>"],
  "contradicts": ["<ids of entries shown to be wrong>"]
}}
"""

ROOT_RULE = (
    "- This is the ROOT cell. The result must be a complete, self-contained solution of the "
    "original problem, ending with the final answer in \\boxed{...}."
)


def get_theory_verifier_chain(llm):
    """Step 6: independent check of one proposal and the results it rests on."""
    prompt = ChatPromptTemplate.from_template("""
You are an independent verifier for Theory Synthesis. You did not write the solution below. Do not
trust its claims: recheck every step.

Domain: {domain_label}
Checklist: {checks}

Problem:
---
{problem}
---

Proposed solution:
---
{solution}
---

Stated final answer: {final_answer}

Body-of-work entries this solution rests on:
{entries}

Score the probability that the final answer AND the argument are correct. For each entry above,
say whether you confirmed it or found it wrong.

Respond with JSON only:
{{
  "score": <0.0-1.0>,
  "verdict": "<valid|minor_gaps|major_gaps|invalid>",
  "issues": ["<specific error or gap, and the step where it occurs>"],
  "answer_check": "<an independent check of the final answer, such as small cases or a substitution>",
  "confirmed": ["<entry ids you checked and found correct>"],
  "refuted": ["<entry ids you found wrong>"]
}}
""")
    return prompt | llm | StrOutputParser()


def get_theory_arbiter_chain(llm):
    """Step 7: settle a conflict between two entries of the body of work."""
    prompt = ChatPromptTemplate.from_template("""
You are the Theory Arbiter. Two entries in the swarm's body of work conflict. Decide which one holds.
Recheck the mathematics yourself; do not count votes.

Problem the swarm is solving:
---
{problem}
---

Why they conflict: {reason}

ENTRY A ({a_id}, {a_kind}, {a_status}, written by {a_support} QNN(s)):
Statement: {a_statement}
Answer: {a_answer}
Derivation:
{a_result}

ENTRY B ({b_id}, {b_kind}, {b_status}, written by {b_support} QNN(s)):
Statement: {b_statement}
Answer: {b_answer}
Derivation:
{b_result}

Verdicts: "A" (A holds, B is wrong), "B" (B holds, A is wrong), "neither" (both are wrong),
"both" (they do not actually contradict each other, for example because they ask different things).

Respond with JSON only:
{{"verdict": "<A|B|neither|both>", "reason": "<the decisive step, in one or two sentences>"}}
""")
    return prompt | llm | StrOutputParser()


def get_theory_persona_evolver_chain(llm):
    """Step 8a: qualitative Mirror Descent on one dissenting QNN's personas (one call)."""
    prompt = ChatPromptTemplate.from_template("""
You are the Persona Evolver for a Theory Synthesis member (qualitative Mirror Descent). This QNN's
proposal lost the swarm vote, failed verification, or rests on refuted results. Rewrite its agents
so its next attempt fails differently.

Loss this round: {loss} (0 = agreed with the swarm and verified, 1 = dissent or failure)
Learning rate: {learning_rate} (0 = almost no change, 1 = moderate, 2 = radical reinvention)

Verifier issues:
{issues}

Weakest cells of this QNN's last tree (lowest confidence or refuted):
{weak_cells}

Current agents (JSON):
{agents}

Rules:
- Keep every id and its role (layer 0 = solver, layers 1+ = synthesizer).
- A wrong or weak leaf: sharpen that solver's techniques and demand explicit checks.
- Gaps in synthesis: make the synthesizers demand the missing links and full case coverage.
- Do NOT mention, guess or hint at any final answer.

Respond with JSON only:
{{"agents": [{{"id": "<same id>", "system_prompt": "<rewritten>", "skills": ["<4-6 techniques>"]}}]}}
""")
    return prompt | llm | StrOutputParser()


def get_theory_reframer_chain(llm):
    """Step 8b: alternative representations of the same problem for dissenting QNNs."""
    prompt = ChatPromptTemplate.from_template("""
You are the Theory Reframer. The swarm of QNNs disagrees about this problem. Write {count} different
framings of the SAME problem for the QNNs that will try again. A framing is a change of
representation or method, not a new problem.

Domain: {domain_label}

Problem (ground truth; never change it):
---
{problem}
---

How the swarm split (the answers are hidden from you on purpose):
{dispute}

Open questions in the body of work:
{open_questions}

Rules:
- Keep every condition of the problem. Do not simplify it, add assumptions or change what is asked.
- Each framing names a representation or method and the first step it suggests (for example
  {framing_examples}).
- Do not state or guess any answer.

Respond with JSON only:
{{"framings": ["<framing 1>", "<framing 2>"]}}
""")
    return prompt | llm | StrOutputParser()


def get_theory_report_chain(llm):
    """Step 9 (optional): a readable write-up that keeps the mathematics unchanged."""
    prompt = ChatPromptTemplate.from_template("""
You write the final Theory Synthesis report. Do not change the mathematics: keep every step of the
winning solution and its final answer exactly. You may fix formatting and wording.

Problem:
---
{problem}
---

Winning solution:
---
{solution}
---

Final answer: {final_answer}

How the swarm agreed:
{consensus}

Body of work:
{body}

Verifier notes:
{verifier}

Write markdown with these sections: "## Answer", "## Solution", "## How the swarm agreed",
"## Body of work", "## Checks and caveats". End the caveats with: agreement between QNNs is not a
proof; check the argument.
""")
    return prompt | llm | StrOutputParser()
