<p align="center">
  <img width="1280" alt="open-deepthink — Qualitative Neural Networks, Diffusion, Distillation, RSI" src="static/open-deepthink-banner.png" />
</p>

# open-deepthink: Evolvable Agent Networks for Deep, Structured Reasoning

> **∑ New: Theory Synthesis.** Brainstorming mode is now Theory Synthesis, a harness for parallel problem solving. A swarm of QNNs works on one math problem, algorithm design task, physics question or IMO-style instance. Each QNN splits the problem into cells until every cell is easy for one of its solvers, and a hierarchy of synthesizers collapses the cells back into a proposal. Every cell writes into one shared, consistency-checked body of work, and the swarm converges through a vote weighted by mirror descent. [Jump to Theory Synthesis ↓](#-theory-synthesis-a-swarm-of-qnns-for-one-hard-problem)

> **🛡 RSI for cybersecurity.** A recursive self-improvement loop that covers an institution's security policies with a lattice of agents, finds where the rules clash, and evolves them with controls that must raise a measured posture. It is built on the *Agent Mechanics* treatise, and a local model learns from every pass through a LoRA loop. [Jump to RSI ↓](#-rsi-recursive-self-improvement-for-cybersecurity)

**Not another flat panel of 16 agents brainstorming once.**  
A **Qualitative Neural Network (QNN)** that runs layered forward passes, reflects on its own performance, mutates its agents' cognitive identities, raises the difficulty of the problem, and records the entire developmental history as high-signal training data.

Most agentic systems give you breadth through parallelism. open-deepthink gives you **depth through structured iteration and self-modification**.

### What this is (and is not)

This is a **structured multi-agent loop**: layered LLM calls, lexical retrieval of past cells, prompt rewrite between epochs, and JSON traces. Neural-net words (Mirror Descent, self-attention, diffusion) are a **design language**. They are not those algorithms. There is one exception: Theory Synthesis updates its swarm's vote weights with real entropic mirror descent (exponentiated gradient on the simplex). Output quality versus a single long prompt is **unevaluated**. See [`docs/DESIGN_NOTES.md`](./docs/DESIGN_NOTES.md) and [`docs/EVAL.md`](./docs/EVAL.md).

Two modes are new and are described first, below. **Theory Synthesis** replaces Brainstorming mode in the web UI: a swarm of QNNs solves one hard problem and builds a shared body of work. **RSI** applies recursive self-improvement to cybersecurity: an institution's policies are read as a law, covered with Agent Mechanics agents, healed where they clash, and evolved with posture-gated controls.

**Start small.** Auto mode caps at 24 agents. Prefer `2×2×1` or `3×3×2`. Estimate before you spend:

```bash
deepthink estimate theory --swarm 5 --rounds 2   # upper bound on LLM calls
deepthink estimate qnn --layers 2 --width 2 --epochs 1
deepthink eval          # mock LLMs, no API key — structural checks only
```

---

## ∑ Theory Synthesis: a swarm of QNNs for one hard problem

**Theory Synthesis runs many QNNs in parallel on one problem and makes their work add up.** It is built for math problems, algorithm design, physics, and single instances of IMO-style problems. In the web UI it replaces Brainstorming mode.

Each QNN in the swarm splits the problem into a tree of cells until every cell is easy for one of its solvers. It solves those cells, then synthesizes the results back up into one proposal. Every cell's result goes into one shared, consistency-checked **body of work** that every agent reads. The swarm votes on the final answer with weights updated by mirror descent. It repeats with the dissenting QNNs until enough of the swarm agrees.

> Agreement is not proof. The vote measures how much of the swarm backs an answer. The verifier and the arbiter are model calls and can be wrong. Because the QNNs share a body of work, they build on each other's results, so their votes are not independent samples. Check the argument. Quality on real olympiad problems is unevaluated.

### What a run gives you

| Output | What it is |
|---|---|
| **Answer** | The leading answer and the winning QNN's complete solution. It says how many QNNs back the answer (support) and whether that reached the quorum. |
| **Vote** | Every answer cluster in every round: QNNs, support, mass, mean check score, and the entropy of the weights. |
| **Decomposition** | The winning QNN's cell tree. For each cell: who solved or synthesized it, its easiness, its claim, and its body-of-work entry. |
| **Body of work** | Every result the swarm wrote. Each has a status (verified, corroborated, proposed, disputed, suspect or refuted), authors and dependencies. Conflicts, their verdicts and reuse are also listed. |
| **Trace** | Every QNN's seeds, personas, framings, proposals, losses and weights per round, and the body-of-work entries each cell was shown, as JSON. |

### The mechanism, step by step

**1. One QNN, modified.** Each member of the swarm keeps the QNN heuristic. Seed verbs and nouns (proof moves, mathematical objects, and words from far fields) are sampled into per-column word-vectors. Each column spans one persona per layer. **Layer 0 agents are solvers** with concrete techniques. **Layers 1 and up are synthesizers.** A QNN is L layers × W columns. *Auto* sizes it per problem, with at most 24 agents per QNN.

**2. Descent: split the problem until each cell is easy.** The root cell is the problem. A decomposer picks the solver of that QNN whose skills fit the cell best. It rates the cell's **easiness**: the probability that this solver, working alone in one pass, gets the cell completely right.

- If easiness ≥ τ (0.75 by default), the cell is a leaf.
- Otherwise it is split into 2 to *b* subcells whose results combine into the cell's result. A subcell is a lemma, a case, a computation, a construction, a bound, a reduction or a check. A subcell may depend on earlier siblings.
- Each subcell comes with its own easiness rating, so an easy subcell costs no further call.
- The descent stops at depth *D* or when the QNN's cell budget is spent.

**3. Ascent: solve the leaves, then synthesize upward.** All leaves are solved in parallel by layer-0 solvers. A cell waits only for the cells it depends on. Each internal cell is collapsed by a synthesizer from the layer that matches its height in the tree, so layer *k* sees layer *k−1*'s outputs, as in the QNN forward pass. A synthesizer checks that the pieces really combine, fills missing steps and fixes wrong subresults. The root synthesis is the QNN's **proposal**, ending with a boxed final answer.

**4. The body of work.** Without it, an agent sees only its own thought stream: its persona, the cells below it and its own past outputs. With it, every solved or synthesized cell is written to one shared ledger, with its author (QNN, round, cell) and the entries it was built on. Every agent reads the entries most relevant to its cell and cites them by id. This is the QNN's self-attention, reading the shared ledger instead of private memory. The ledger keeps the body of work consistent:

- The same result derived by two QNNs merges into one entry, which becomes **corroborated**.
- A conflict opens when two entries answer the same question differently, when an agent declares its cell the `same_as` an entry with a different answer, or when an agent lists an entry under `contradicts`. Both sides become **disputed**.
- After each round an **arbiter** rechecks up to *N* conflicts and decides which side holds. The losing entry is **refuted**, and the winner is **verified**.
- An entry whose every derivation rests on a refuted or suspect entry becomes **suspect**. So does any proposal built on it, and a suspect proposal loses half its vote.
- The verifier of a proposal can confirm the entries the proposal rests on (**verified**) or refute them.
- A decomposer can mark a subcell as already established, and the QNN then **reuses** a verified, corroborated or proposed entry instead of solving it again. Reuse is accepted only when the entry is about the same thing as the subcell.

Proposals are written to the ledger but never shown to other agents: QNNs share lemmas, not verdicts. In **private** mode each QNN reads only its own entries. That keeps the votes independent, as in classic self-consistency, and conflicts are still detected across the swarm.

**5. Verification.** An independent verifier rechecks each proposal against the domain's checklist. It scores the probability that both the answer and the argument are correct. A proposal that scores below a floor (0.2 by default) abstains.

**6. The vote.** Final answers are put in a canonical form before they are compared:

- `\boxed{\frac{1}{2}}`, `0.5` and `1/2` are one answer;
- `{3, 1, 2}`, `\{1,2,3\}` and `n = 1, 2, 3` are one set;
- `Yes` and `true` are one verdict;
- numbers with units match within a tolerance (relative 1e-3 for physics).

Each QNN carries a weight *wᵢ* (the weights sum to 1) and a quality *qᵢ*. The quality is the verifier score, halved if the proposal is suspect and zero if it is refuted.

- The **mass** of an answer is Σ *wᵢ·qᵢ* over the QNNs that back it. The heaviest answer leads.
- The **support** of an answer is Σ *wᵢ* over the same QNNs. QNNs that abstain keep their weight in the total, so two agreeing QNNs out of ten are 20 % support.
- The swarm **converges** when the leader's support reaches the quorum (0.5 by default).

With `vote = plurality`, weights and quality are ignored: if 100 QNNs answer and answer 1 is the most common, answer 1 wins.

**7. Mirror descent.** After a round that did not converge, each QNN gets a loss:

```text
ℓᵢ = ½ · (1 − support of its answer) + ½ · (1 − qᵢ)
wᵢ ← wᵢ · exp(−η ℓᵢ) / Σⱼ wⱼ · exp(−η ℓⱼ)
```

This is entropic mirror descent on the simplex (exponentiated gradient), so here "Mirror Descent" is the algorithm itself, not a metaphor. Weight flows to QNNs that agree with the swarm and pass their checks, and the leader's support rises until it reaches the quorum or the rounds run out.

**8. Dissent: evolve, respawn, reframe.** QNNs that back another answer, abstain, or rest on refuted results try again. QNNs in the leading cluster keep their proposals.

- Each dissenting QNN's personas are rewritten (the QNN's qualitative Mirror Descent), more strongly the larger its loss: the rewrite rate is η·ℓᵢ.
- A QNN whose weight falls below a floor (0.35 of an even share) respawns from fresh seeds.
- A reframer gives each one a new representation of the same problem, such as an invariant, a change of variables or an energy method. The reframer is not shown any answer.

Then they descend and ascend again, reading everything the swarm has established so far.

```text
                         problem
                            │
        ┌───────────────────┼───────────────────┐     S QNNs in parallel
      QNN 0               QNN 1      …        QNN S−1
        │  descent: split cells until easiness ≥ τ (depth ≤ D, ≤ C cells)
        │  ascent: layer 0 solves leaves, layers 1…L−1 synthesize upward
        │  proposal → verifier
        └──────────────── body of work ─────────────────┘   merge · conflicts · arbitration
                            │                               taint · reuse
              vote: canonical answers, mass, support
                            │
              quorum reached → answer
              otherwise → mirror descent on the weights;
                          dissenters evolve, respawn or get a new framing ↺
```

### Domains

| Domain | Solvers are told to | The verifier checks |
|---|---|---|
| `math` | state their notation, justify every step, prefer exact values | each step follows, edge cases, small cases, and that the answer answers the question asked |
| `olympiad` | treat the problem as one IMO-style instance, test small cases first, prove both directions of "find all", find the equality case | small cases, completeness of "find all" answers, equality cases, explicit constructions, no unproven "well-known" claims |
| `algorithms` | give the algorithm, a correctness argument and its complexity, and claim optimality only with a lower bound | correctness in every case, the complexity bound, edge cases, a lower bound for any optimality claim |
| `physics` | state the model, use conservation laws and symmetry, carry units, check dimensions and limiting cases | dimensions, limiting cases, conservation, order of magnitude, signs |

With `auto`, a sizer classifies the domain and the answer type, with a keyword guess as the fallback. On a "prove that" problem every QNN's answer is "proof" unless one finds a counterexample, so the vote turns on the verifier's scores.

### Controls

| Control (UI / param) | Default | Meaning |
|---|---|---|
| QNNs in the swarm / `swarm_size` | 5 | how many QNNs run in parallel (1–128) |
| Rounds / `rounds` | 2 | swarm rounds, the QNN's epochs |
| Quorum / `quorum` | 0.5 | the leader's support needed to stop |
| Topology / `qnn_mode`, `manual_layers`, `manual_width` | auto | each QNN's layers × width; layer 0 solves, the rest synthesize |
| Max depth / `max_depth` | 2 | how many times a cell may be split |
| Branching / `branching` | 3 | subcells per split, at most |
| Easiness τ / `easiness` | 0.75 | a cell is a leaf once its solver rates it this easy |
| Cells per QNN / `max_cells` | 10 | size cap of each QNN's tree |
| Vote / `vote` | weighted | `weighted` (mirror-descent weights × checks) or `plurality` |
| Learning rate η / `learning_rate` | 1.0 | the mirror-descent step and the strength of persona rewrites |
| Body of work / `ledger` | shared | `shared` across QNNs or `private` to each |
| Verify proposals / `verify`, `min_verify` | on, 0.2 | proposal checks; below the floor a proposal abstains |
| `max_arbitrations` | 4 | conflicts settled per round |
| Parallel calls / `concurrency` | 8 | simultaneous LLM calls across the whole swarm |
| Max LLM calls / `max_calls` | 0 | a hard budget (0 = none); the run stops cleanly when it is spent |
| Temperature min–max / `temperature_min`, `temperature_max` | provider default | an optional temperature spread across QNNs |
| Per-Agent Models (settings) / `--member-models` | — | a mixed-model swarm: QNNs take the models in turn |
| Polish the report / `polish` | off | one extra call to rewrite the report; the mathematics is kept |
| `seed` | random | makes the sampling of seed words reproducible |

### Run it

**Web UI:** the **∑ Theory Synthesis** tab. Set the swarm and the descent, state one problem in the chat, and watch the swarm panel:

- every QNN is a square, colored by the answer it backs and pulsing while it works;
- the vote is a set of bars, one per answer, against a quorum marker;
- the body of work is a running count: results, verified, corroborated, refuted, conflicts settled, reuses.

The report arrives in the chat: the answer, the solution, how the swarm agreed, the winning QNN's decomposition, the body of work and the checks. *Simulation Mode* runs the whole engine on a mock model, with no API key.

**CLI:**

```bash
deepthink theory --debug -p "Let a_1 = 2 and a_{n+1} = a_n + 1. Find a_2025."    # Simulation Mode
deepthink theory --domain olympiad --swarm 5 --rounds 2 --input problem.txt --yes
deepthink theory -p "…" --member-models <model-a>,<model-b> --swarm 6 --max-calls 300 --yes   # mixed-model swarm
deepthink theory -p "…" --private-ledger --vote plurality --swarm 20 --depth 1 --yes          # independent votes
deepthink estimate theory --swarm 5 --rounds 2 --depth 2
```

**Library:**

```python
import asyncio
from deepthink import create_llm, run_theory

async def main():
    llm = create_llm()  # OpenRouter from env, or create_llm(provider="llamacpp")
    r = await run_theory(
        llm,
        "Find all positive integers n such that 3 divides 2^n + 1.",
        params={"domain": "olympiad", "swarm_size": 5, "rounds": 2, "concurrency": 8},
    )
    print(r["final_answer"], r["converged"], r["support"])
    print(r["body_of_work"]["stats"])   # verified, corroborated, refuted, conflicts, reuses

asyncio.run(main())
```

**Cost.** `deepthink estimate theory` prints an upper bound. Each cell costs at most one decomposer call and one solve or synthesis call. Each proposal adds one verification, each round up to `max_arbitrations` arbitrations, and each later round one reframe and one evolve or respawn call per dissenting QNN. The defaults (5 QNNs, 2 rounds, depth 2, branching 3, 10 cells) come to at most 171 calls. Easy problems, early convergence and reused results cost much less: a simulated 8-QNN run that converged in 2 rounds made 65 calls. The CLI refuses runs estimated above 80 calls unless you pass `--yes` or cap the run with `--max-calls 80`.

### What is measured and what is not

- **Measured (in the test suite):** that canonical forms match equal answers and keep different ones apart; the vote; the mirror-descent step (closed form, rising leader support); the ledger's rules for merging, conflicts, arbitration and taint; the limits on depth, branching and cells; dependency order; the concurrency cap and the call budget; and that the cost estimate bounds worst-case runs.
- **Readings, not measurements:** easiness ratings, verifier scores and arbiter verdicts are model judgments.
- **Not claimed:** that the swarm solves IMO problems, that agreement means correctness, or that a larger swarm is better. Quality on real problems is unevaluated.

---

## 🛡 RSI: recursive self-improvement for cybersecurity

**RSI covers an institution's security policies with a lattice of agents.** It finds where the policies contradict themselves, heals them, and evolves them with new controls. A control is kept only when the policy's measured posture improves. Every pass is distilled into a LoRA adapter for the local model that drives the agents, so the next pass needs fewer reasoning rounds.

It runs on **Agent Mechanics**, a formal theory of agents as conductors of eight species of information ([the treatise](./docs/Agent_Mechanics_Treatise_EN.pdf)). It also uses **NX-1**, a nomenclature complement written for this engine ([`docs/NOMENCLATURE_COMPLEMENT.md`](./docs/NOMENCLATURE_COMPLEMENT.md)).

> RSI reads three kinds of artifact: the **text** of your policies, runbooks and standards; the **code and configuration** of a repository; and a **role-level profile** of the procedures people run, for social engineering. It never executes code, probes a network, contacts or tests a person, or certifies anything. Its findings are leads for a human reviewer, and its posture score is bookkeeping over the policy text. Nothing here has been validated against real incidents.

### What a run gives you

| Output | Produced by | What it is |
|---|---|---|
| **Findings** | the leak-driven side (defensive) | Clauses that cannot both hold, and functions the mapping puts at odds. Each comes with the clauses involved, the licensed operator that healed it in the lattice, and the remediation it implies. Findings are ranked by severity. |
| **Controls** | the stock-driven side (evolving) | New control clauses written by agents that have a surplus. Each one is kept only if posture rises and no new contradiction appears. |
| **Hardened policy** | `artifact_emit` | The policy with the kept controls marked, plus the findings and function coverage, in one Markdown report. |
| **NX-1 log** | the world | Every ensemble as a canonical string, the table of correspondences, and a tick-stamped event log in NX-1 notation. |
| **LoRA dataset** | the `lora` node | Heal and control traces as chat-format JSONL, ready for a trainer. |

### The mechanism, step by step

**1. The policy becomes a law.** Clauses are parsed onto four levels: **L3 principles** (purpose, scope), **L2 policies** (who may do what), **L1 controls** (prevent, detect, respond, recover) and **L0 assets**. Section headers decide the level when present; keywords decide it otherwise. Each level carries a *base* (how it packs the level below) and a *modulus* (what closes it, such as least privilege for L2).

**2. Every clause, modulus and viable operation becomes an agent.** A **table of correspondences** maps each element onto the eight species of Agent Mechanics. Each species is read as a security function:

| Species | Security function | Species | Security function |
|---|---|---|---|
| **Ni** | threat forecasting, attack paths | **Ti** | access rules, input validation |
| **Ne** | red-team testing, discovery | **Te** | incident response, patching |
| **Si** | audit logging, retention, baselines | **Fi** | identity, integrity, least privilege |
| **Se** | real-time monitoring, alerting | **Fe** | coordination, notification, rollout |

The mapping is done by a **mapper** model. One scoring **above 70 on ARC-AGI-2** is recommended, since the mapping is an abstraction-transfer task. The score you declare is recorded with every row, not verified. Without a mapper, the local driver maps, and offline a lexicon does. Viable operations, one *add control* per under-covered function, become agents too. Each agent gets a name in the treatise's nomenclature, for example `A7=Res-Fi>Ti-stem-FdO/s=2/L=0.24/S=0.76/Lk=Fi:op0.31/…/op=add_control`. Agents that store each other's current cluster into **ensembles** with canonical names.

**3. The agents tick.** The Agent Mechanics tick law runs on the lattice. An agent offers its stock through the channel matrix `R`, gated by its energy. Receivers store what fits and lose the rest as **Leak**. Leak is split by cause: hostile current (`op`), overdrive (`ov`) and sub-quantum arrivals (`q`). Total reserve is conserved except for intake and Leak, and the test suite checks this to 1e-6. Agents climb ranks (Seed → Res → Cyc → Attr) only through licensed operators paid for with energy.

**4. Three domains, joined by a gauge field.** The engine works in three domains at once:

```text
   AX  the policy (clauses, moduli, operations)   ← where the problem lives
    │   link phase A = how faithfully an agent carries its clause's function
   TC  the table of correspondences                ← the gauge connection
    │
   AM  the Agent Mechanics lattice of agents       ← where the laws are explicit
```

Contradictory clauses carry phase π on their artifact edge. Hostile agents carry phase π on their lattice edge. Around every square (clause → clause → agent → agent), the curvature `B = d₁A` measures whether the correspondences can all hold. Its **frustration** `(1 − cos B)/2` is how the engine detects a **clash**. Curvature is gauge invariant (`A → A + d₀χ` leaves it unchanged), so relabelling cannot create or hide a clash. The tests check this.

**5. Leak drives defence.** An agent whose stability `S = G/(G + Leak)` falls below `S_min` (blended with its clash frustration by `λ_AM`) **heals in the lattice**. It simulates each licensed operator and applies the one that lowers its Leak most:

- `a`: flip attitude
- `r`: cover the axis complement
- `ra`: move to the norm axis
- `σ`: open a stem
- `cut`: shield a hostile edge
- `→`: absorb an overload

The **symbolic gauge coderivative** `δ_A` then carries the operator back to the policy as an edit: `r` becomes "cover the complementary function", `cut` becomes "isolate the clashing scope", and so on. The heal is recorded as a **finding**. This is the hyperparameter trick of the design: permutations are searched where the laws are explicit and the search is small (16 stacks, 4 writings, a handful of operators), and the winner is pulled back to the policy.

**6. Stock drives evolution.** An agent whose total reserve grows while it emits (`Δŵ ≥ 0`, energy `x ≥ 1`) has a surplus. It spends it drafting a **control** for the function its operator points to; the local driver phrases it. The control is gated by **posture**: the Shannon diversity of the eight functions across the policy, divided by one plus the number of contradictory clause pairs. If posture rises and no contradiction is added, the control enters the policy and is born as a new agent (a `pr` edge from its payer). Its success flows back as mass intake to the agent that wrote it. If posture does not rise, the control is rolled back.

**7. The loop distils itself into the local model.** Each loop appends its heal and control traces to a JSONL dataset. When the loop measurably improves (fewer reasoning rounds per solve, lower ensemble Leak, or higher posture), the adapter scale is raised one step and **clamped into the local llama.cpp driver** through `POST /lora-adapters`. When the next window does not improve, the scale returns to the last good value. Training runs only if the server sets `RSI_LORA_TRAIN_CMD`. Without an adapter, the same traces act as a distilled memory, so a known heal costs one round. That offline stand-in is what the tests measure. In treatise terms, a LoRA is side mass: stock the loop already paid for, folded into weights, making the most-used routes cheaper.

**7b. Optional: weight-space programming.** Instead of raising the scale by a fixed step, a powerful **editing model** (for example Claude Fable 5.2 or GPT-6.2 Astra) can program the adapter. On each improving window it reads the loop metrics and a sample of traces, then returns one JSON program with three parts:

- `data`: traces to drop, shorter answers to rewrite, and up to 16 examples to add;
- `train`: rank, alpha, dropout, learning rate, epochs and target modules for the trainer;
- `mix`: a weighted sum of the adapters already loaded on llama.cpp (task arithmetic), applied in one `POST /lora-adapters`.

The engine validates every program before it is used. It snaps the rank to 4–64, caps alpha at 4× rank and the learning rate at 1e-3, and drops unknown modules. It ignores adapter ids that are not loaded, clamps scales to `[0, clamp_max]` (negative only with `allow_negative`), and never drops more than half the dataset. Every correction is logged. The programmed dataset and spec are written next to the raw traces and passed to the trainer as `{dataset}` and `{spec}`. The keep/rollback gate is unchanged: a window that does not improve returns to the last good mix.

The editor is opt-in and has its own tier. It never falls back to the local driver. Without an API key or model id, the program is a deterministic offline one (deduplicated traces, default spec, one clamp step). It sends no sampling parameters, which newer models reject. **It sees trace text that quotes your policy**, so only enable it for policies you may send to that provider.

**7c. Optional: active evolution (alternate universes).** See [Active evolution](#active-evolution-alternate-universes) below. When it is on, a phase between stock and lora branches the lattice into hypothetical universes, breeds new structures in them, and reports each one as a what-if on the policy.

**8. Space and time.** A covered cell is a (clause, function) pair held by a stable, active agent. The benchmark reports cells covered, reasoning rounds per solve, findings, controls and ensemble Leak for each drive. Faster loops and shorter thinking both widen coverage per unit of time. Coverage means an agent stands for that function. It does not mean the function is secure.

### Beyond the policy: code and people

A written rule can be contradicted by what actually runs, or by how people carry it out. Two optional inputs bring those artifacts into the loop. Each observation becomes a **fact**, and each fact is read against the policy clauses on the same topic:

| Status | Meaning | In the lattice |
|---|---|---|
| **contradicts** | a clause forbids or requires the opposite | edge with phase π to that clause, so it clashes like two contradictory clauses |
| **uncovered** | the policy says nothing about it | edge with phase π/2 to the controls modulus |
| **permitted** | a clause allows it | edge with phase 0 |

Every fact becomes an agent beside the policy's own (`AX:code/C…`, `AX:soc/S…`). A leaking fact agent heals into a finding that quotes the clause it breaks and the concrete remediation. A stocking one proposes that remediation as a new control, which the posture gate keeps or rolls back like any other. Fact agents have their own heal budget, so they never crowd out the policy's internal clashes. Every report also carries a deterministic **discrepancy register** of all facts.

**Repository analysis** (`--repo DIR`, `params["repo_path"]`, [`repo.py`](./deepthink/nomosloop/repo.py)) reads code and configuration for: hard-coded credentials, private keys, TLS verification turned off, debug mode, inbound rules open to the internet, services on every interface, public storage, remote desktop switched on, SQL or shell commands built from strings, weak hashes, containers running as root, world-writable permissions, wildcard CORS, authentication or CSRF checks disabled, and logging turned off. Files are only read. Symbolic links, VCS and dependency folders, binaries and files over 400 KB are skipped, and secret values are redacted before they reach a fact, a prompt or the report.

**Social engineering** (`--social profile.json`, `params["social_profile"]`, [`social.py`](./deepthink/nomosloop/social.py)) reads a role-level profile: for each role, its contact channels, what it can do, how it verifies a requester, how many approvals it needs, whether it is publicly listed and whether it is trained. It finds account resets on weak verification (pretexting), payment or bank-detail changes without dual control (business e-mail compromise), no written rule against urgency or authority overrides, no channel to report a suspicious contact, exposed roles without training, publicly reachable privileged roles, and procedures that are never exercised. A policy that demands a callback while the help desk resets passwords on an employee number is reported as a contradiction between the rule and the routine.

The profile needs an `authorization` block naming who authorized the assessment and its scope. Only role-level fields are admitted: names, contact details, biographies, personality data (Big Five, MBTI), behavioural reports and lists of individuals are dropped before analysis and listed in the report as not admitted. The layer finds weak procedures; it does not profile people or write pretexts and lures. See [`examples/security_surface/`](./examples/security_surface/).

Code and procedure facts are mapped with the offline lexicon (plus the species each detector assigns) unless you pass `surface_remote_mapping`, so repository content and role descriptions do not reach a remote mapper by default. In the web UI, a repository path is accepted only under the folders listed in `RSI_REPO_ROOTS` on the server; the CLI reads any local path you give it.

### Active evolution: alternate universes

Heal and stock only move when something leaks or has a surplus. At equilibrium, with nothing leaking and no control left that raises posture, the loop has nothing new to try until the policy changes. **Active evolution** is a toggle that keeps it exploring. Each loop, agents branch the world into alternate universes, mutate inside the lattice with the treatise's operators, and breed new structures. Every universe reads as a **what if** on the policy.

| Operator | In the lattice (treatise) | What if, in the policy |
|---|---|---|
| `ev_a`, `ev_r`, `ev_ra` | the online writing moves to a Klein image; the identifier is kept (Apparatus §2; IV·bis §8, evolution) | the clause is mirrored (inbound ↔ outbound, read ↔ write), serves its complementary function, or becomes institution-wide |
| `σ` stem | a storing edge to the agent whose hero covers this agent's weakest slot (NX-1 §5; Part IV §5, I→II) | the clause is explicitly backed by the clause that covers what it holds least |
| `→×4` store4 | three children close a four-cycle of stems (Part IV §5, II→III) | the loop is closed with three controls: log, alert, review |
| `η_g` hairpin | a Klein-close child of its parent, paid with energy (Part IV §5, III→IV) | the clause and its mirror, complement or norm form become one standard |
| `a·(αX, βY)` birth | the birth law: a child Klein-close to the winner, stocked with `r` of the residual the pair cannot store (Part I §9; IV·bis Table 4) | a new clause for what two linked clauses lose between them |
| `→` arrow | the parent dissolves and two children share its mass (NX-1 §6.4; restriction) | the clause is split in two at its "and" |
| `cut` | the opponent's edge is cut, `w → 0` (NX-1 §6.4) | the clause yields: "…, except where §j applies" |
| `†` | dissolution: tombstone `!x` (IV·bis §8) | the clause is removed |
| `ω` | off-license emission, heat only (NX-1 §5) | the restriction is reversed (a probe) |

**Operators unlock by rank**, as in Part IV, and every step is paid with energy (`x ≥ 1`). Any agent can flip or stem. A `Res` agent can close a ring. A `Cyc` agent can hairpin. A linked pair whose encounter leaks can give birth. So structures breed structures across generations: a ring bred in generation 1 lifts its members toward `Cyc`, and in generation 2 they can hairpin.

**How a loop runs.** The current world is forked into a baseline and *N* universes, each with one licensed mutation sampled by weight. Repair operators weigh more where agents leak; generative ones weigh more where they stock. Every universe syncs to its hypothetical policy, ticks, and is measured in both spaces against the baseline ticked the same number of times:

- **policy:** posture, contradictions, and functions covered or lost;
- **lattice:** ensemble leak, clash mass, leaking agents, new rings and the ranks of bred agents.

Fitness weighs the two with `λ_AM`. The survivors breed the next generation, each with one more mutation, and the two best with disjoint edits are crossed over (the union of their edits, replayed from the base). `ω` probes run beside them, reversing restrictions one at a time to show which clauses are **load-bearing**.

**Reading a universe.** Each one carries:

- its **what if**, in plain words;
- its **diff**: `~` rewritten, `+` added, `−` removed, `↔` linked;
- its **consequences**, for example "dissolves the contradiction §5 × §8", "covers audit logging" or "new clash in the lattice: §13 × §6·review";
- what changed in the lattice, and the **agents it bred**, with their NX-1 names and whether they are Klein-close to their parents;
- a **verdict**.

| Verdict | Meaning |
|---|---|
| `dissolves` | removes a contradiction from the policy |
| `breaks` | adds a contradiction |
| `leaks` | the policy text shows no contradiction, but the lattice leaks: a strong new clash, a new leaking agent, or higher ensemble leak |
| `breeds` | bred a ring or an agent at `Res` or above, without leaking |
| `improves` | posture rises |
| `neutral` | no measurable change |
| `load-bearing` / `slack` | for `ω` probes: reversing the restriction heats the lattice, or barely changes it |

**Adoption.** By default, universes are what-ifs only and the real loop is untouched; the tests check that posture, controls and findings come out identical with evolution on. With `adopt: gate`, the best universe whose posture rises with no new contradiction is applied to the policy, and its agents are re-derived from the adopted text at the next correspond. Only additions, splits, links and explicit precedences can be adopted. Rewrites (`ev`), removals (`†`), reversals (`ω`) and anything that weakens a restriction stay what-ifs: a restriction yielding to a permission, for example, is never adopted.

Universes are seeded and reproducible (`seed`, or one derived from the policy and the loop), make **no model calls**, and use the offline lexicon for new or rewritten clauses. With `trigger: calm`, branching happens only when no agent leaks, which is the equilibrium case. Every branch is logged as an NX-1 event (`branch(U1-2.3: η_a@§12) => breeds …`), and the best universes are added to the LoRA dataset as what-if traces.

```powershell
deepthink rsi --input examples/security/institution_access_policy.txt --offline --iterations 2 --evolve --universes 8 --generations 2 --evolve-seed 1
deepthink rsi --input examples/security/institution_access_policy.txt --offline --evolve --evolve-trigger calm --evolve-adopt gate
```

In the web UI, tick **Active evolution** in the RSI tab. The universes appear in the **Universes** results tab, and `evolve` joins the loop ring. Mass shares, licensing thresholds and fitness weights are conventions of this engine; the treatise leaves them open ([OG6]).

### Drives and sensitivity

| Drive | Intent | `S_min` | `ζ` | heals/loop | controls/loop | `θ` |
|---|---|---|---|---|---|---|
| `leak` | **defensive**: heal clashes first | 0.62 | 0.30 | 12 | 2 | 0.20 |
| `bal` | heal what leaks, evolve where there is surplus | 0.50 | 0.15 | 6 | 4 | 0.30 |
| `stock` | **evolving**: spend surplus on controls | 0.38 | 0.08 | 3 | 8 | 0.45 |

- **`λ_AM`** (0–1) sets how sensitive the lattice is to the treatise. At `0` decisions follow artifact signals only (clash frustration, mapping confidence). At `1` they follow the Agent Mechanics balance only.
- Every treatise parameter can be overridden through `sensitivity`. The table below lists them.

| Parameter | Meaning |
|---|---|
| `S_min` | storing threshold: below it an agent leaks |
| `ζ` | damage Leak does to capacity |
| `θ` | share of stored flux turned into energy |
| `q` | registration quantum: smaller arrivals are lost |
| `r_scale`, `r_antisym` | channel strength and one-way conductivity |
| `energy_intake` | compute budget per agent |
| `η` | rewriting step, divided by `1 + side mass` |

All values are uncalibrated conventions, as the treatise says of its own ([OG6]).

### Models

- **Driver** (lattice, heals, control drafting) **must be local**: a llama.cpp server on a loopback or private host, or an in-process model. A remote driver is refused, and driver calls never leave the machine.
- **Mapper** (table of correspondences) may be remote, via OpenRouter. If it is missing, the driver maps, and the run logs that the recommendation was not met.
- **Editor** (optional, weight-space programming of the LoRA) may be remote, via OpenRouter: set `RSI_EDITOR_MODEL` or `--editor-model`, or fill *Editing model* in the LoRA panel.
- **Offline** (Simulation Mode, `--offline`): the real engine with deterministic heuristics and no model calls.

### Run it

**Web UI:** the **♾️ RSI Loop** tab. Load an example policy, pick a drive, and watch the *Ouroboros Lattice*:

- the loop as a ring with an ingest tail; an accent arc shows how far the current pass has gone;
- the policy as an outer band, with red chords between contradictory clauses;
- the gauge spokes running to the agents (red where a link is twisted);
- agents as plain dots, brighter with rank, red while leaking and green while stocking;
- controls appearing on the band in green as they are kept, and the clamped LoRA scale as an arc around the core.

Findings, controls, the hardened policy, the correspondences, the LoRA history and the loop log are listed below the canvas.

**CLI:**

```bash
deepthink rsi --input examples/security/institution_access_policy.txt --drive leak --iterations 3 --offline
deepthink rsi --input examples/security/institution_access_policy.txt \
              --repo examples/security_surface/repo \
              --social examples/security_surface/social_profile.json --offline   # code + people
deepthink rsi --input my_policy.md --drive stock --objective "protect finance data" \
              --driver-url http://localhost:8080/v1 --lora      # local driver, LoRA clamp
deepthink rsi --input my_policy.md --driver-url http://localhost:8080/v1 --lora               --lora-program --editor-model <router-id> --lora-adapters 0,1   # weight-space programming
deepthink estimate rsi --iterations 3
python -m deepthink.nomosloop.benchmark                         # every example × every drive
```

**Library:**

```python
import asyncio
from deepthink import run_rsi
from deepthink.models import ChatLlamaCpp

async def main():
    driver = ChatLlamaCpp(base_url="http://localhost:8080/v1", model="local")  # or None offline
    text = open("examples/security/network_segmentation_standard.txt", encoding="utf-8").read()
    r = await run_rsi(driver, text, params={"drive": "leak", "iterations": 3,
                                            "objective": "limit lateral movement"})
    for f in r["findings"][:3]:
        print(f["severity"], f["finding"])
    print(r["posture_baseline"], "->", r["posture_final"], r["artifact_path"])

asyncio.run(main())
```

**Benchmark** (offline, three loops, `python -m deepthink.nomosloop.benchmark`):

```text
case                               posture        contra  finds  ctrls    L_G  space    rps
incident_response_runb/leak       0.618 -> 0.829       0     15      4  0.283      3   2.84
incident_response_runb/stock      0.618 -> 0.946       0      6     16  0.332     38   0.82
institution_access_pol/leak       0.245 -> 0.286       2     19      3  0.037      1   2.18
institution_access_pol/stock      0.245 -> 0.320       2      9     17  0.159     43   0.96
network_segmentation_s/leak       0.146 -> 0.200       3     13      4  0.149     14   2.41
network_segmentation_s/stock      0.146 -> 0.232       3      3     15  0.276     47   0.39
```

Leak-driven runs produce the most findings and the lowest ensemble Leak. Stock-driven runs write the most controls, cover the most cells and need the fewest rounds per solve. In the access policy, the leak drive finds the two contradictions seeded into the example (§2 × §7 and §5 × §8).

### What is measured and what is not

- **Measured:** contradictions found by the parser, code and procedure facts and their status against the policy, each universe's posture, contradictions, ensemble leak, clash mass and bred structures against a baseline ticked as long, posture before and after, conservation of the tick law, gauge invariance, which controls pass the gate, rounds per solve, and the LoRA keep/rollback decisions.
- **Readings, not measurements:** the mapping of clauses to species, the climate field of names, and every treatise parameter.
- **Not claimed:** that a higher posture means a safer institution, that findings are complete (repository analysis is pattern based and misses what its detectors do not describe; the social layer only sees what the profile states), that a universe's verdict predicts what the change would do in the institution (it is what the change does to the text and the lattice), or that the LoRA adapter improves your model. Whether a trained adapter helps is for you to evaluate on your own traces.

---

## 🎰 Qualitative Diffusion (QDAD)

**An N×N noun×verb grid: high-temp invent, critic rewrite, then one synthesizer.** Available as:

1. **App Slot Machine Mode** in this app (full LangGraph engine + logs + matrix transparency)
2. **Portable `/qdad` skill** for Claude Code, Cursor, Codex, and other agentic coders — no server required

| Classical diffusion | Qualitative Diffusion (QDAD) |
|---------------------|------------------------------|
| Continuous noise in latent space | **Controlled qualitative noise** at high temperature |
| Denoising network | **Critic agents** (reverse diffusion / score matching in language) |
| Pixel / embedding basis | **Nouns × verbs** as orthogonal basis directions |
| Image from a vague prompt | **Buildable agentic coding prompt** from a vague aesthetic app prompt |

**How it works**

1. **Foundation** — From your Midjourney-style intent, sample **N nouns** and **N verbs** (shared qualitative basis).
2. **Grid** — Build an **N×N** grid of FeatureAgents. Cell *(i, j)* is permanently bound to `nouns[i] × verbs[j]`.
3. **Forward diffusion** — In parallel, each agent invents one wild, imperfect feature at **noise temperature**.
4. **Reverse diffusion** — For each denoising step, CriticAgents with the *same* noun+verb signature clean, sharpen, and implementabilize the matrix.
5. **Decode** — A Synthesizer collapses the clean matrix into a structured **# App Build Prompt** you can hand to Claude Code, Cursor, Codex, or any agentic coder.

**Philosophy (strict)**

- Language is the computational medium.
- Nouns and verbs act as orthogonal basis directions.
- High temperature = a sampling knob on the invent step.
- Critic agents rewrite the same cell (design-language: reverse diffusion / score matching).
- The whole process turns a vague aesthetic prompt into a concrete, buildable app specification. Quality versus one long prompt is unevaluated.

**Why a grid (not a flat feature list)?** Features sit on a forced tensor product of basis directions. Exploration is *systematic*. Critics cannot change the signature of a cell — they score-match *along* that noun×verb chart. Synthesis is a separate decode step, not “pick the best idea.”

| Run it… | Location |
|---------|----------|
| In-app | **App Slot Machine Mode** · `deepthink/qdad/` |
| In Claude Code / Cursor / Codex / any agent | [`skills/qdad/SKILL.md`](./skills/qdad/SKILL.md) · `/qdad` · release zip `qdad-skill-*.zip` |

---

## The Core Problem with "Just Add More Agents"

Typical multi-agent setups (including many "16 expert" or "army of agents" brainstorming interfaces) work like this:

- Spawn N agents with static or lightly templated personas.
- Run them in parallel or loose conversation.
- Synthesize once (or a few turns).
- Done.

You get diversity of perspective, but the agents themselves do not become meaningfully better at the *specific* problem over time. There is no topology, no persistent specialization, no mechanism that rewires *how* the system thinks, and almost never a reusable artifact of the reasoning process.

open-deepthink treats agents like **neurons in a network** whose "weights" are rich natural-language personas, and whose learning rule is **Mirror Descent** — an LLM rewriting those prompts (not gradient descent). See [`docs/DESIGN_NOTES.md`](./docs/DESIGN_NOTES.md).

---

## What Makes a QNN Different

A QNN is a directed, layered graph of LLM agents with three repeating phases per epoch:

1. **Forward Pass** — Problem is decomposed across the topology. Layer 0 runs in parallel. Each subsequent layer receives context from the previous layer and builds deeper analysis. Information flows structurally, not just through a shared chat.

2. **Qualitative Self-Attention** — Within each epoch, neurons do **not** only see graph neighbors. Each agent scores a capped pool of past / non-local neurons by **lexical overlap** (persona tokens vs past text) and injects the top-k excerpts. No MatMul, no extra LLM call. Inspired by [colony](https://github.com/iblameandrew/colony). Implementation: `deepthink/self_attention.py`. In Theory Synthesis the same idea reads the swarm's shared body of work instead of private memory.

3. **Reflection + Mirror Descent** — After synthesis, the system does not just "critique the answer." It:
   - Evaluates which agents struggled vs. succeeded on their specific sub-problems.
   - Extracts attributes and "hard requests" from current personas.
   - Uses a dense-spanner mechanism (or explicit mixing in Distillation) to **rewrite the system prompts, attributes, and skills** of agents for the next round.
   - In Knowledge Distillation mode, literally **spawns evolved child agents** that inherit context memory and replace struggling parents in the live topology.

4. **Problem Reframing** — A dedicated re-framer node looks at the current solution and formulates a *harder, more advanced version* of the problem. The network is then forced to solve the harder problem in the next epoch with its newly evolved agents.

This loop (decompose → structured forward **+ attend** → synthesize → reframe the goal → mutate the thinkers) is repeated for as many epochs as you allocate. The result is compounding depth rather than repeated breadth.

---

## Four Powerful Operating Modes (plus the classic QNN)

### 1. ⚗️ Knowledge Distillation Mode (The Data Engine)

The most distinctive and high-leverage mode.

- Fixed powerful topology: **1×2×2×2×2×2×1** (7 layers, 12 agents).
- 12 distinct cognitive archetypes (The Initiator, Builder, Connector, Preserver, Performer, Analyst, Diplomat, Transformer, Explorer, Architect, Visionary, Dreamer) with hand-crafted system prompts, attributes, and skills.
- **Task Master** decomposes the anchor question into 12 Socratically-linked sub-questions.
- Full forward pass with layer-to-layer context.
- **Mirror Descent** evaluates every agent-question pair. Hard agents trigger **live evolutionary replacement**: a Mixing Agent combines the struggling agent with the best resonant helper from the *current* grid. The child inherits the parent's 100k-token context memory and keeps the difficult question.
- **Seed Creator** evolves the topic set itself each epoch, generating ontologically adjacent new topics.
- Runs until your token budget is exhausted.

**Primary output**: A structured JSON dataset of every (epoch, agent, archetype, question, answer) pair, plus a complete `topology_archive.json` containing the full evolutionary history (every system prompt mutation, every inheritance, every difficulty judgment).

This is not generic chat logs. It is **structured developmental trace data** (QA pairs + topology archive). The intent is that traces like these could help train later models. That use is **unevaluated**.

### 2. ∑ Theory Synthesis (formerly Brainstorming Mode)

The web UI's second tab. A swarm of QNNs solves one problem in parallel: each QNN splits it into cells that are easy for its solvers and synthesizes the cells back into a proposal. Every cell writes into one shared body of work, and the swarm converges through a vote weighted by mirror descent. It is built for math problems, algorithm design, physics, and single instances of IMO-style problems. The mechanism, controls, CLI and library are in the [Theory Synthesis section](#-theory-synthesis-a-swarm-of-qnns-for-one-hard-problem) at the top.

#### The classic QNN (Solution-Space Report)

The tab used to run the classic QNN, which maps divergent strategies for an impasse instead of solving a problem. That pipeline is unchanged. It is still the engine of `run_qnn`, `deepthink qnn` and the portable `/qnn` skill, and API clients can still post `mode: "brainstorm"` to `/build_and_run_graph`.

**Step-by-step each run:**

0. **Brief** — Impasse/enrich summary from prompt + attachments  
1. **Topology** — Auto (complexity → L×W×E) or Manual/Massive  
2. **Seeds** — Verbs + nouns from the problem space (related + far semantic fields), sampled into per-column word-vectors — same spanning DNA as qualitative verb/noun bases  

3. **Personas** — Input-span careers/attributes/skills from those guiding_words (layer 0 diverge; deeper layers converge/critique)  
4. **Epoch loop** — Layered forward pass **with qualitative self-attention** → epoch map → Mirror Descent → harder reframe  
5. **Solution-Space Report** — Divergent strategies with falsifiers and first probes (handoff to edit→run→debug)

#### Self-attention inside the epoch (classic QNN, from colony QSA)

Feed-forward edges only connect a neuron to the **previous layer**. Self-attention adds a second path: each agent scores a capped pool of **non-neighbor past neurons** (skipped earlier layers this epoch + other agents’ multi-epoch memory) and injects the top‑k as a sparse attended-value block.

| Transformer | Brainstorm QSA (`deepthink/self_attention.py`) |
|-------------|------------------------------------------------|
| Tokens | QNN neurons (`agent_{layer}_{width}`) |
| Q / K | Query persona traits vs past solution text |
| Softmax | Strength buckets `none/low/med/high` + distance `near/mid/far` |
| V | Excerpt + rationale injected into the agent prompt |
| Attention matrix | `state["attention_edges"]` (per-agent edge lists) |

Logs: `LOG: [QNN ATTEND] agent_L_W self-attention → k non-local past neuron(s): …`

- **Auto mode**: Complexity estimator recommends a small topology; hard cap **24 agents**.
- **Manual mode**: You set L×W. CLI refuses runs estimated at >80 LLM calls unless `--yes` (or `--debug`).
- Intermediate epochs produce compact **epoch maps**; the final epoch polishes the full report.
- Rich markdown chat interface for the report and logs for each QNN step.

Use this when you want deeper insight than a single model or a flat expert panel can deliver.

### 3. 🎰 App Slot Machine Mode (Qualitative Diffusion App Designer)

Replaces Algorithm Design Mode with **Qualitative Diffusion App Designer (QDAD)** — diffusion re-implemented at a purely qualitative scale.

**Philosophy:** language = computational medium · nouns/verbs = orthogonal basis · high T = controlled noise · critics = reverse diffusion / score matching · vague prompt → buildable app spec (Midjourney for apps).

| Layer | Path |
|-------|------|
| Package | `deepthink/qdad/` (`state`, `nodes`, `graph`, `pipeline`, `utils`) |
| Chains | `deepthink/chains/qdad_chains.py` |
| GUI | Same chat shell as Theory Synthesis + N / Temperature Scale / Denoising Steps / Noun-Verb T |
| Graph | LangGraph: `foundation → grid → noise → denoise⟲ → synthesize` |

- **Phase 0** — N nouns + N verbs (shared qualitative basis, noun/verb temperature)
- **Phase 1** — N×N FeatureAgents permanently bound to `nouns[i] × verbs[j]`
- **Phase 2** — Parallel forward diffusion (noise temperature)
- **Phase 3** — Iterative reverse diffusion via CriticAgents (same signature)
- **Phase 4** — Synthesizer → structured **agentic coding prompt** + transparent matrix

Feed a Midjourney-style app prompt; hand the result to Claude Code, Cursor, Codex, or any agentic coder.

### 4. 🛡 RSI Loop (Recursive Self-Improvement for Cybersecurity)

Agent Mechanics agents cover an institution's security policy. Leak-driven runs find and heal contradictions; stock-driven runs evolve the policy with posture-gated controls; a LoRA loop distils every pass into the local driver model. The full mechanism, drives, models and benchmark are in the [RSI section](#-rsi-recursive-self-improvement-for-cybersecurity) at the top.

---

## The Outputs That Actually Matter

A single deep open-deepthink run produces far more than an answer:

- **Evolved QNN artifacts** — Portable, versionable "trained" multi-agent systems you can share and reuse.
- **Full evolutionary traces** — Every prompt before/after Mirror Descent, every difficulty classification, every child/parent relationship, every reframed problem.
- **Structured distillation datasets** — Purpose-built for fine-tuning or synthetic data pipelines targeting advanced reasoning and multi-agent behavior.
- **Interpretable intermediate state** — Because everything is explicit natural-language personas and traceable sub-problems, you can diagnose *why* the system thought what it thought at any layer and epoch.
- **A body of work** (Theory Synthesis) — Every lemma, case and computation the swarm wrote, with authors, dependencies, conflicts, arbitration verdicts and reuse.
- **Accumulated executable knowledge** (in code modes) — Real modules that survived sandbox validation and were re-used.

These artifacts are the real product. The final synthesized answer is a byproduct.

---

## Why This Matters for Agentic Coding and Reasoning Research

- **Test-time compute, done right and observably.** Many frontier systems hide their long reasoning inside a single model. open-deepthink makes the structure, specialization, and adaptation explicit and archivable.
- **Traces, not proven training data.** Distillation writes QA pairs and topology archives. Whether those traces improve a base model is unevaluated.
- **Reusable specialized reasoners.** An exported QNN stores evolved personas and structure. That is a different artifact from a single system prompt; we do not claim it is a better reasoner.
- **Local and long-horizon by design.** Small topologies (2×2×1, 3×3×2) fit a laptop. Larger runs are an explicit, estimated cost.

---

## Technical Strengths

- Built on **LangGraph** with cyclic graphs, parallel layers, and shared library engines for the UI, CLI, and skills.
- Phase test suite + control-flow tests (`python tests/run_all.py`) with mock LLMs — no API keys. Includes a free **structural eval** (`deepthink eval`).
- Clean provider model: only OpenRouter (cloud) and LlamaCpp / llama.cpp server (local).
- Robust JSON handling, token tracking, streaming logs, RAPTOR indexing, AST+subprocess sandbox, disk-backed sessions, and a pre-run **cost estimator**.
- Real export/import of QNN state. One `GraphState` (library = web).
- **Theory Synthesis** runs a swarm of QNNs under one concurrency cap and call budget. Its body of work merges, checks and arbitrates every result. The vote, the mirror-descent step, the ledger rules and the cost bound are checked in the test suite.
- **RSI** keeps a control only if the measured posture of the policy rises; the tick law's conservation and the gauge invariance of clash detection are checked in the test suite.
- Manual mode can grow large if you pass `--yes`; auto mode caps at 24 agents. Estimate first.

---

## Quick Start

### A. Python package (algorithms library)

Every mode ships as an importable package — **no web server required**.

```bash
git clone https://github.com/iblameandrew/open-deepthink
cd open-deepthink
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # macOS/Linux
pip install -e .                 # library: Theory Synthesis + QNN + QDAD + RSI + distillation
cp .env.example .env             # set OPENROUTER_API_KEY (or use llamacpp)
```

```python
import asyncio
from deepthink import create_llm, run_qnn, run_qdad, run_rsi, run_theory, DistillationGraph

async def main():
    llm = create_llm()  # OpenRouter from env, or create_llm(provider="llamacpp")

    # 0) Theory Synthesis → a swarm of QNNs solves one problem and builds a body of work
    theory = await run_theory(
        llm,
        "Find all positive integers n such that 3 divides 2^n + 1.",
        params={"domain": "olympiad", "swarm_size": 5, "rounds": 2},
    )
    print(theory["final_answer"], theory["converged"], theory["body_of_work"]["stats"])

    # 1) Qualitative Neural Network → Solution-Space Report
    qnn = await run_qnn(
        llm,
        "How do we break this ownership deadlock under concurrency?",
        params={"qnn_mode": "auto", "num_epochs": 2},
    )
    print(qnn["proposed_solution"])

    # 2) Qualitative Diffusion (QDAD) → App Build Prompt
    qdad = await run_qdad(
        llm,
        "cozy night writing app, soft dark mode, offline-first",
        params={"grid_size": 3, "denoising_steps": 2},
    )
    print(qdad)

    # 3) Knowledge Distillation → evolutionary dataset + topology archive
    graph = DistillationGraph(
        llm,
        topics=["concurrency", "ownership"],
        anchor_question="Design a latch-free ownership protocol",
        token_budget=100_000,
    )
    while graph.is_running:
        if not await graph.run_epoch():
            break
    print(len(graph.distilled_data), "QA pairs →", graph.dataset_path)

    # 4) RSI → findings + hardened policy (driver must be local; None = offline)
    policy = open("examples/security/institution_access_policy.txt", encoding="utf-8").read()
    rsi = await run_rsi(None, policy, params={"drive": "leak", "iterations": 2})
    print(len(rsi["findings"]), "findings; posture", rsi["posture_baseline"], "->", rsi["posture_final"])

asyncio.run(main())
```

**CLI (same library, no browser:**

```bash
deepthink theory --prompt "Find all n such that 3 divides 2^n + 1." --domain olympiad --yes
deepthink theory --prompt "…" --debug        # Simulation Mode: mock LLM, no API key
deepthink qnn  --prompt "Break this deadlock…" --verbose
deepthink qdad --prompt "cozy night writing app" --n 3 --steps 2
deepthink rsi  --input examples/security/institution_access_policy.txt --drive leak --offline
deepthink qnn  --prompt "…" --debug          # mock LLM, no API key
deepthink estimate theory --swarm 5 --rounds 2
deepthink estimate qnn --layers 2 --width 2 --epochs 1
deepthink estimate rsi --iterations 2
deepthink eval                               # structural checks, mock LLMs
deepthink version
```

| Algorithm | Import | What you get |
|-----------|--------|----------------|
| **Theory Synthesis** | `from deepthink import run_theory` | A swarm of QNNs: descent into easy cells, parallel solving, synthesis, a shared body of work, a mirror-descent vote → answer + solution |
| **QNN** | `from deepthink import run_qnn` | Layered multi-agent epochs + Mirror Descent + self-attention → Solution-Space Report |
| **QDAD** | `from deepthink import run_qdad` | Noun×verb diffusion grid → buildable App Build Prompt |
| **RSI** | `from deepthink import run_rsi` | Recursive self-improvement for cybersecurity → findings, posture-gated controls, hardened policy, LoRA traces |
| **Distillation** | `from deepthink import DistillationGraph` | 1×2×2×2×2×2×1 evolutionary topology → dataset + `topology_archive.json` |
| **Self-attention** | `from deepthink import compute_self_attention` | Qualitative attention over non-local past neurons |
| **Providers** | `from deepthink import create_llm` | OpenRouter or llama.cpp chat models |
| **Cost / eval** | `estimate_theory_cost`, `estimate_qnn_cost`, `estimate_rsi_cost`, `run_structural_eval` | Call-count estimate; mock structural eval (not a quality bench) |

Package layout:

```
deepthink/
  api.py              # high-level run_theory / run_qnn / run_qdad / run_rsi / run_distillation
  providers.py        # create_llm(...)
  theory/             # Theory Synthesis (pipeline, ledger = body of work, consensus, answers, domains)
  qnn/                # classic QNN pipeline (Solution-Space Report)
  qdad/               # qualitative diffusion
  nomosloop/          # RSI engine (mechanics/correspondence/world/security/lora/graph/…)
  distillation/       # evolutionary knowledge distillation
  self_attention.py   # lexical overlap “attention”
  cost.py             # LLM-call / token estimates (no network)
  eval_structural.py  # mock structural eval
  mocks.py            # CoderMockLLM / DistillationMockLLM
  sessions.py         # disk-backed session store
  rag.py              # RAPTOR index (web UI)
  runtime/            # log bus + leftover LangGraph nodes
  chains/             # all LangChain prompt factories
  config.py           # typed Settings (.env / env / TOML)
```

Full API surface: `from deepthink import …` (see `deepthink/__init__.py` and `deepthink/api.py`).

### B. Web app (optional UI)

```bash
pip install -e ".[web]"          # library + FastAPI UI
# or: pip install -r requirements.txt
open-deepthink                   # or: deepthink serve  |  python -m deepthink
# or: python app.py
```

Open http://127.0.0.1:8000.

**Supported providers**: OpenRouter (bring your own key) and LlamaCpp / llama.cpp server (local).

### Docker

```bash
cp .env.example .env   # set OPENROUTER_API_KEY
docker compose up --build
```

Volumes mount `distillation_output/`, `.deepthink-state/`, and `skills/` for persistence.

### Configuration

Typed settings live in `deepthink/config.py` (Pydantic Settings). Override via:

| Source | Example |
|--------|---------|
| Environment / `.env` | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `PORT` |
| Optional TOML | `deepthink.toml` or `OPEN_DEEPTHINK_CONFIG` |
| UI / request params | Still override per-run (backward compatible) |

Never commit real API keys. The web UI may store keys in browser localStorage only.

### Install extras

| Command | What you get |
|---------|----------------|
| `pip install -e .` | **Algorithms library** only |
| `pip install -e ".[web]"` | Library + FastAPI web UI |
| `pip install -e ".[dev]"` | Library + web + ruff/mypy/httpx/pytest |

### Tests (no API keys required)

```bash
pip install -e ".[dev]"
python tests/run_all.py
```

## License

MIT — see [LICENSE](./LICENSE).

---

## Hyperparameters & Hardware Reality

- **Layers / Width**: Start at 2×2 or 3×3. Auto mode will not exceed 24 agents. Manual is opt-in and expensive.
- **Epochs**: The number of full forward + reflection + reframing cycles. This is where the power law lives.
- **Learning rate / Density / Prompt alignment**: Control how aggressively agents mutate and how strongly the original problem shapes their identities.
- **Token budget** (Distillation): The real governor. Set it high if you want serious evolutionary runs.
- **Swarm size, rounds, depth** (Theory Synthesis): Cost grows with QNNs × rounds × cells per QNN. Start with 3–5 QNNs, depth 2 and 2 rounds, cap the run with *Max LLM calls*, and raise *Parallel calls* only as far as your provider's rate limit allows. On a local model, keep *Parallel calls* near the number of slots your llama.cpp server runs.

**Practical guidance**:
- 32 GB RAM CPU laptop: 2×2 to 4×4 topologies, 2–4 epochs.
- 64 GB + decent GPU: 6×6 to 10×10, more epochs, or serious Distillation runs.
- Design bet (unevaluated): more time and more epochs can beat needing a bigger base model. Measure cost with `deepthink estimate` first.

---

## Vision

Every serious open-deepthink run is a small laboratory experiment in collective intelligence. The structured traces it produces — complete with evolutionary dynamics, difficulty signals, and topology mutations — are some of the richest open data currently being generated about *how* LLMs can be orchestrated to think harder.

The long-term bet is that collecting thousands of such runs will let us train models that no longer need elaborate hand-written system prompts or external scaffolding, because they have internalized the patterns of decomposition, specialization, critique, and progressive deepening directly.

This is why the distillation dataset + full topology archives are treated as first-class outputs.

---

## Portable Skills for Agentic Coders (Claude Code, Cursor, Codex, etc.)

Drop these into your agent host’s skills root — e.g. `~/.claude/skills/` for Claude Code, or the equivalent rules/skills folder for Cursor, Codex, and other harnesses. **No open-deepthink server required.**

| Skill | Technique | Use when | Output |
|-------|-----------|----------|--------|
| **`/qdad`** | **Qualitative Diffusion** | Vague app / product *vibe* needs a full build brief | **App Build Prompt** (then implement) |
| **`/qnn`** | Qualitative Neural Network | Sticky debug **or** thin feature needs strategy depth | **Solution-Space Report** (then implement) |

### `/qdad` — Qualitative Diffusion (App Slot Machine in a skill)

```
/qdad a cozy productivity app for writers who work at night, soft dark mode, offline-first
/qdad N=3 steps=2 — garden-like habit tracker
```

Runs the full QDAD procedure — prefer **executing the engine**, not only simulating:

```bash
export OPEN_DEEPTHINK_ROOT=/path/to/open-deepthink
# Skill hosts materialize skills/qdad/run_template.py → .skill-runs/run_qdad.py
python skills/qdad/run_template.py --prompt "cozy night writing app…" --n 3 --denoising-steps 2
```

Library: `await run_qdad_pipeline(llm, params={...}, user_prompt=...)` — see [`skills/qdad/CODE_REFERENCE.md`](./skills/qdad/CODE_REFERENCE.md).

| Artifact | Location |
|----------|----------|
| Skill body | [`skills/qdad/SKILL.md`](./skills/qdad/SKILL.md) |
| Code contract | [`skills/qdad/CODE_REFERENCE.md`](./skills/qdad/CODE_REFERENCE.md) |
| CLI | [`skills/qdad/run_template.py`](./skills/qdad/run_template.py) |
| Engine | `deepthink.qdad.run_qdad_pipeline` |
| Release zip | `qdad-skill-<version>.zip` on [Releases](https://github.com/iblameandrew/open-deepthink/releases) |

### `/qnn` — Qualitative Neural Network escape hatch

```
/qnn explore this deadlock / performance regression
/qnn richer metrics for the training dashboard
```

```bash
python skills/qnn/run_template.py --prompt "explore this deadlock" --qnn-mode auto
```

Library: `await run_qnn_pipeline(llm, prompt, params={...})` — see [`skills/qnn/CODE_REFERENCE.md`](./skills/qnn/CODE_REFERENCE.md).

| Artifact | Location |
|----------|----------|
| Skill body | [`skills/qnn/SKILL.md`](./skills/qnn/SKILL.md) |
| Code contract | [`skills/qnn/CODE_REFERENCE.md`](./skills/qnn/CODE_REFERENCE.md) |
| CLI | [`skills/qnn/run_template.py`](./skills/qnn/run_template.py) |
| Engine | `deepthink.qnn.run_qnn_pipeline` |
| Release zip | `qnn-skill-<version>.zip` on [Releases](https://github.com/iblameandrew/open-deepthink/releases) |

### Install both (user-level skills)

```bash
# Linux / macOS — copy full skill folders (SKILL + CODE_REFERENCE + run_template.py)
# SKILLS_ROOT is your host's skills folder, e.g. ~/.claude/skills (Claude Code)
SKILLS_ROOT=~/.claude/skills
mkdir -p "$SKILLS_ROOT/qnn" "$SKILLS_ROOT/qdad"
cp skills/qnn/{SKILL.md,CODE_REFERENCE.md,run_template.py,INSTALL.md} "$SKILLS_ROOT/qnn/"
cp skills/qdad/{SKILL.md,CODE_REFERENCE.md,run_template.py,INSTALL.md} "$SKILLS_ROOT/qdad/"
export OPEN_DEEPTHINK_ROOT="$(pwd)"   # so runners import deepthink.*

# Or from release assets
gh release download --repo iblameandrew/open-deepthink --pattern "*-skill-*.zip"
unzip qnn-skill-*.zip -d "$SKILLS_ROOT"
unzip qdad-skill-*.zip -d "$SKILLS_ROOT"
```

Full skills index: [`skills/README.md`](./skills/README.md).

The full open-deepthink server remains the place for long evolutionary runs, App Slot Machine logs/matrices, export/import of trained QNNs, and distillation datasets. Portable skills are the lightweight escape hatches — with **runnable code entrypoints** when the package is available.

---

## Contributing & Benchmarking

This is research software. Quality versus single-shot prompting is unevaluated (`docs/EVAL.md`). The most valuable contributions right now are:

- Deep, long runs on interesting problems (especially with local models) and sharing the exported QNNs + distillation datasets.
- Bug reports that include the graph trace / logs.
- Ideas for tightening the code execution loop, adding real tool use inside agents, or improving the Mirror Descent signal.
- P2P/distributed ideas for running truly massive topologies across machines.

Open an issue with your traces and thoughts.

---

## License & Credits

Open-source research project. The goal is to push forward what small teams and individuals can do with structured, long-horizon agentic systems.

If open-deepthink helps you go deeper on hard problems or generates useful traces, star the repo and share what you built with the exported QNNs or distillation data.

---

**open-deepthink** — Turn time and structure into depth.  
Not more agents. Better *becoming* agents.

---

*Version 0.4.0 — See [RELEASE_NOTES.md](./RELEASE_NOTES.md) for the full history.*