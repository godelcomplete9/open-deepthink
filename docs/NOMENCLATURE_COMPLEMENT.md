# Nomenclature complement NX-1

**For:** *Agent Mechanics* (complete treatise), English edition, November 2026 (A. Ulloa de la Torre).

**Purpose:** decide whether the treatise's nomenclature is rich enough to name the agents, ensembles and events of the RSI engine, and complete it where it is not. The engine's use case is cybersecurity: the artifact is an institution's security policy.

**Status of this file:** a complement, not a revision. Everything here defers to the treatise's Canonical Glossary and Formal Apparatus. Where it adds something, the addition is marked **[C]** (a convention of this complement). Its thresholds are uncalibrated, like the treatise's own [OG6]. Page numbers are the printed page numbers of the treatise.

---

## 1. Verdict

The nomenclature is **rich in structure but not sufficient in explanatory power**. It is not sufficient for its own stated requirements, and it is not sufficient for driving a self-improving loop.

What it does well, and better than most agent taxonomies:

- **A class name that encodes a regime of current, not a personality.** `Rank-Orbit-Regime-Climate` says how much variety the agent holds, which writing is online, what it does with arriving flux, and where the field's mass sits. The step from one name to the next is legal only through a licensed operator paid for with energy (Part IV §5). This turns "evolution" into a checkable transition system.
- **Measures kept apart from class.** `/s /L /E /Ms /lam` (Part IV §4) compare two agents of the same class without inflating the class itself.
- **An ensemble language with an algebra.** Identifiers are stable and tombstones are never reused. The nine edge types form a closed set, and circuits carry storage signatures. Union, coupling, restriction, contraction-with-pointer, evolution and dissolution each have a checkable invariant (Part IV·bis §8). Illegal names exist, so names can be validated by a program.
- **A non-biological extension that already fits software.** It adds class `A`, the prosthesis-and-chain notation (`X -pr-> A`, `/end /aut /th`), and row A9 of Table 6, a deployed LLM read as "inert mass (weights) with a pattern; the query activates it". Policy clauses, security tooling and language models can be named without new species.

Where it falls short, in order of severity:

| # | Gap | Why it matters | Evidence in the book |
|---|---|---|---|
| G1 | **R4 (recomputability) does not hold from the string.** The name carries the support count `s`, not the vector `w`. | `W_G` and `Φ_G` cannot be recomputed from a canonical string, so the header is a checksum that cannot be checked. | Studio-6 needs Table 7 (vectors outside the name), p. 32. The Glossary admits that "the magnitude of `w_i`" is not covered, p. ii. |
| G2 | **R6 (canonical form) is undefined for the newer measures.** The canonical order is `/s /L /end /m /t /q /aut`, but `/E /Ms /lam /th` were added later without a slot. | Two programs can write the same agent as two strings. | IV·bis §7 (p. 29) versus Part IV §3 (p. 21) and §6 (p. 22). |
| G3 | **The grammar is not closed under its own extension.** `Brk` and `Path` are written as ranks, `TEAM` as an orbit and `L=med` as a measure, and signatures such as `Cyc3/mix` are written into the Regime field. None of these is in the grammar. | Names produced by the chain taxonomy fail the grammar of §5. | IV·bis §5 (p. 28) versus extension §6 (p. 36). |
| G4 | **Undefined operators and terminals.** The operators `σ` and `ω` are in the choice menu, and rank II is reached by `a∘σ`, but neither is defined. `σ` also names the storage signature, the task species sequence and an element of `K`. `ω` also names the Kuramoto natural frequency. The climate terminals `Ei, Nv, An/Rt, Ex/Pr, Bd` are used but never listed. | Promotion I→II cannot be executed mechanically. Climate cannot be computed. | Part I §8 (p. 18), Part IV Table 1 (p. 20), Part V §6 (pp. 50–51), O9 (p. 64). |
| G5 | **The online-writing mixture `π ∈ Δ(K)` is not in the name.** The Orbit field shows one side (`.a .r .ra`), but `|K_on|` is a component of `κ`. | A rank cannot be verified from its name. | Apparatus §2 (p. 3), Part IV §1 (p. 20). |
| G6 | **Leak is a single scalar.** `/L` is the fraction lost, with no species and no cause. The Apparatus already distinguishes overdrive `(1−χ)In⁺`, sub-quantum arrivals and hostile arrival `In⁻`. | Diagnosis is impossible: "leaks a lot" is not a finding. In security, the cause *is* the finding. | Apparatus §8 (p. 6). |
| G7 | **Stability `S` and stored gain `G` have no annotation.** The stock side of the leak/stock balance is invisible in names. | A loop driven by leak *and* stock needs both. | Apparatus §8–9. |
| G8 | **No time.** Names are snapshots. Chains (`→` with operator labels) are informal, and there is no tick stamp and no event grammar. | A self-improving loop is a process. Its log must be in the same language as its states. | Part IV §7 (p. 22), IV·bis ext. §7 (p. 37). |
| G9 | **One graph, one domain.** Identifiers are `Class Int` inside one ensemble. Nothing names the correspondence between an agent and the real artifact it stands for, beyond prose in Table 6. | The engine needs three domains (lattice, correspondence, artifact) and a map between them. | Part IV·ter §9.3: "a reading, not a measurement" (p. 44). |
| G10 | **Gauge objects have no notation.** Link variables, faces, curvature `B` and holonomy are defined in Apparatus §7 and Volume II §4, but no name or edge carries them. | Clash detection *is* a curvature statement. Without notation it cannot be logged or compared. | Apparatus §7 (pp. 5–6), Vol. II §4 (p. 69). |
| G11 | **Operations are not agents.** The treatise names entities and encounters ("the enzyme is an encounter", Part IV §2). An operation such as a refactor, a rule split or a patch has no atom. | The engine instantiates one agent per viable operation and per law modulus. | Part IV §2 (p. 21). |
| G12 | **Qualitative values are not machine-comparable.** `L=low/high/med` and `x≥1` mix bands and numbers. | Thresholds must be explicit to be swept as parameters. | Part IV §4, IV·bis §5. |
| G13 | **The word "leak" collides with the security term.** In the treatise, Leak is *arriving flux that is not stored* (heat, conflict). In security, a leak is data *leaving*. | A cybersecurity use without disambiguation will misread every report. | Apparatus §8. |
| G14 | **Ability and neurosis have no notation.** Volume II separates them as two regimes of one circuit, but circuits are named only by storage signature. | A loop that keeps healing without discharging is exactly the failure mode to detect. | Vol. II §5 (p. 70). |

The rest of this file closes G1–G14 as conventions, without touching the treatise's definitions.

---

## 2. Extended grammar (NX-1)

This grammar supersedes Part IV·bis §5 **for machine use only**. Every string legal under the book's grammar stays legal here. Terminals are in `code`.

```ebnf
Ens      ::= Title "[" Head "]" "{" List "|" Graph "|" Climate "}"
Head     ::= "n=" Int ";x=" Int ";lam=" Num ";r=" Int "/" Int "/" Int "/" Int "/" Int
             ";s=" Int ";L=" Num [ ";t=" Int ] [ ";R=" RProfile ] [ ";D=" Drive ]
List     ::= Atom { "," Atom }
Atom     ::= DomId "=" Name { Measure } "@" Num
DomId    ::= [ Domain ":" ] Class Int                       (* [C] domain prefix, G9 *)
Domain   ::= "AM" | "TC" | "AX"
Class    ::= "B" | "I" | "E" | "A" | "M" | "T" | "Q"
Name     ::= Rank "-" Orbit "-" Regime "-" Climate
Rank     ::= "Inr" | "Seed" | "Res" | "Cyc" | "Attr"         (* Brk and Path are NOT ranks, see §6 *)
Orbit    ::= FourLetters [ Flip ] | Stack | Double | "*"
Flip     ::= ".a" | ".r" | ".ra"
Stack    ::= Species { ">" Species }
Double   ::= FourLetters "|" FourLetters                     (* IV·ter §4.3 *)
Regime   ::= "cond" | "stem" | "orb" | "store4" | "pol4" | "leak" | "hp"
             | "Cyc" Int "/" Sig                              (* ext. §6, made legal *)
Sig      ::= "store" | "pol" | "mix" | "store-break" | "leak-store" | "frag"
Climate  ::= Domain3 OC | Terminal
Domain3  ::= "Tr" | "Fd" | "Fm"
OC       ::= "O" | "C"
Measure  ::= "/s=" Int | "/L=" LVal | "/S=" Num | "/G=" Num | "/Lk=" LeakSpec
           | "/w=" WVec | "/K=" KMix | "/E=" Num | "/Ms=" Num | "/lam=" Num
           | "/end=" Num | "/aut=" ("0"|"1") | "/th=" Num
           | "/m=" SpList ":" Label | "/t=" Type ":" Label | "/q=" Num
           | "/op=" OpName | "/role=" Role | "/Z=" Num | "/fat=" Num
LVal     ::= Num | "low" | "med" | "high"
WVec     ::= Species Num { "," Species Num }                  (* G1 *)
KMix     ::= Side ":" Num { "+" Side ":" Num }                (* G5 *)
Side     ::= "e" | "r" | "a" | "ra"
LeakSpec ::= Species ":" Cause [ Num ] { "+" Species ":" Cause [ Num ] }
Cause    ::= "op" | "ov" | "q"                               (* hostile, overdrive, sub-quantum *)
Species  ::= "Ni" | "Ne" | "Si" | "Se" | "Ti" | "Te" | "Fi" | "Fe"
Drive    ::= "leak" | "stock" | "bal"
RProfile ::= "schem(" Num "," Num "," Num "," Num ")"         (* cover, orbit, diagonal, self *)
```

### 2.1 Canonical order of measures (closes G2) [C]

`/s /L /S /G /Lk /w /K /E /Ms /lam /end /m /t /q /aut /th /op /role /Z /fat`

This is the book's order with each later measure placed next to the quantity it refines. Leak and stock come first, then the stock vector, the writings, energy and mass, then provenance and intake, then the RSI-only fields. Inside `/w` and `/Lk`, species are listed by decreasing value, with ties broken in the order `Ni Ne Si Se Ti Te Fi Fe`. Numbers use at most two decimals and no trailing zeros, as in IV·bis §7.

### 2.2 Bands for qualitative values (closes G12) [C]

| Value | Band | Read as |
|---|---|---|
| `L=low` | `L < 0.2` | almost everything that arrives is stored |
| `L=med` | `0.2 ≤ L < 0.5` | mixed; the name must record the heat (ext. §6 rule 4) |
| `L=high` | `L ≥ 0.5` | most arrivals become heat |
| `x` (from `/E`) | `x < 1` inert or sub-active · `x ≥ 1` emits (`γ ≥ ½`) | unchanged from Part IV §4 |
| `S` (from `/S`) | `S < S_min` leaking · `S ≥ S_min` storing | `S_min = 0.5` by IV·bis convention; a parameter in the engine |

A numeric `/L=` always wins over a band. When both are known, write the number.

### 2.3 Recomputability restored (closes G1)

An atom that carries `/w=` and `/K=`, inside an ensemble whose header carries `;R=`, is enough to recompute `W_G`, `Φ_G`, `s_G` and `κ_G` from the string alone (Apparatus §3, §6; IV·bis §6). `L_G` and the climate stay **measured**, as the book says (p. 33). The complement does not pretend otherwise. It only records the reading (§9).

---

## 3. Domains and identifiers (closes G9)

The RSI engine works in three domains linked by gauge fields. Every identifier may carry a domain prefix.

| Prefix | Domain | What lives there | Example |
|---|---|---|---|
| `AM:` | **Lattice** of agent mechanics | agents with the full tuple `(M, K, w, M_side, E, λ, ψ, Z, a)` | `AM:A7=Res-Ti>Si-stem-FdO/S=0.72/w=Ti2.1,Si1.4/K=e:0.8+a:0.2/E=1.2/end=0/aut=0/th=0.5/op=refine_split@0.6` |
| `TC:` | **Correspondence** (table of correspondences) | one row per mapped artifact element | `TC:12` |
| `AX:` | **Artifact**, the domain of origin: the security policy | clauses by level (L3 principles, L2 policies, L1 controls, L0 assets), level moduli, viable control operations | `AX:pol/L2/§2`, `AX:pol/L1#mod`, `AX:pol/L1/§R1` (a control added by the loop), `AX:op/add_control-Ne-1a2b` |

`AX:` references are paths, not integers. They are stable for as long as the artifact keeps its address. When an artifact element is removed, its references become tombstones, as in IV·bis R2. Inside a single domain the prefix may be omitted, so the book's strings stay valid.

---

## 4. Table of correspondences (closes G9, G11)

The book licenses a statistical model to read a functional profile and project it onto the spectrum (IV·ter §9.3). NX-1 turns that reading into a **row with a schema**, so that it can be audited, re-run and contested.

### 4.1 Row schema [C]

```text
TC:<n>  AX:<ref>  ⟼  AM:<id>=<Name>{Measures}
        basis    = <verbs>                      functional basis in the domain of origin
        p        = <Species Num, ...>           species projection of the basis (sums to 1)
        evidence = <Species: "quote", ...>      the words that gave each species away
        spectrum = <Class> (<why>)              substrate read from function (IV·ter Table 5)
        chain    = <edges>                      route of energy (X -pr-> A, X -> A -> Y)
        A        = <angle in [0, π]>            gauge link on the correspondence edge (§6)
        conf     = <0..1>                       mapper confidence
        mapper   = <model id> [ARC-AGI-2=<declared score>]
```

Rules:

- **X1.** Every element of the artifact's functional basis (each level modulus, each clause, each viable control operation) has at least one row. A row is never deleted. A superseded row becomes a tombstone `TC:<n>!x`.
- **X2.** `p` is a probability vector over the eight species. `/w=` of the instantiated agent is `p` scaled by the agent's mass budget, so `p` and `w` are not the same object (Glossary: `w` is stock).
- **X3.** `evidence` must quote the domain of origin. A row without evidence is legal but has `conf ≤ 0.3`.
- **X4.** The substrate is read from function, never declared first (IV·ter §9.2).
- **X5.** The mapper is recorded on every row. The engine recommends a mapper scoring **above 70 on ARC-AGI-2**, because the projection is an abstraction-transfer task. The score is *declared by the operator*, not verified by the engine. A row mapped by a model below the recommendation, or with an undeclared score, is legal and is flagged.

### 4.2 Functional basis → species lexicon [C]

This is the lexicon the offline mapper uses and the online mapper is prompted with. It reads Part VI §3 (each species as a field operation) into the verbs of security policy and the security function each species stands for.

| Species | Field operation (Part VI) | Verbs in a policy | Security function |
|---|---|---|---|
| **Ni** | geodesic of `A`; minimal-action packet | converges, implies, forecasts, determines | threat forecasting, attack-path analysis |
| **Ne** | `div E > 0`; tunnelling, branching modes | branches, explores, admits variants | red-team testing, discovery of unknown paths |
| **Si** | `∮A·dl` on a loop; memory solenoid | preserves, records, retains, baselines | audit logging, retention, backups, baselines |
| **Se** | Poynting `E×B`; present bandwidth | acts, monitors, detects, transmits | real-time monitoring, alerting, live response |
| **Ti** | Gauss law `∇²φ = −ρ`; rejects contradictions | defines, constrains, validates, closes | access rules, input validation, firewall logic |
| **Te** | work `J·E`; ordered current | executes, remediates, patches, measures | incident response, patch execution |
| **Fi** | dipole `−p·E`; bound state, restoring force | protects, binds, must not, denies | identity, integrity, least privilege, cryptography |
| **Fe** | Kuramoto synchronisation | coordinates, notifies, propagates | coordination, notification, policy rollout |

The lexicon is a reading under the book's own license. It is falsifiable row by row (IV·ter §10.4), not a measurement.

### 4.3 What gets an agent (closes G11)

- **Every level modulus** → one agent. A modulus closes a scale (a cycle, a wrap, a constraint), so its basis loads on Ti (closure) and Si (conservation). Its regime tends to `CycN/store` when the scale's rules are consistent, and to `leak` when they clash.
- **Every clause** → one agent, read from its function (class `A`: a pattern entrusted by the institution, energised by traversal).
- **Every viable operation** → one agent of class `A`, written `/op=<name>`. In the security engine the viable operations are one *add control* per under-covered function (a function whose share of the policy is below an even 1/8). It is a prosthesis of the agent that licenses it, written `X -pr-> A`, and it is energised only by traversal (Apparatus §14.6). A non-viable operation is never instantiated.
- **Every control the loop keeps** → one agent, born by a `pr` edge from the stocking agent whose surplus paid for it ("Birth is a pr edge plus a new atom", IV·bis Table 4).
- An operation agent is tombstoned when its operation is applied or rolled back. Its identifier is never reused.

---

## 5. Operators: definitions the book leaves open (closes G4)

| Operator | Book status | NX-1 operational definition [C] | Licensed when |
|---|---|---|---|
| `e` | defined | keep the given writing | always |
| `a` | defined (`α`, shadow) | invert attitude: sources ↔ sinks | the new axis has energy `x ≥ 1` (Part IV §5) |
| `r` | defined (`β`, subconscious) | cover the axis complement; the hero meets the inferior | a partner stores on the complement |
| `ra` | defined (`αβ = CP`, superego) | publish to the norm axis | the field asks two open questions at once (Part I §7) |
| **`σ`** | **used, undefined** | **stem licensing:** open a storing `st` edge with an agent whose hero covers this agent's weakest valued slot (`R_{f,β(f)} > 0`), so that the I→II step `a∘σ` is "flip, then stem" | the stem stores at the receiver, `S_j(g) ≥ S_min` |
| **`ω`** | **used, undefined** | **off-license emission:** emitting into a channel with `R < 0` or onto an overdriven receiver. It produces heat, never rank ("play off-license and burn mass", Part V §8) | never licensed; recorded as an event so the heat is accounted for |
| `η_g` | defined (hairpin) | fold `M` onto `g·M` with a Klein-close child | Part IV §5, III→IV |
| `→` | defined (directed arrow) | storing arrow | stores at the receiver |
| `sum` | listed, undefined | union `⊕` of IV·bis §8 | always (algebra) |

To remove the symbol clashes, the storage signature keeps `σ` only inside `Cyc<n>/<Sig>`, where it is spelled out, and the task sequence of Volume II is written `τ`.

### 5.1 Climate reading (G4, pending O9) [C]

O9 is open, so the climate is a **reading**, not a derivation. The engine reads it as follows and labels it as such:

- `Tr` (TRACE: the group's own time) when the argmax of `W_G` is in `{Ni, Si}`.
- `Fd` (FIELD: the batch) when it is in `{Ti, Te, Fi, Fe}`.
- `Fm` (FORM: the figure of the state) when it is in `{Ne, Se}`.
- `O` (open) when the ensemble holds at least one `store` circuit with `C⁺ > 0`, and `C` (closed) otherwise.

This reproduces both worked readings of the book: Studio-6 has argmax Ti and a store4 ring, giving `FdO` (p. 33), and the Part IV example leaves `TrC` after integrating Ti and Fe. The named terminals (`Ei, Nv, An/Rt, Ex/Pr, Bd`) are not defined in this edition. NX-1 does not invent them. A terminal reading is written verbatim and excluded from computation.

---

## 6. Gauge objects and the symbolic coderivative (closes G10)

### 6.1 The cross-domain complex

Take the cell complex of Apparatus §7 and extend it across domains [C]:

- **Vertices** `V = V_AM ∪ V_AX`.
- **Edges:**
  - `E_AM`, the attention graph of the lattice.
  - `E_AX`, relations in the domain of origin: *expands-to* between levels, *closes* from a modulus to its clauses, *targets* from an operation to its level, and *contradicts* between policy or control clauses that share a subject with opposite polarity.
  - `E_TC`, one edge per correspondence row `AX:a ↔ AM:i`.
- **Faces:** every square `□ = (a₁ → a₂ ⇝ i₂ ← i₁ ⇜ a₁)` where `a₁ → a₂ ∈ E_AX`, `i₁ → i₂ ∈ E_AM`, and both rows exist. A face says "these two artifacts are related, and so are their agents".

### 6.2 Link variables [C]

Each edge carries a phase `A(e) ∈ [0, π]` and a link variable `U(e) = e^{iA(e)}`. This is the book's `U(1)` link with `q/ℏ = 1` (Apparatus §7).

| Edge | `A(e)` | Reading |
|---|---|---|
| correspondence `a ↔ i` | `arccos(cos(p_a, w_i))` | how faithfully the agent carries the artifact's function |
| lattice `i → j` | `0` if `m(i,j) ≥ 0`, `π` if `m(i,j) < 0` | cooperative versus hostile encounter of the species (Apparatus §3) |
| artifact `a₁ → a₂` | `0` if consistent, `π` if contradictory | e.g. *granted* versus *denied* on the same share |

The current between two agents is **parallel-transported** along their elements' artifact relation: the attention edge `i → j` carries the phase of `a₁ → a₂`, and matter couples through `cos A` (Apparatus §7). A contradiction therefore arrives as hostile current, which is Leak of cause `op`. The face phase of the lattice edge, by contrast, is the sign of the encounter mass, so the two sides of a face are measured independently.

### 6.3 Curvature and clash

The curvature on a face is `B(□) = (d₁A)(□)`, the signed sum around the square. Its **frustration** is

```text
f(□) = (1 − cos B(□)) / 2   ∈ [0, 1]
```

`f ≈ 1` means the correspondences around the face cannot all hold at once. Either the artifacts contradict each other while their agents cooperate, or the artifacts cooperate while their agents are hostile. **That is a clash, and it is gauge invariant:** for any `χ` on vertices, `A ↦ A + d₀χ` leaves `B` unchanged because `d₁d₀ = 0` (Apparatus §7). Re-labelling which side of a correspondence is "positive" can never create or erase a clash. The engine tests this invariance.

Notation:

- `F(a₁,a₂;i₁,i₂)` names a face.
- `B=` and `f=` are its curvature and frustration.
- `Hol(γ)=n` is the winding of a lattice cycle (Volume II §4).

### 6.4 The symbolic gauge coderivative

Changes are solved in the lattice, where the laws are explicit, and then carried to the artifact. Let `ΔΨ` be a change at a lattice vertex. It is either a numeric change `Δw_i` or a licensed operator `o ∈ {a, r, ra, σ, η, →, cut}` applied to agent `i`. The coderivative transports it along correspondence edges:

```text
(δ_A ΔΨ)(a) = Σ_{rows a↔i}  ½(1 + cos A(a↔i)) · 𝓜[o_i]  ⊖  ½(1 − cos A(a↔i)) · 𝓜[o_i]⁻¹
```

It is the codifferential of the correspondence coboundary (`δ = M₀⁻¹ d_TCᵀ M₁`, as `div` in Apparatus §7), twisted by the link variables. Aligned rows (`A ≈ 0`) carry the edit, anti-aligned rows (`A ≈ π`) carry its inverse, and the weights interpolate in between. It is **symbolic** because `𝓜` maps operators to edit *symbols* rather than numbers:

| Treatise operator `o` | `𝓜[o]` on a security policy |
|---|---|
| `a` (sources ↔ sinks) | add the dual check on the opposite side (ingress ↔ egress) |
| `r` (axis complement) | cover the complementary function (e.g. Fi-integrity without Te-response → add a response playbook) |
| `ra` (norm axis) | promote the local control to policy |
| `σ` (stem) | cover the weakest function |
| `η` (hairpin) | consolidate overlapping controls into one standard |
| `→` (arrow, overload) | split the overloaded scope |
| `→` ×4 storing (`store4`) | close the loop: log, alert, review |
| `cut` (shielding, `w_border → 0`) | isolate the clashing scope behind identity and least privilege |
| `ω` (off-license) | nothing; record heat |

This is the "hyperparameter trick" the engine relies on. Permutation problems are hard in the domain of origin, because there are too many edits to try. In the lattice they are small and lawful: 16 valid stacks, 4 writings and a handful of licensed operators. The engine searches in the lattice and pulls the winner back through `δ_A`.

---

## 7. Leak, stock and events (closes G6, G7, G8, G13, G14)

### 7.1 Leak diagnosis by cause

`/Lk=` resolves the Apparatus §8 decomposition by species and cause:

| Cause | Apparatus term | Reading in software and security [C] |
|---|---|---|
| `op` (hostile) | `In⁻ = w·D⁻` | contradictory input: two artifacts that clash, an adversarial message, a policy conflict |
| `ov` (overdrive) | `(1−χ)In⁺` with `V⁺ > 1` | more valid input than capacity: saturation, alert fatigue, denial of service |
| `q` (sub-quantum) | `In⁺,<q` | arrivals below the registration quantum: low-and-slow probing, events below the logging threshold |

So `AM:A3=…/L=0.62/Lk=Fi:op0.41+Se:ov0.12/…` reads: *62 % of what arrives is not stored, mostly hostile current on the integrity function and some overdrive on monitoring.*

### 7.2 Disambiguation (closes G13)

In this complement, **Leak** (capitalised, treatise sense) always means *arriving flux that is not stored*. The security sense, data leaving the institution, is written **exfiltration** and is read as unwanted *emission* `O_i` on a species, never as Leak. The user-level phrase "an agent leaks" means *Leak at that agent exceeds its stock gain*. Its typical cause in the engine is a clash (`op`) between the artifacts its correspondence rows point to.

### 7.3 Stock

`/S=` is stability `G/(G+Leak)` and `/G=` is stored flux per tick (Apparatus §8). An agent is **stocking** when `Δŵ ≥ 0` (on the total reserve, Corrigenda A.1) and `x ≥ 1`: it has a surplus that emits. An agent is **leaking** when `S < S_min`. Both can hold briefly for one agent. The engine's drive setting decides which one wins (§8).

### 7.4 Event grammar (closes G8) [C]

```ebnf
Event   ::= "t=" Int " " DomId " " Act
Act     ::= "leak[" LeakSpec "]" [ " heal(" Op ")" " => " Name ]
          | "stock[" Num "]" " evolve(" EditSym ")" " => " AXRef
          | "pr -> " DomId
          | "Brk(" Circuit "->" Sig ")"
          | "clash F(" AXRef "," AXRef ";" DomId "," DomId ") f=" Num
          | "clamp(lora=" Label ",scale=" Num ")"
          | "omega[" Species "]"
          | "!x"
Op      ::= "a" | "r" | "ra" | "sigma" | "eta" | "cut" | "arrow"
```

Examples:

```text
t=3 AM:A3 leak[Fi:op0.41] heal(r) => Res-Fi>Se-stem-FdO/S=0.58
t=3 AM:A3 clash F(AX:pol/L2/§2,AX:pol/L2/§7;AM:A3,AM:A5) f=0.97
t=4 AM:A9 stock[0.8] evolve(r: real-time monitoring and alerting) => AX:pol/L1/§R1
t=4 AM:A9 pr -> AM:A12
t=5 AM:I1 Brk(Cyc4/store4->op(A3,A5))
t=6 AM:A9 clamp(lora=heal-0005,scale=0.35)
```

A chain in the book's sense (Part IV §7) is the projection of an event log onto one identifier.

### 7.5 Ability and neurosis on circuits (closes G14)

A circuit name may end with `/reg=abl` or `/reg=neu`, following Volume II Table 2:

- `abl` (ability): `ι_S ≥ (1+ζ)L_S + W_S` with `W_S > 0`.
- `neu` (neurosis): sustained `L_S ≥ λ*`, `W_S ≈ 0`, and either non-zero winding or insufficient intake.

In the engine, an ensemble that keeps healing without exporting any artifact improvement is `neu`. The loop raises its intake or cuts it, instead of feeding it more heal cycles.

---

## 8. Drive and sensitivity (ensemble header `;D=`)

The header field `;D=` records which balance drives the run. Its parameters are the treatise's own; the presets are conventions [C].

| Drive | Intent | `S_min` | `ζ` (leak damage) | heal budget | stock budget | transduction `θ` |
|---|---|---|---|---|---|---|
| `leak` | defensive: find and heal clashes first | high (0.62) | high | high | low | low |
| `stock` | evolving: spend surplus on new controls | low (0.38) | low | low | high | high |
| `bal` | both, by measured balance | 0.5 | medium | medium | medium | medium |

A single scalar **`λ_AM ∈ [0, 1]`** sets how sensitive the lattice is to the treatise. At `λ_AM = 0` decisions follow artifact fitness alone. At `λ_AM = 1` they follow the lattice balance alone. In between, the two are blended. The other knobs (`h, E₀, c, q, R` scale and antisymmetry, `θ, θ_K`) are exposed unchanged.

---

## 9. LoRA as side mass [C]

A LoRA adapter trained on the loop's own traces is read in the book's language. It is inert mass (weights) entrusted to the driver model: `AM:L1 -pr-> AM:A9`, where A9 is the local LLM of Table 6. Its effect is the Apparatus §14.4 law: **mass is accumulated stock, and it resists change.** The traces are routes that used to be paid for every time, such as heal chains and successful artifact edits. Folding them into the weights lowers the edge cost of the routes that are used most (Volume II §7). A shorter route in the band graph is fewer chain-of-thought rounds.

- The clamp scale is written `clamp(lora=<label>,scale=<s>)`.
- It is raised only when the loop measurably improves, meaning fewer rounds per solve, lower `L_G`, or higher artifact capacity.
- It is restored to its previous value otherwise: the same keep/rollback gate as the lattice.

**Weight-space programming [C].** An editing model may decide how the side mass is formed instead of a fixed clamp step. Its program edits the trace dataset (which routes are folded in), sets the adapter's training spec, and writes a mix of loaded adapters: `clamp(lora=<l1>,scale=<s1>; lora=<l2>,scale=<s2>)`. In the book's terms, the mix decides how much of each stock of side mass the driver carries. The program is a proposal, not a licence: the engine clamps it to the legal ranges and logs every correction. The same keep/rollback gate decides whether it stays.

---

## 10. Space, time and coverage (for a defended institution)

- **Covered cell:** a pair `(AX element, species)` carried by at least one agent with `S ≥ S_min` and `x ≥ 1`.
- **Space** `Σ_t`: the number of covered cells at tick `t`.
- **Permutations:** the agents' writings range over the 16 valid stacks × 4 sides. Each licensed flip can open new species on the same artifact element.
- **Time:** ticks `t` and chain-of-thought rounds `ρ_t` spent per solve.
- **Fortification rate** [C]: `Δ Σ_t / (Δt · ρ_t)`. Coverage gained per tick per thinking round. Evolving faster (more ticks) and thinking shorter (LoRA) both raise it.

This is a bookkeeping definition, not a claim of protection. Coverage means that some agent stands for that function and is currently stable. It does not mean the function is secure.

---

## 11. Worked example: a small institution

The artifact is an access-control and logging policy with three clauses. §2 grants staff read access to the `/finance` share at all times. §7 denies all access to the share outside business hours. §10 logs every access. Staff access outside business hours is a contradiction between §2 and §7. (This is a reduced version of `examples/security/institution_access_policy.txt`.)

```text
TC:1  AX:pol/L2/§2  ⟼ AM:A1=Seed-Ti>Fi-cond-FdC/S=0.71/w=Ti1.2,Fi0.8/end=0.25/aut=0
      basis=constrains,protects  p=Ti0.6,Fi0.4  evidence=Ti:"granted read access",Fi:"staff"
      spectrum=A (clause; energy by traversal)  chain=I1 -pr-> A1  A=0.12  conf=0.8
TC:2  AX:pol/L2/§7  ⟼ AM:A2=Seed-Fi>Ti-cond-FdC/S=0.44/Lk=Fi:op0.38/w=Fi1.3,Ti0.7/end=0.25/aut=0
      basis=protects,constrains  p=Fi0.7,Ti0.3  evidence=Fi:"is denied outside"  A=0.15  conf=0.8
TC:3  AX:pol/L1/§10 ⟼ AM:A3=Res-Si>Se-stem-TrO/S=0.83/w=Si1.5,Se0.9/end=0.25/aut=0
      basis=preserves,acts  p=Si0.7,Se0.3  evidence=Si:"is logged and retained"  A=0.1  conf=0.9

Inst-3[n=3;x=0;lam=1.8;r=0/2/1/0/0;s=4;L=0.29;t=2;R=schem(2,1,-2,0);D=leak]{
  A1=Seed-Ti>Fi-cond-FdC/S=0.71/w=Ti1.2,Fi0.8/end=0.25/aut=0@0.6,
  A2=Seed-Fi>Ti-cond-FdC/S=0.44/Lk=Fi:op0.38/w=Fi1.3,Ti0.7/end=0.25/aut=0@0.6,
  A3=Res-Si>Se-stem-TrO/S=0.83/w=Si1.5,Se0.9/end=0.25/aut=0@0.6
  |
  A1 -op[Fi]-> A2; A3 -fl[Si]-> A1
  |FdC}

t=2 AM:A1 clash F(AX:pol/L2/§2,AX:pol/L2/§7;AM:A1,AM:A2) f=0.98
t=2 AM:A2 leak[Fi:op0.38] heal(cut) => Seed-Fi>Ti-cond-FdC/S=0.62
t=3 AM:A3 stock[0.6] evolve(r: real-time monitoring and alerting) => AX:pol/L1/§R1  posture 0.245->0.271
t=3 AM:A3 pr -> AM:A4
```

How to read it:

- The artifact edge §2 → §7 is contradictory (`A = π`). Transported onto the attention edge, it makes the current from A1 arrive at A2 as hostile current, hence the typed edge `A1 -op[Fi]-> A2` and A2's Leak of cause `op`.
- On the lattice side of the face the two agents still cooperate: they share Ti and Fi, so their encounter mass is positive and the phase is 0. Exactly one side of the square is twisted, so `B ≈ π` and the frustration is near 1. The clash is found in the lattice and named in the policy. Had the mapper put §2 and §7 on hostile species, both sides would be twisted and the face would carry no clash: the lattice would already model the opposition.
- A2 heals by `cut`. Pulled back through `δ_A`, that means "isolate the clashing scope behind identity and least privilege", which is the inspection finding.
- A3, the logging control, is stable and has a surplus. It stocks by `r` (memory `Si` covers its complement): the loop drafts a monitoring control, the posture gate accepts it because function diversity rises, and the control is born as agent A4.

---

## 12. Legality rules added

- **X6.** An atom with `/Lk=` must have `/L` at least the sum of its parts.
- **X7.** `heal(o)` in an event must name a licensed operator (§5). `ω` appears only as `omega[...]`.
- **X8.** `stock[...]` requires `x ≥ 1` and `Δŵ ≥ 0` on the agent at that tick.
- **X9.** A `clash` event must name a face whose four edges exist.
- **X10.** `clamp` scales never exceed the configured maximum, and a clamp that is not followed by measured improvement within the evaluation window must be reverted by a later `clamp` event.
- **X11.** `AX:` references in a canonical string are sorted lexicographically after `AM:` atoms. Domain prefixes never change an identifier's integer.

---

## 13. What stays open

NX-1 makes the nomenclature executable. It does not settle the treatise's open obligations:

- The climate is read, not derived (O9).
- `R`, `w` and `E` are uncalibrated (O6, O7, O16).
- The tick law's stability is unproven (O2).
- The species lexicon of §4.2 is a reading that real flows may contradict.

Every number an engine prints under this nomenclature is an output of the model, not a measurement of the world.
