/**
 * RsiLattice — "Ouroboros Lattice" visualisation for the RSI loop.
 *
 *   • Outer ring  = the self-mutable LangGraph loop. The five looping stages
 *                   (mutation_proposal → mesh_spawn → fitness → artifact_emit
 *                   → rectify) sit on a circle; ingest and scale_induction
 *                   arrive along a spiral tail that feeds into the ring, so
 *                   the whole graph reads as a serpent swallowing its tail.
 *   • Comet       = the active node; it only ever travels forward.
 *   • Echo orbits = one faint ring per completed iteration (recursion depth).
 *   • Mandala     = the scale lattice. The most compact level sits at the
 *                   centre; each lower level is a wider ring, carrying its
 *                   mesh instantiations. "expands-to" links run outward,
 *                   spawned peers/children are dashed, and every node hosts
 *                   orbiting inhabitant agents (amber = spawned offspring).
 *   • Swarm       = mutation proposals: born at mutation_proposal, judged at
 *                   fitness, absorbed into the lattice at rectify (or shed).
 *
 * Events are queued and replayed with a minimum dwell so the offline engine
 * (which finishes in milliseconds) still reads as a loop.
 */
const RsiLattice = (() => {
    'use strict';

    const LOOP = ['mutation_proposal', 'mesh_spawn', 'fitness', 'artifact_emit', 'rectify'];
    const TAIL = ['ingest', 'scale_induction'];
    const STAGE_META = {
        ingest: { label: 'ingest', hint: 'tokenising the formal artifact' },
        scale_induction: { label: 'scale induction', hint: 'mining motifs → inducing the scale lattice' },
        mutation_proposal: { label: 'mutate', hint: 'spawning the mutator swarm' },
        mesh_spawn: { label: 'mesh spawn', hint: 'materialising instantiations · agents inhabit the mesh' },
        fitness: { label: 'fitness', hint: 'measuring capacity Δ against the holdout gate' },
        artifact_emit: { label: 'artifact', hint: 'emitting a runnable artifact' },
        rectify: { label: 'rectify', hint: 'keep / rollback · elevating the lattice' },
    };
    const COL = {
        violet: [139, 92, 246],
        blue: [59, 130, 246],
        cyan: [6, 182, 212],
        green: [52, 211, 153],
        red: [239, 68, 68],
        amber: [251, 191, 36],
        white: [255, 255, 255],
    };
    const DWELL_MS = 620;
    const TAU = Math.PI * 2;

    const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
    const mix = (a, b, t) => [0, 1, 2].map(i => Math.round(a[i] + (b[i] - a[i]) * t));
    const ease = t => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);
    const now = () => performance.now();

    function levelColor(i, n) {
        const t = n <= 1 ? 0 : i / (n - 1);
        return t < 0.5 ? mix(COL.violet, COL.blue, t * 2) : mix(COL.blue, COL.cyan, (t - 0.5) * 2);
    }

    // ── state ────────────────────────────────────────────────────────────
    let mount, canvas, ctx, tip, hud = {};
    let W = 800, H = 560, cx = 400, cy = 280, R = 200, dpr = 1;
    let raf = null, t0 = now();
    let onState = null, onStage = null;

    let stages = {};          // name -> {x, y, angle, glow, done}
    let tailPts = [];         // spiral polyline: ingest → scale_induction → ring
    let active = null;
    let comet = null;         // {mode:'tail'|'ring', s, ts, a, ta}
    let trail = [];
    let levels = [];          // [{id, name, rules:[...] , color, born}]
    let nodes = new Map();    // id -> {id, level, rule, born, dying}
    let edges = [];
    let agents = [];
    let props = new Map();    // id -> {id, status, x, y, alpha, absorbed, shed}
    let absorbed = new Set();
    let shed = new Set();
    let ripples = [];
    let echoes = [];          // completed iteration count
    let flashes = [];         // {stage, color, t}
    let queue = [], pumping = false, lastPlay = 0;
    let version = null, iteration = 0, maxIter = 1;
    let hover = null, mouse = null;
    let running = false;

    // ── geometry ─────────────────────────────────────────────────────────
    function layout() {
        const rect = mount.getBoundingClientRect();
        W = Math.max(280, rect.width);
        H = Math.max(320, rect.height);
        dpr = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = Math.floor(W * dpr);
        canvas.height = Math.floor(H * dpr);
        canvas.style.width = W + 'px';
        canvas.style.height = H + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        R = Math.min(H * 0.39, (W / 2 - 24) / 1.5);
        cx = W / 2 + R * 0.16;
        cy = H / 2 + 8;

        LOOP.forEach((name, i) => {
            const a = -Math.PI / 2 + i * TAU / LOOP.length;
            stages[name] = { ...(stages[name] || { glow: 0, done: 0 }), x: cx + Math.cos(a) * R, y: cy + Math.sin(a) * R, angle: a };
        });

        // Spiral tail: r grows as θ winds back from the ring entry at -90°.
        tailPts = [];
        const th0 = -Math.PI / 2, th1 = -Math.PI * 0.985, k = 0.48;
        for (let i = 0; i <= 80; i++) {
            const s = i / 80;
            const th = th1 + (th0 - th1) * s;
            const r = R * (1 + k * (th0 - th) / (th0 - th1));
            tailPts.push({ x: cx + Math.cos(th) * r, y: cy + Math.sin(th) * r, s });
        }
        const at = s => tailPts[Math.round(s * 80)];
        stages.ingest = { ...(stages.ingest || { glow: 0, done: 0 }), ...at(0), angle: th1, s: 0 };
        stages.scale_induction = { ...(stages.scale_induction || { glow: 0, done: 0 }), ...at(0.52), angle: th1, s: 0.52 };
    }

    function levelRadius(i, n) {
        if (n <= 1) return R * 0.4;
        return R * (0.13 + 0.6 * i / (n - 1));
    }

    function rot() { return (now() - t0) / 1000 * 0.035; }

    function nodePos(n) {
        const li = levels.findIndex(l => l.id === n.level);
        if (li < 0) return null;
        const peers = n._peers;
        const r = levelRadius(li, levels.length) + (n.gen || 0) * 7;
        const a = rot() + ((peers.offset || 0) + peers.idx / peers.count) * TAU;
        return { x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r, li };
    }

    /**
     * Radial-tree ordering: the top level is spread evenly; each lower ring
     * is ordered by the mean angle of its parents, so expansion links fan
     * outward instead of crossing the mandala.
     */
    function reindexNodes() {
        const byLevel = new Map();
        for (const n of nodes.values()) {
            if (!byLevel.has(n.level)) byLevel.set(n.level, []);
            byLevel.get(n.level).push(n);
        }
        const parents = new Map();
        edges.forEach(([s, d]) => {
            if (!parents.has(d)) parents.set(d, []);
            parents.get(d).push(s);
        });
        const order = levels.map(l => l.id).concat([...byLevel.keys()].filter(k => !levels.some(l => l.id === k)));
        const angleOf = new Map(); // id -> fraction of a turn
        order.forEach(lid => {
            const list = byLevel.get(lid);
            if (!list) return;
            const key = n => {
                const ps = (parents.get(n.id) || []).filter(p => angleOf.has(p));
                if (!ps.length) return null;
                // circular mean of parent angles
                let x = 0, y = 0;
                ps.forEach(p => { x += Math.cos(angleOf.get(p) * TAU); y += Math.sin(angleOf.get(p) * TAU); });
                return ((Math.atan2(y, x) / TAU) + 1) % 1;
            };
            const keyed = list.map(n => ({ n, k: key(n) }));
            keyed.sort((a, b) => {
                if (a.k !== null && b.k !== null && a.k !== b.k) return a.k - b.k;
                if (a.k === null && b.k !== null) return 1;
                if (b.k === null && a.k !== null) return -1;
                return (a.n.rule + a.n.id).localeCompare(b.n.rule + b.n.id);
            });
            // rotate so the ring starts near its first parent's angle
            const offset = keyed.length && keyed[0].k !== null ? keyed[0].k : 0;
            keyed.forEach(({ n }, i) => {
                n._peers = { idx: i, count: keyed.length, offset };
                angleOf.set(n.id, (offset + i / keyed.length) % 1);
            });
        });
    }

    // ── data ingestion ───────────────────────────────────────────────────
    function applySnapshot(st, node) {
        if (!st) return;
        const t = now();
        const lat = st.lattice || {};
        const newVersion = lat.version;
        const prevIds = new Set(levels.map(l => l.id));
        levels = (lat.levels || []).map((lv, i, arr) => ({
            ...lv,
            color: levelColor(i, arr.length),
            born: prevIds.has(lv.id) ? (levels.find(l => l.id === lv.id) || {}).born || t : t + i * 120,
        }));
        version = newVersion;
        iteration = st.iteration || 0;
        maxIter = st.max_iterations || maxIter;

        // mesh
        const mesh = st.mesh || { nodes: [], edges: [] };
        if (mesh.nodes.length) {
            const keep = new Set();
            const gen = new Map();
            (st.agents || []).forEach(a => { if (a.gen > (gen.get(a.node) || 0)) gen.set(a.node, a.gen); });
            mesh.nodes.forEach((m, i) => {
                keep.add(m.id);
                const ex = nodes.get(m.id);
                if (ex) { ex.level = m.level; ex.rule = m.rule; ex.dying = 0; ex.gen = gen.get(m.id) || 0; }
                else nodes.set(m.id, { ...m, born: t + i * 28, dying: 0, gen: gen.get(m.id) || 0 });
            });
            for (const n of nodes.values()) if (!keep.has(n.id) && !n.dying) n.dying = t;
            edges = mesh.edges;
            agents = (st.agents || []).map((a, i) => ({
                ...a,
                phase: (i * 2.399) % TAU,
                speed: 0.9 + (i % 5) * 0.18,
            }));
            reindexNodes();
        }

        // proposals
        (st.proposals || []).forEach(p => {
            const ex = props.get(p.id);
            if (ex) { ex.status = p.status; ex.score = p.score; ex.op = p.op; }
            else if (!absorbed.has(p.id) && !shed.has(p.id)) {
                const s = stages.mutation_proposal;
                props.set(p.id, { ...p, x: s.x, y: s.y, alpha: 0, born: t });
            }
        });

        if (node === 'fitness') {
            for (const p of props.values()) if (p.status === 'rejected' && !shed.has(p.id)) { shed.add(p.id); p.shedAt = t; }
        }
        if (node === 'rectify') {
            for (const p of props.values()) if (p.status === 'accepted' && !absorbed.has(p.id)) { absorbed.add(p.id); p.absorbAt = t; }
            const caps = st.capacity || [];
            const prev = [...caps].reverse().find(c => c.iteration < iteration) || caps[0];
            const gained = caps.length >= 2 && caps[caps.length - 1].capacity > prev.capacity + 1e-9;
            ripples.push({ t, color: gained ? COL.green : COL.white });
            if ((st.rollbacks || []).some(r => r.iteration === iteration)) flashes.push({ stage: 'rectify', color: COL.red, t });
            echoes.push({ t, i: echoes.length });
        }
        if (node === 'scale_induction') ripples.push({ t, color: COL.violet });
    }

    // ── event queue (paced replay) ───────────────────────────────────────
    function push(ev) { queue.push(ev); pump(); }

    function pump() {
        if (pumping) return;
        pumping = true;
        const step = () => {
            if (!queue.length) { pumping = false; return; }
            const wait = Math.max(0, lastPlay + DWELL_MS - now());
            setTimeout(() => {
                const ev = queue.shift();
                lastPlay = now();
                play(ev);
                step();
            }, wait);
        };
        step();
    }

    function play(ev) {
        if (ev.type === 'complete') {
            setActive(null);
            if (ev.snapshot) applySnapshot(ev.snapshot, 'complete');
            running = false;
            updateHud();
            if (onState) onState(ev.snapshot, ev);
            return;
        }
        if (ev.phase === 'enter') {
            setActive(ev.node);
        } else {
            const s = stages[ev.node];
            if (s) { s.done += 1; flashes.push({ stage: ev.node, color: COL.violet, t: now() }); }
            applySnapshot(ev.state, ev.node);
        }
        updateHud();
        if (onState) onState(ev.state, ev);
    }

    function setActive(name) {
        active = name;
        if (onStage) onStage(name);
        if (!name) return;
        if (TAIL.includes(name)) {
            const s = stages[name].s;
            if (!comet || comet.mode !== 'tail') comet = { mode: 'tail', s: 0, ts: s };
            else comet.ts = Math.max(comet.ts, s);
            return;
        }
        const ta = stages[name].angle;
        if (!comet || comet.mode === 'tail') {
            comet = { mode: 'tail', s: comet ? comet.s : 0, ts: 1, then: ta };
            return;
        }
        let target = ta;
        while (target < comet.ta - 1e-6) target += TAU;
        comet.ta = target;
    }

    // ── render ───────────────────────────────────────────────────────────
    function cometPos() {
        if (!comet) return null;
        if (comet.mode === 'tail') {
            const p = tailPts[Math.round(Math.min(1, comet.s) * 80)];
            return p;
        }
        return { x: cx + Math.cos(comet.a) * R, y: cy + Math.sin(comet.a) * R };
    }

    function stepComet() {
        if (!comet) return;
        if (comet.mode === 'tail') {
            comet.s += (comet.ts - comet.s) * 0.06 + 0.002 * Math.sign(comet.ts - comet.s);
            if (comet.s >= 0.995 && comet.ts >= 1) {
                comet = { mode: 'ring', a: -Math.PI / 2, ta: comet.then !== undefined ? comet.then : -Math.PI / 2 };
                while (comet.ta < comet.a - 1e-6) comet.ta += TAU;
            }
            if (comet && comet.mode === 'tail' && Math.abs(comet.ts - comet.s) < 0.002) comet.s = comet.ts;
        } else {
            const d = comet.ta - comet.a;
            comet.a += d * 0.07 + (d > 0.001 ? 0.004 : 0);
            if (comet.a > comet.ta) comet.a = comet.ta;
            if (!running && !active) { comet.ta += 0.006; }
        }
        const p = cometPos();
        if (p) { trail.push({ x: p.x, y: p.y }); if (trail.length > 42) trail.shift(); }
    }

    function drawBackground(t) {
        ctx.fillStyle = '#040405';
        ctx.fillRect(0, 0, W, H);
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 1.6);
        g.addColorStop(0, 'rgba(139,92,246,0.10)');
        g.addColorStop(0.5, 'rgba(59,130,246,0.04)');
        g.addColorStop(1, 'transparent');
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, W, H);
    }

    function drawLoop(t) {
        // echo orbits — one per completed iteration
        echoes.forEach((e, i) => {
            const age = ease((t - e.t) / 1200);
            const r = R + 16 + i * 7;
            ctx.save();
            ctx.setLineDash([2, 6]);
            ctx.lineDashOffset = -t / 60 * (i % 2 ? 1 : -1);
            ctx.strokeStyle = rgba(COL.violet, 0.22 * age);
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.arc(cx, cy, r * (0.9 + 0.1 * age), 0, TAU); ctx.stroke();
            ctx.restore();
        });

        // main ring
        ctx.lineWidth = 1.4;
        const grad = ctx.createLinearGradient(cx - R, cy - R, cx + R, cy + R);
        grad.addColorStop(0, rgba(COL.violet, 0.55));
        grad.addColorStop(1, rgba(COL.cyan, 0.4));
        ctx.strokeStyle = grad;
        ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.stroke();

        // flowing direction ticks along the ring
        ctx.save();
        ctx.setLineDash([1, 14]);
        ctx.lineDashOffset = -t / 40;
        ctx.strokeStyle = 'rgba(255,255,255,0.35)';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.stroke();
        ctx.restore();

        // tail spiral
        ctx.beginPath();
        tailPts.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
        const tg = ctx.createLinearGradient(tailPts[0].x, tailPts[0].y, tailPts[80].x, tailPts[80].y);
        tg.addColorStop(0, 'rgba(255,255,255,0.05)');
        tg.addColorStop(1, rgba(COL.violet, 0.55));
        ctx.strokeStyle = tg;
        ctx.lineWidth = 1.4;
        ctx.stroke();
        ctx.save();
        ctx.setLineDash([1, 12]);
        ctx.lineDashOffset = -t / 40;
        ctx.strokeStyle = 'rgba(255,255,255,0.25)';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();

        // back-edge chevron: rectify → mutation_proposal closes the loop
        const a = stages.rectify.angle + (TAU / 5) * 0.5;
        const px = cx + Math.cos(a) * R, py = cy + Math.sin(a) * R;
        ctx.save();
        ctx.translate(px, py); ctx.rotate(a + Math.PI / 2);
        ctx.fillStyle = rgba(COL.violet, 0.8);
        ctx.beginPath(); ctx.moveTo(5, 0); ctx.lineTo(-4, -4); ctx.lineTo(-4, 4); ctx.closePath(); ctx.fill();
        ctx.restore();
    }

    function drawStages(t) {
        const all = [...TAIL, ...LOOP];
        all.forEach(name => {
            const s = stages[name];
            const isActive = active === name;
            s.glow += ((isActive ? 1 : 0) - s.glow) * 0.08;
            const r = 7 + s.glow * 3;
            if (s.glow > 0.02) {
                const g = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, 34);
                g.addColorStop(0, rgba(COL.violet, 0.45 * s.glow));
                g.addColorStop(1, 'transparent');
                ctx.fillStyle = g;
                ctx.beginPath(); ctx.arc(s.x, s.y, 34, 0, TAU); ctx.fill();
                // breathing halo
                ctx.strokeStyle = rgba(COL.violet, 0.5 * s.glow);
                ctx.lineWidth = 1;
                ctx.beginPath(); ctx.arc(s.x, s.y, r + 6 + 3 * Math.sin(t / 240), 0, TAU); ctx.stroke();
            }
            ctx.fillStyle = '#07070a';
            ctx.strokeStyle = isActive ? '#fff' : (s.done ? rgba(COL.violet, 0.85) : 'rgba(255,255,255,0.28)');
            ctx.lineWidth = 1.4;
            ctx.beginPath(); ctx.arc(s.x, s.y, r, 0, TAU); ctx.fill(); ctx.stroke();
            if (s.done) {
                ctx.fillStyle = rgba(COL.violet, 0.9);
                ctx.beginPath(); ctx.arc(s.x, s.y, 2.6, 0, TAU); ctx.fill();
            }

            // label, pushed outward from the ring centre and kept on-canvas.
            // Narrow screens show only the active label (the rail lists all).
            if (W < 560 && !isActive) return;
            const dx = s.x - cx, dy = s.y - cy, d = Math.hypot(dx, dy) || 1;
            const label = STAGE_META[name].label.toUpperCase();
            ctx.font = '600 10px -apple-system, "Segoe UI", sans-serif';
            if ('letterSpacing' in ctx) ctx.letterSpacing = '1.5px';
            const tw = ctx.measureText(label).width;
            let lx = s.x + dx / d * 22;
            const ly = Math.max(10, Math.min(H - 10, s.y + dy / d * 20));
            const align = Math.abs(dx / d) < 0.3 ? 'center' : (dx > 0 ? 'left' : 'right');
            const left = align === 'center' ? lx - tw / 2 : align === 'left' ? lx : lx - tw;
            lx += Math.max(0, 8 - left) - Math.max(0, left + tw - (W - 8));
            ctx.textAlign = align;
            ctx.textBaseline = 'middle';
            ctx.fillStyle = isActive ? '#fff' : 'rgba(255,255,255,0.45)';
            ctx.fillText(label, lx, ly);
            if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';
        });

        flashes = flashes.filter(f => t - f.t < 900);
        flashes.forEach(f => {
            const s = stages[f.stage]; const k = (t - f.t) / 900;
            ctx.strokeStyle = rgba(f.color, 0.7 * (1 - k));
            ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(s.x, s.y, 10 + 26 * ease(k), 0, TAU); ctx.stroke();
        });
    }

    function drawComet() {
        if (trail.length < 2) return;
        for (let i = 1; i < trail.length; i++) {
            const k = i / trail.length;
            ctx.strokeStyle = `rgba(255,255,255,${0.5 * k * k})`;
            ctx.lineWidth = 2.5 * k;
            ctx.beginPath();
            ctx.moveTo(trail[i - 1].x, trail[i - 1].y);
            ctx.lineTo(trail[i].x, trail[i].y);
            ctx.stroke();
        }
        const p = trail[trail.length - 1];
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 14);
        g.addColorStop(0, 'rgba(255,255,255,0.95)');
        g.addColorStop(0.3, rgba(COL.violet, 0.6));
        g.addColorStop(1, 'transparent');
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(p.x, p.y, 14, 0, TAU); ctx.fill();
    }

    function drawLattice(t) {
        const n = levels.length;
        if (!n) {
            // ghost scaffold while idle
            for (let i = 0; i < 4; i++) {
                ctx.strokeStyle = 'rgba(255,255,255,0.05)';
                ctx.setLineDash([2, 8]);
                ctx.beginPath(); ctx.arc(cx, cy, levelRadius(i, 4), 0, TAU); ctx.stroke();
                ctx.setLineDash([]);
            }
            return;
        }
        const rr = rot();
        levels.forEach((lv, i) => {
            const k = ease((t - lv.born) / 900);
            if (k <= 0) return;
            const r = levelRadius(i, n) * (0.85 + 0.15 * k);
            ctx.save();
            ctx.setLineDash([1, 5]);
            ctx.lineDashOffset = (i % 2 ? 1 : -1) * t / 90;
            ctx.strokeStyle = rgba(lv.color, 0.28 * k);
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.arc(cx, cy, r, 0, TAU); ctx.stroke();
            ctx.restore();
            // rule ticks: the grammar's rules as radial notches on their scale
            const rules = lv.rules || [];
            rules.forEach((ru, j) => {
                const a = -rr * (i % 2 ? 1 : -1) * 0.6 + (j / Math.max(1, rules.length)) * TAU + i;
                ctx.strokeStyle = rgba(lv.color, 0.5 * k);
                ctx.lineWidth = 1.2;
                ctx.beginPath();
                ctx.moveTo(cx + Math.cos(a) * (r - 4), cy + Math.sin(a) * (r - 4));
                ctx.lineTo(cx + Math.cos(a) * (r + 4), cy + Math.sin(a) * (r + 4));
                ctx.stroke();
            });
            // level id label at the left of each ring
            ctx.font = '600 9px Menlo, Consolas, monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = rgba(lv.color, 0.75 * k);
            ctx.fillText(lv.id, cx, cy - r - 7);
        });

        // positions this frame
        const pos = new Map();
        for (const nd of nodes.values()) {
            const p = nodePos(nd);
            if (p) pos.set(nd.id, p);
        }

        // edges
        edges.forEach(([s, d, rel]) => {
            const a = pos.get(s), b = pos.get(d);
            const na = nodes.get(s), nb = nodes.get(d);
            if (!a || !b || !na || !nb) return;
            const k = Math.min(ease((t - na.born) / 600), ease((t - nb.born) / 600)) * fade(na, t) * fade(nb, t);
            if (k <= 0) return;
            const col = levels[a.li] ? levels[a.li].color : COL.white;
            ctx.save();
            if (rel === 'expands-to') {
                ctx.strokeStyle = rgba(col, 0.2 * k);
                ctx.lineWidth = 0.8;
                const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
                const ox = (mx - cx) * 0.12, oy = (my - cy) * 0.12;
                ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.quadraticCurveTo(mx + oy, my - ox, b.x, b.y); ctx.stroke();
                // a signal bead sliding outward: the law expanding downward
                const u = ((t / 2400) + (s.charCodeAt(6) || 0) / 37) % 1;
                const qx = (1 - u) * (1 - u) * a.x + 2 * (1 - u) * u * (mx + oy) + u * u * b.x;
                const qy = (1 - u) * (1 - u) * a.y + 2 * (1 - u) * u * (my - ox) + u * u * b.y;
                ctx.fillStyle = rgba(col, 0.55 * k);
                ctx.beginPath(); ctx.arc(qx, qy, 1.1, 0, TAU); ctx.fill();
            } else {
                ctx.setLineDash([2, 3]);
                ctx.strokeStyle = rgba(COL.amber, 0.38 * k);
                ctx.lineWidth = 0.9;
                ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
            }
            ctx.restore();
        });

        // agents per node
        const byNode = new Map();
        agents.forEach(ag => { if (!byNode.has(ag.node)) byNode.set(ag.node, []); byNode.get(ag.node).push(ag); });

        // nodes
        hover = null;
        for (const nd of nodes.values()) {
            const p = pos.get(nd.id);
            if (!p) continue;
            const k = ease((t - nd.born) / 500) * fade(nd, t);
            if (k <= 0) continue;
            const col = levels[p.li] ? levels[p.li].color : COL.white;
            const size = 2.6 + (n - p.li) * 0.5;
            const isHover = mouse && Math.hypot(mouse.x - p.x, mouse.y - p.y) < 9;
            if (isHover) hover = { nd, p };
            const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, size * 4);
            g.addColorStop(0, rgba(col, 0.35 * k));
            g.addColorStop(1, 'transparent');
            ctx.fillStyle = g;
            ctx.beginPath(); ctx.arc(p.x, p.y, size * 4, 0, TAU); ctx.fill();
            ctx.fillStyle = rgba(mix(col, COL.white, isHover ? 0.6 : 0.25), k);
            ctx.beginPath(); ctx.arc(p.x, p.y, size * (isHover ? 1.6 : 1) * (0.6 + 0.4 * k), 0, TAU); ctx.fill();

            (byNode.get(nd.id) || []).forEach(ag => {
                const ar = size + 4 + ag.gen * 2.5;
                const aa = ag.phase + (t / 1000) * ag.speed * (ag.gen ? -1.4 : 1);
                ctx.fillStyle = ag.gen ? rgba(COL.amber, 0.9 * k) : rgba(COL.white, 0.7 * k);
                ctx.beginPath(); ctx.arc(p.x + Math.cos(aa) * ar, p.y + Math.sin(aa) * ar, ag.gen ? 1.3 : 1, 0, TAU); ctx.fill();
            });
        }
        for (const [id, nd] of nodes) if (nd.dying && t - nd.dying > 900) nodes.delete(id);
        canvas.style.cursor = hover ? 'pointer' : 'default';
        renderTip();

        // capacity ripples from the lattice core
        ripples = ripples.filter(rp => t - rp.t < 1800);
        ripples.forEach(rp => {
            const k = (t - rp.t) / 1800;
            ctx.strokeStyle = rgba(rp.color, 0.45 * (1 - k));
            ctx.lineWidth = 2 * (1 - k) + 0.5;
            ctx.beginPath(); ctx.arc(cx, cy, 6 + ease(k) * R * 0.95, 0, TAU); ctx.stroke();
        });

        // core: the compact law itself
        const pulse = 0.5 + 0.5 * Math.sin(t / 700);
        const cg = ctx.createRadialGradient(cx, cy, 0, cx, cy, 18);
        cg.addColorStop(0, `rgba(255,255,255,${0.75 + 0.2 * pulse})`);
        cg.addColorStop(0.4, rgba(COL.violet, 0.5));
        cg.addColorStop(1, 'transparent');
        ctx.fillStyle = cg;
        ctx.beginPath(); ctx.arc(cx, cy, 18, 0, TAU); ctx.fill();
    }

    function fade(nd, t) { return nd.dying ? Math.max(0, 1 - (t - nd.dying) / 900) : 1; }

    function proposalTargets() {
        const groups = { mutation_proposal: [], fitness: [], rectify: [] };
        const t = now();
        for (const p of props.values()) {
            if (absorbed.has(p.id) || shed.has(p.id)) continue;
            if (p.status === 'proposed') groups.mutation_proposal.push(p);
            else if (p.status === 'accepted') (active === 'rectify' ? groups.rectify : groups.fitness).push(p);
            else groups.fitness.push(p);
        }
        Object.entries(groups).forEach(([stage, list]) => {
            const s = stages[stage];
            list.forEach((p, k) => {
                const row = Math.floor(k / 9), col = k % 9, inRow = Math.min(9, list.length - row * 9);
                const a = s.angle + (col - (inRow - 1) / 2) * 0.085;
                const r = R + 30 + row * 12;
                p.tx = cx + Math.cos(a) * r; p.ty = cy + Math.sin(a) * r;
            });
        });
        for (const p of props.values()) {
            if (absorbed.has(p.id)) { p.tx = cx; p.ty = cy; }
            if (shed.has(p.id) && p.tx !== undefined) {
                const dx = p.x - cx, dy = p.y - cy, d = Math.hypot(dx, dy) || 1;
                p.tx = p.x + dx / d * 3; p.ty = p.y + dy / d * 3;
            }
        }
        return t;
    }

    function drawSwarm(t) {
        proposalTargets();
        for (const [id, p] of props) {
            if (p.tx === undefined) continue;
            const sp = absorbed.has(id) ? 0.05 : 0.09;
            p.x += (p.tx - p.x) * sp; p.y += (p.ty - p.y) * sp;
            let a = ease((t - p.born) / 500);
            if (absorbed.has(id)) a *= Math.max(0, 1 - (t - p.absorbAt) / 1600);
            if (shed.has(id)) a *= Math.max(0, 1 - (t - p.shedAt) / 1800);
            if (a <= 0.01 && (absorbed.has(id) || shed.has(id)) && t - p.born > 1000) { props.delete(id); continue; }
            const col = p.status === 'accepted' ? COL.green : p.status === 'rejected' ? COL.red : COL.white;
            const s = p.op === 'rewire_topology' ? 3.6 : 3;
            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(Math.PI / 4 + (p.op === 'rewire_topology' ? t / 600 : 0));
            ctx.fillStyle = rgba(col, 0.85 * a);
            ctx.fillRect(-s / 2, -s / 2, s, s);
            ctx.strokeStyle = rgba(col, 0.35 * a);
            ctx.strokeRect(-s, -s, s * 2, s * 2);
            ctx.restore();
            if (absorbed.has(id) && a > 0.05) {
                ctx.strokeStyle = rgba(COL.green, 0.12 * a);
                ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(cx, cy); ctx.stroke();
            }
        }
    }

    function renderTip() {
        if (!tip) return;
        if (!hover) { tip.classList.remove('is-visible'); return; }
        const { nd, p } = hover;
        const ag = agents.filter(a => a.node === nd.id);
        const lv = levels[p.li] || {};
        tip.innerHTML =
            `<b>${nd.id}</b><span>${lv.id || nd.level} · ${escapeHtml(lv.name || '')}</span>` +
            `<span>rule <code>${escapeHtml(nd.rule)}</code></span>` +
            `<span>${ag.length} inhabitant${ag.length === 1 ? '' : 's'}${ag.some(a => a.gen) ? ' · spawned' : ''}</span>`;
        tip.style.left = Math.min(W - 200, p.x + 14) + 'px';
        tip.style.top = Math.max(8, p.y - 12) + 'px';
        tip.classList.add('is-visible');
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    function frame() {
        raf = requestAnimationFrame(frame);
        if (!mount || mount.offsetParent === null || document.hidden) return;
        const t = now();
        stepComet();
        drawBackground(t);
        drawLattice(t);
        drawLoop(t);
        drawSwarm(t);
        drawComet();
        drawStages(t);
    }

    // ── HUD ──────────────────────────────────────────────────────────────
    function updateHud() {
        if (!hud.version) return;
        hud.version.textContent = version && version !== 'v-none' ? version : '—';
        hud.loop.textContent = running || iteration ? `loop ${Math.min(iteration + (running ? 1 : 0), maxIter)} / ${maxIter}` : 'idle';
        const meta = active ? STAGE_META[active] : null;
        hud.stage.textContent = meta ? meta.label : (running ? '…' : (iteration ? 'converged' : 'awaiting a formal artifact'));
        hud.hint.textContent = meta ? meta.hint : (iteration ? 'the lattice has absorbed its accepted mutations' : 'paste a law, a spec, a repo — any generative formal system');
        hud.root.classList.toggle('is-running', running);
    }

    // ── public API ───────────────────────────────────────────────────────
    function init(el, opts = {}) {
        mount = typeof el === 'string' ? document.getElementById(el) : el;
        if (!mount) return;
        onState = opts.onState || null;
        onStage = opts.onStage || null;
        mount.classList.add('rsi-lattice');
        mount.innerHTML = `
            <canvas class="rsi-lattice__canvas"></canvas>
            <div class="rsi-lattice__grain"></div>
            <div class="rsi-lattice__hud">
                <div class="rsi-hud__corner">
                    <span class="rsi-hud__eyebrow">lattice</span>
                    <span class="rsi-hud__version" data-k="version">—</span>
                    <span class="rsi-hud__loop" data-k="loop">idle</span>
                </div>
                <div class="rsi-hud__stage">
                    <span class="rsi-hud__stage-name" data-k="stage"></span>
                    <span class="rsi-hud__stage-hint" data-k="hint"></span>
                </div>
                <div class="rsi-hud__legend">
                    <span><i class="rsi-dot rsi-dot--node"></i>instantiation</span>
                    <span><i class="rsi-dot rsi-dot--agent"></i>agent</span>
                    <span><i class="rsi-dot rsi-dot--spawn"></i>spawned</span>
                    <span><i class="rsi-dot rsi-dot--prop"></i>mutation</span>
                </div>
            </div>
            <div class="rsi-lattice__tip"></div>`;
        canvas = mount.querySelector('canvas');
        ctx = canvas.getContext('2d');
        tip = mount.querySelector('.rsi-lattice__tip');
        hud = { root: mount.querySelector('.rsi-lattice__hud') };
        mount.querySelectorAll('[data-k]').forEach(n => { hud[n.dataset.k] = n; });

        canvas.addEventListener('mousemove', e => {
            const r = canvas.getBoundingClientRect();
            mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
        });
        canvas.addEventListener('mouseleave', () => { mouse = null; });
        if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => layout()).observe(mount);
        window.addEventListener('resize', layout);
        layout();
        comet = { mode: 'ring', a: -Math.PI / 2, ta: -Math.PI / 2 };
        updateHud();
        if (!raf) frame();
    }

    function reset() {
        queue = []; active = null; levels = []; nodes = new Map(); edges = []; agents = [];
        props = new Map(); absorbed = new Set(); shed = new Set(); ripples = []; echoes = []; flashes = [];
        version = null; iteration = 0; trail = [];
        Object.values(stages).forEach(s => { s.done = 0; s.glow = 0; });
        comet = { mode: 'tail', s: 0, ts: 0 };
        running = true;
        layout();
        updateHud();
    }

    function stopIdle() {
        running = false; queue = []; active = null;
        comet = { mode: 'ring', a: -Math.PI / 2, ta: -Math.PI / 2 };
        updateHud();
    }

    return {
        init,
        reset,
        stop: stopIdle,
        event: ev => push(ev),
        complete: snapshot => push({ type: 'complete', snapshot }),
        isRunning: () => running,
        STAGES: [...TAIL, ...LOOP],
        STAGE_META,
    };
})();

if (typeof window !== 'undefined') window.RsiLattice = RsiLattice;
