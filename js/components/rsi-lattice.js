/**
 * RsiLattice: "Ouroboros Lattice" for the RSI security loop.
 *
 *   • Ring   = the loop. The five looping stages (correspond, tick, heal,
 *              stock, lora) sit on a circle; ingest and scale induction
 *              spiral into it as a tail. A comet marks the active node and
 *              each finished loop leaves an echo orbit.
 *   • AX band (outer)   = the security policy: clauses grouped by level
 *              (principles, policies, controls, assets) and the viable control
 *              operations. Contradictory clauses are joined by red chords.
 *              Controls added by the loop are born here in green.
 *   • TC spokes         = the table of correspondences: one gauge link from
 *              each clause to its agent, tinted by its phase A (aligned or not).
 *   • AM lattice (inner) = Agent Mechanics agents, coloured by hero species,
 *              placed on their clause's spoke at a radius set by rank (Attr
 *              near the core, Inr near the band). A red halo means the agent
 *              leaks; a green halo means it stocks. Typed edges are drawn in
 *              the lattice: stems and duals in species colour, hostile (op)
 *              edges dashed red, parentage (pr) amber.
 *
 * Heals flash red from the agent out along its spokes to the clashing
 * clauses. Kept controls travel green from the stocking agent to the band.
 * LoRA clamps pulse the core. Events are replayed with a minimum dwell so the
 * offline engine, which finishes in milliseconds, still reads as a loop.
 */
const RsiLattice = (() => {
    'use strict';

    const LOOP = ['correspond', 'tick', 'heal', 'stock', 'lora'];
    const TAIL = ['ingest', 'scale_induction'];
    const STAGE_META = {
        ingest: { label: 'ingest', hint: 'tokenising the policy' },
        scale_induction: { label: 'induce', hint: 'clauses → principles, policies, controls, assets' },
        correspond: { label: 'correspond', hint: 'table of correspondences · one agent per clause, modulus and operation' },
        tick: { label: 'tick', hint: 'the Agent Mechanics tick law · flux, Leak, stock, ranks, ensembles' },
        heal: { label: 'heal', hint: 'leaking agents heal with licensed operators → findings' },
        stock: { label: 'stock', hint: 'surplus becomes controls · kept only if posture improves' },
        lora: { label: 'lora', hint: 'distil the traces · clamp the adapter when the loop improves' },
        artifact_emit: { label: 'report', hint: 'hardened policy, findings and the NX-1 log' },
    };
    const SPECIES_COL = {
        Ni: [139, 92, 246], Ne: [192, 132, 252], Si: [59, 130, 246], Se: [6, 182, 212],
        Ti: [45, 212, 191], Te: [163, 230, 53], Fi: [251, 191, 36], Fe: [244, 114, 182],
    };
    const COL = {
        violet: [139, 92, 246], cyan: [6, 182, 212], green: [52, 211, 153],
        red: [239, 68, 68], amber: [251, 191, 36], white: [255, 255, 255],
    };
    const RANK_R = { Attr: 0.2, Cyc: 0.3, Res: 0.4, Seed: 0.5, Inr: 0.58 };
    const GROUPS = ['L3', 'L2', 'L1', 'L0', 'op'];
    const GROUP_NAME = { L3: 'principles', L2: 'policies', L1: 'controls', L0: 'assets', op: 'operations' };
    const DWELL_MS = 560;
    const TAU = Math.PI * 2;

    const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
    const mix = (a, b, t) => [0, 1, 2].map(i => Math.round(a[i] + (b[i] - a[i]) * t));
    const ease = t => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);
    const now = () => performance.now();
    const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

    let mount, canvas, ctx, tip, hud = {};
    let W = 800, H = 560, cx = 400, cy = 280, R = 200, dpr = 1;
    let raf = null, t0 = now();
    let onState = null, onStage = null;

    let stages = {};
    let tailPts = [];
    let active = null;
    let comet = null;
    let trail = [];
    let echoes = [];
    let ripples = [];
    let flashes = [];
    let pulses = [];      // travelling light {from:{x,y}, to:{x,y}, t, col}
    let chords = [];      // contradiction chords {a, b, A}
    let ax = new Map();   // ref -> {ref, kind, text, group, angle, x, y, born, rsi}
    let agents = new Map(); // id -> {..view, x, y, tx, ty, born}
    let edges = [];
    let toc = [];
    let seen = new Set();
    let queue = [], pumping = false, lastPlay = 0;
    let version = null, iteration = 0, maxIter = 1, posture = null, clampScale = 0;
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
        R = Math.min(H * 0.4, (W / 2 - 24) / 1.5);
        cx = W / 2 + R * 0.16;
        cy = H / 2 + 6;
        LOOP.forEach((name, i) => {
            const a = -Math.PI / 2 + i * TAU / LOOP.length;
            stages[name] = { ...(stages[name] || { glow: 0, done: 0 }), x: cx + Math.cos(a) * R, y: cy + Math.sin(a) * R, angle: a };
        });
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
        placeAx();
    }

    function groupOf(ref) {
        if (ref.startsWith('op/')) return 'op';
        const m = /^pol\/(L\d)/.exec(ref);
        return m ? m[1] : 'op';
    }

    function placeAx() {
        const list = [...ax.values()];
        const order = r => GROUPS.indexOf(r.group) * 10000 + (r.ref.includes('#mod') ? 0 : 1) * 1000 + (parseFloat((r.ref.match(/§R?([\d.]+)/) || [])[1]) || 0) + (r.ref.includes('§R') ? 500 : 0);
        list.sort((a, b) => order(a) - order(b) || a.ref.localeCompare(b.ref));
        const groups = GROUPS.filter(g => list.some(n => n.group === g));
        const gap = 0.12;
        const usable = TAU - gap * groups.length;
        let a = -Math.PI / 2 + gap / 2;
        const rax = R * 0.8;
        let prev = null;
        list.forEach(n => {
            if (prev && prev !== n.group) a += gap;
            prev = n.group;
            n.angle = a + (usable / Math.max(1, list.length)) / 2;
            n.x = cx + Math.cos(n.angle) * rax;
            n.y = cy + Math.sin(n.angle) * rax;
            a += usable / Math.max(1, list.length);
        });
        for (const ag of agents.values()) targetAgent(ag);
    }

    function targetAgent(ag) {
        if (ag.px === undefined) { ag.px = cx; ag.py = cy; }
        if (ag.id === 'I1' || !ag.ax) { ag.tx = cx; ag.ty = cy; return; }
        const node = ax.get(ag.ax);
        const r = R * (RANK_R[ag.rank] || 0.5);
        const ang = node ? node.angle : 0;
        ag.tx = cx + Math.cos(ang) * r;
        ag.ty = cy + Math.sin(ang) * r;
    }

    // ── data ─────────────────────────────────────────────────────────────
    function applySnapshot(st) {
        if (!st) return;
        const t = now();
        version = (st.lattice || {}).version || version;
        iteration = st.iteration || 0;
        maxIter = st.max_iterations || maxIter;
        const post = st.posture || [];
        if (post.length) posture = post[post.length - 1];
        if (st.lora) clampScale = st.lora.scale || 0;
        const rsiRules = new Set();
        (st.lattice?.levels || []).forEach(lv => (lv.rules || []).forEach(r => { if (r.rsi) rsiRules.add(`pol/${lv.id}/${r.id}`); }));
        if (Array.isArray(st.ax)) {
            const keep = new Set();
            st.ax.forEach(n => {
                keep.add(n.ref);
                const ex = ax.get(n.ref);
                if (ex) { ex.text = n.text; ex.rsi = rsiRules.has(n.ref); }
                else ax.set(n.ref, { ...n, group: groupOf(n.ref), born: t, rsi: rsiRules.has(n.ref) });
            });
            for (const k of [...ax.keys()]) if (!keep.has(k)) ax.delete(k);
            chords = (st.ax_edges || []).filter(e => e[2] >= Math.PI / 2).map(e => ({ a: e[0], b: e[1], A: e[2] }));
            placeAx();
        }
        if (Array.isArray(st.agents)) {
            const keep = new Set();
            st.agents.forEach(v => {
                keep.add(v.id);
                const ex = agents.get(v.id);
                if (ex) Object.assign(ex, v);
                else agents.set(v.id, { ...v, born: t });
                targetAgent(agents.get(v.id));
            });
            for (const k of [...agents.keys()]) if (!keep.has(k)) agents.delete(k);
        }
        if (Array.isArray(st.edges)) edges = st.edges;
        if (Array.isArray(st.toc)) toc = st.toc;
        (st.events || []).forEach(ev => {
            const key = ev.text;
            if (seen.has(key)) return;
            seen.add(key);
            effect(ev, t);
        });
    }

    function effect(ev, t) {
        const ag = agents.get(ev.agent);
        if (ev.kind === 'leak' && ag) {
            flashes.push({ x: ag.px, y: ag.py, col: COL.red, t });
            const own = toc.filter(r => r.agent === ag.id).map(r => ax.get(r.ax)).filter(Boolean);
            own.forEach(n => pulses.push({ from: { x: ag.px, y: ag.py }, to: { x: n.x, y: n.y }, t, col: COL.red, dur: 900 }));
        } else if (ev.kind === 'clash') {
            const a = ax.get(ev.a1), b = ax.get(ev.a2);
            if (a && b) pulses.push({ from: { x: a.x, y: a.y }, to: { x: b.x, y: b.y }, t, col: COL.red, dur: 1100, chord: true });
        } else if (ev.kind === 'stock' && ag) {
            flashes.push({ x: ag.px, y: ag.py, col: COL.green, t });
            const n = ev.ref ? ax.get(ev.ref) : null;
            const to = n ? { x: n.x, y: n.y } : { x: cx, y: cy - R * 0.8 };
            pulses.push({ from: { x: ag.px, y: ag.py }, to, t, col: COL.green, dur: 1000 });
        } else if (ev.kind === 'clamp') {
            ripples.push({ t, col: ev.decision === 'rollback' ? COL.amber : COL.violet });
        } else if (ev.kind === 'brk' && ag) {
            flashes.push({ x: ag.px, y: ag.py, col: COL.amber, t });
        }
    }

    // ── queue ────────────────────────────────────────────────────────────
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
            if (ev.snapshot) applySnapshot(ev.snapshot);
            running = false;
            updateHud();
            if (onState) onState(ev.snapshot, ev);
            return;
        }
        if (ev.phase === 'enter') {
            setActive(ev.node);
        } else {
            const s = stages[ev.node];
            if (s) s.done += 1;
            if (ev.node === 'lora') echoes.push({ t: now() });
            applySnapshot(ev.state);
        }
        updateHud();
        if (onState) onState(ev.state, ev);
    }

    function setActive(name) {
        active = name;
        if (onStage) onStage(name);
        if (!name || !stages[name]) return;
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
    function stepComet() {
        if (!comet) return;
        if (comet.mode === 'tail') {
            comet.s += (comet.ts - comet.s) * 0.06 + 0.002 * Math.sign(comet.ts - comet.s);
            if (comet.s >= 0.995 && comet.ts >= 1) {
                comet = { mode: 'ring', a: -Math.PI / 2, ta: comet.then !== undefined ? comet.then : -Math.PI / 2 };
                while (comet.ta < comet.a - 1e-6) comet.ta += TAU;
            } else if (Math.abs(comet.ts - comet.s) < 0.002) comet.s = comet.ts;
        } else {
            const d = comet.ta - comet.a;
            comet.a += d * 0.07 + (d > 0.001 ? 0.004 : 0);
            if (comet.a > comet.ta) comet.a = comet.ta;
            if (!running && !active) comet.ta += 0.006;
        }
        const p = comet.mode === 'tail'
            ? tailPts[Math.round(Math.min(1, Math.max(0, comet.s)) * 80)]
            : { x: cx + Math.cos(comet.a) * R, y: cy + Math.sin(comet.a) * R };
        if (p) { trail.push({ x: p.x, y: p.y }); if (trail.length > 42) trail.shift(); }
    }

    function drawBackground() {
        ctx.fillStyle = '#040405';
        ctx.fillRect(0, 0, W, H);
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 1.6);
        g.addColorStop(0, 'rgba(139,92,246,0.09)');
        g.addColorStop(0.5, 'rgba(59,130,246,0.035)');
        g.addColorStop(1, 'transparent');
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, W, H);
    }

    function drawLoop(t) {
        echoes.forEach((e, i) => {
            const age = ease((t - e.t) / 1200);
            ctx.save();
            ctx.setLineDash([2, 6]);
            ctx.lineDashOffset = -t / 60 * (i % 2 ? 1 : -1);
            ctx.strokeStyle = rgba(COL.violet, 0.22 * age);
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.arc(cx, cy, (R + 16 + i * 7) * (0.9 + 0.1 * age), 0, TAU); ctx.stroke();
            ctx.restore();
        });
        ctx.lineWidth = 1.4;
        const grad = ctx.createLinearGradient(cx - R, cy - R, cx + R, cy + R);
        grad.addColorStop(0, rgba(COL.violet, 0.55));
        grad.addColorStop(1, rgba(COL.cyan, 0.4));
        ctx.strokeStyle = grad;
        ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.stroke();
        ctx.save();
        ctx.setLineDash([1, 14]);
        ctx.lineDashOffset = -t / 40;
        ctx.strokeStyle = 'rgba(255,255,255,0.35)';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.stroke();
        ctx.restore();
        ctx.beginPath();
        tailPts.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
        const tg = ctx.createLinearGradient(tailPts[0].x, tailPts[0].y, tailPts[80].x, tailPts[80].y);
        tg.addColorStop(0, 'rgba(255,255,255,0.05)');
        tg.addColorStop(1, rgba(COL.violet, 0.55));
        ctx.strokeStyle = tg;
        ctx.lineWidth = 1.4;
        ctx.stroke();
        const a = stages.lora.angle + (TAU / 5) * 0.5;
        ctx.save();
        ctx.translate(cx + Math.cos(a) * R, cy + Math.sin(a) * R); ctx.rotate(a + Math.PI / 2);
        ctx.fillStyle = rgba(COL.violet, 0.8);
        ctx.beginPath(); ctx.moveTo(5, 0); ctx.lineTo(-4, -4); ctx.lineTo(-4, 4); ctx.closePath(); ctx.fill();
        ctx.restore();
    }

    function drawStages(t) {
        [...TAIL, ...LOOP].forEach(name => {
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
    }

    function drawComet() {
        if (trail.length < 2) return;
        for (let i = 1; i < trail.length; i++) {
            const k = i / trail.length;
            ctx.strokeStyle = `rgba(255,255,255,${0.5 * k * k})`;
            ctx.lineWidth = 2.5 * k;
            ctx.beginPath(); ctx.moveTo(trail[i - 1].x, trail[i - 1].y); ctx.lineTo(trail[i].x, trail[i].y); ctx.stroke();
        }
        const p = trail[trail.length - 1];
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 14);
        g.addColorStop(0, 'rgba(255,255,255,0.95)');
        g.addColorStop(0.3, rgba(COL.violet, 0.6));
        g.addColorStop(1, 'transparent');
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(p.x, p.y, 14, 0, TAU); ctx.fill();
    }

    function drawDomains(t) {
        const rax = R * 0.8;
        // AX band and level arcs
        ctx.save();
        ctx.setLineDash([1, 5]);
        ctx.lineDashOffset = t / 90;
        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        ctx.beginPath(); ctx.arc(cx, cy, rax, 0, TAU); ctx.stroke();
        ctx.restore();
        if (!ax.size) {
            for (let i = 0; i < 4; i++) {
                ctx.strokeStyle = 'rgba(255,255,255,0.05)';
                ctx.setLineDash([2, 8]);
                ctx.beginPath(); ctx.arc(cx, cy, R * (0.2 + 0.13 * i), 0, TAU); ctx.stroke();
                ctx.setLineDash([]);
            }
        }
        // group labels
        if (W >= 560) {
            GROUPS.forEach(g => {
                const nodes = [...ax.values()].filter(n => n.group === g);
                if (!nodes.length) return;
                const mid = nodes[Math.floor(nodes.length / 2)].angle;
                ctx.font = '600 9px -apple-system, "Segoe UI", sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = 'rgba(255,255,255,0.3)';
                ctx.fillText(`${g === 'op' ? '' : g + ' · '}${GROUP_NAME[g]}`.toUpperCase(), cx + Math.cos(mid) * (rax + 22), cy + Math.sin(mid) * (rax + 22));
            });
        }
        // TC spokes: gauge links clause -> agent
        toc.forEach(r => {
            const n = ax.get(r.ax), ag = agents.get(r.agent);
            if (!n || !ag || ag.px === undefined) return;
            const al = Math.cos(r.A || 0);
            const col = al > 0.5 ? COL.cyan : al > 0 ? COL.violet : COL.red;
            ctx.strokeStyle = rgba(col, 0.1 + 0.12 * Math.abs(al));
            ctx.lineWidth = 0.8;
            ctx.beginPath(); ctx.moveTo(n.x, n.y); ctx.lineTo(ag.px, ag.py); ctx.stroke();
        });
        // contradiction chords between clauses
        chords.forEach(c => {
            const a = ax.get(c.a), b = ax.get(c.b);
            if (!a || !b) return;
            const pulse = 0.5 + 0.5 * Math.sin(t / 420);
            ctx.strokeStyle = rgba(COL.red, 0.35 + 0.35 * pulse);
            ctx.lineWidth = 1.4;
            ctx.beginPath(); ctx.moveTo(a.x, a.y);
            ctx.quadraticCurveTo(cx + (a.x + b.x - 2 * cx) * 0.25, cy + (a.y + b.y - 2 * cy) * 0.25, b.x, b.y);
            ctx.stroke();
        });
        // AM edges
        edges.forEach(([s, d, typ, sp]) => {
            const a = agents.get(s), b = agents.get(d);
            if (!a || !b || a.px === undefined || b.px === undefined) return;
            ctx.save();
            if (typ === 'op') { ctx.setLineDash([2, 3]); ctx.strokeStyle = rgba(COL.red, 0.4); }
            else if (typ === 'pr') { ctx.setLineDash([1, 4]); ctx.strokeStyle = rgba(COL.amber, 0.35); }
            else if (typ === 'st' || typ === 'du') ctx.strokeStyle = rgba(SPECIES_COL[sp] || COL.white, typ === 'du' ? 0.5 : 0.32);
            else ctx.strokeStyle = 'rgba(255,255,255,0.06)';
            ctx.lineWidth = typ === 'du' ? 1.3 : 0.8;
            ctx.beginPath(); ctx.moveTo(a.px, a.py); ctx.lineTo(b.px, b.py); ctx.stroke();
            ctx.restore();
        });
        // AX nodes
        hover = null;
        for (const n of ax.values()) {
            const k = ease((t - n.born) / 600);
            const isHover = mouse && Math.hypot(mouse.x - n.x, mouse.y - n.y) < 8;
            if (isHover) hover = { kind: 'ax', n };
            const col = n.rsi ? COL.green : n.group === 'op' ? COL.amber : n.ref.includes('#mod') ? COL.violet : COL.white;
            const s = (n.ref.includes('#mod') ? 3 : 2.6) * (isHover ? 1.6 : 1) * (0.5 + 0.5 * k);
            ctx.save();
            ctx.translate(n.x, n.y);
            ctx.rotate(Math.PI / 4);
            ctx.fillStyle = rgba(col, (n.rsi ? 0.95 : 0.7) * k);
            ctx.fillRect(-s, -s, s * 2, s * 2);
            ctx.restore();
            if (W >= 700 && n.group !== 'op' && !n.ref.includes('#mod')) {
                const id = (n.ref.split('/').pop() || '').slice(0, 6);
                ctx.font = '9px Menlo, Consolas, monospace';
                ctx.textAlign = Math.cos(n.angle) > 0.2 ? 'left' : Math.cos(n.angle) < -0.2 ? 'right' : 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = rgba(col, 0.55 * k);
                ctx.fillText(id, n.x + Math.cos(n.angle) * 9, n.y + Math.sin(n.angle) * 9);
            }
        }
        // agents
        for (const ag of agents.values()) {
            if (ag.tx === undefined) continue;
            ag.px += (ag.tx - ag.px) * 0.08;
            ag.py += (ag.ty - ag.py) * 0.08;
            const k = ease((t - ag.born) / 600);
            const col = SPECIES_COL[ag.hero] || COL.white;
            const size = { Attr: 5.5, Cyc: 4.6, Res: 3.8, Seed: 3.2, Inr: 2.4 }[ag.rank] || 3;
            const isHover = mouse && Math.hypot(mouse.x - ag.px, mouse.y - ag.py) < size + 4;
            if (isHover) hover = { kind: 'agent', ag };
            if (ag.state === 'leak' || ag.state === 'stock') {
                const hc = ag.state === 'leak' ? COL.red : COL.green;
                ctx.strokeStyle = rgba(hc, 0.55 + 0.25 * Math.sin(t / 260));
                ctx.lineWidth = 1.3;
                ctx.beginPath(); ctx.arc(ag.px, ag.py, size + 4, 0, TAU); ctx.stroke();
            }
            const g = ctx.createRadialGradient(ag.px, ag.py, 0, ag.px, ag.py, size * 3.5);
            g.addColorStop(0, rgba(col, 0.35 * k));
            g.addColorStop(1, 'transparent');
            ctx.fillStyle = g;
            ctx.beginPath(); ctx.arc(ag.px, ag.py, size * 3.5, 0, TAU); ctx.fill();
            ctx.fillStyle = rgba(mix(col, COL.white, isHover ? 0.5 : 0.15), k);
            ctx.beginPath(); ctx.arc(ag.px, ag.py, size * (isHover ? 1.4 : 1), 0, TAU); ctx.fill();
        }
        canvas.style.cursor = hover ? 'pointer' : 'default';
        renderTip();
        // travelling pulses
        pulses = pulses.filter(p => t - p.t < p.dur + 300);
        pulses.forEach(p => {
            const u = ease((t - p.t) / p.dur);
            const x = p.from.x + (p.to.x - p.from.x) * u;
            const y = p.from.y + (p.to.y - p.from.y) * u;
            const fade = 1 - Math.max(0, (t - p.t - p.dur) / 300);
            ctx.strokeStyle = rgba(p.col, 0.35 * fade);
            ctx.lineWidth = 1.2;
            ctx.beginPath(); ctx.moveTo(p.from.x, p.from.y); ctx.lineTo(x, y); ctx.stroke();
            const g = ctx.createRadialGradient(x, y, 0, x, y, 9);
            g.addColorStop(0, rgba(p.col, 0.9 * fade));
            g.addColorStop(1, 'transparent');
            ctx.fillStyle = g;
            ctx.beginPath(); ctx.arc(x, y, 9, 0, TAU); ctx.fill();
        });
        flashes = flashes.filter(f => t - f.t < 900);
        flashes.forEach(f => {
            const k = (t - f.t) / 900;
            ctx.strokeStyle = rgba(f.col, 0.7 * (1 - k));
            ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(f.x, f.y, 6 + 22 * ease(k), 0, TAU); ctx.stroke();
        });
        ripples = ripples.filter(rp => t - rp.t < 1800);
        ripples.forEach(rp => {
            const k = (t - rp.t) / 1800;
            ctx.strokeStyle = rgba(rp.col, 0.45 * (1 - k));
            ctx.lineWidth = 2 * (1 - k) + 0.5;
            ctx.beginPath(); ctx.arc(cx, cy, 6 + ease(k) * R * 0.95, 0, TAU); ctx.stroke();
        });
        // core: the driver model; brighter with the clamped LoRA scale
        const pulse = 0.5 + 0.5 * Math.sin(t / 700);
        const cr = 14 + 10 * clampScale;
        const cg = ctx.createRadialGradient(cx, cy, 0, cx, cy, cr);
        cg.addColorStop(0, `rgba(255,255,255,${0.7 + 0.2 * pulse})`);
        cg.addColorStop(0.4, rgba(COL.violet, 0.45 + 0.3 * clampScale));
        cg.addColorStop(1, 'transparent');
        ctx.fillStyle = cg;
        ctx.beginPath(); ctx.arc(cx, cy, cr, 0, TAU); ctx.fill();
    }

    function renderTip() {
        if (!tip) return;
        if (!hover) { tip.classList.remove('is-visible'); return; }
        let html = '';
        let x = 0, y = 0;
        if (hover.kind === 'agent') {
            const a = hover.ag;
            x = a.px; y = a.py;
            html = `<b>${esc(a.id)} · ${esc(a.state)}</b><code class="rsi-tip__name">${esc(a.name)}</code>`
                + `<span>S ${a.S} · L ${a.L} · x ${a.x}</span>`
                + (a.ax ? `<span>⟼ ${esc(a.ax)}</span>` : '');
        } else {
            const n = hover.n;
            x = n.x; y = n.y;
            html = `<b>${esc(n.ref)}</b><span>${esc((n.text || '').slice(0, 160))}</span>` + (n.rsi ? '<span class="rsi-tip__rsi">added by the loop</span>' : '');
        }
        tip.innerHTML = html;
        tip.style.left = Math.max(8, Math.min(W - 260, x + 14)) + 'px';
        tip.style.top = Math.max(8, Math.min(H - 120, y - 12)) + 'px';
        tip.classList.add('is-visible');
    }

    function frame() {
        raf = requestAnimationFrame(frame);
        if (!mount || mount.offsetParent === null || document.hidden) return;
        const t = now();
        stepComet();
        drawBackground();
        drawDomains(t);
        drawLoop(t);
        drawComet();
        drawStages(t);
    }

    function updateHud() {
        if (!hud.version) return;
        hud.version.textContent = version && version !== 'v-none' ? version : '—';
        hud.loop.textContent = running || iteration ? `loop ${Math.min(iteration + (running ? 1 : 0), maxIter)} / ${maxIter}` : 'idle';
        hud.posture.textContent = posture ? Number(posture.score).toFixed(3) : '—';
        hud.contra.textContent = posture ? `${posture.contradictions} contradiction${posture.contradictions === 1 ? '' : 's'}` : '';
        const meta = active ? STAGE_META[active] : null;
        hud.stage.textContent = meta ? meta.label : (running ? '…' : (iteration ? 'converged' : 'awaiting a security policy'));
        hud.hint.textContent = meta ? meta.hint : (iteration ? 'findings, controls and the hardened policy are below' : 'paste a policy, a runbook or a standard');
        hud.root.classList.toggle('is-running', running);
    }

    function init(el, opts = {}) {
        mount = typeof el === 'string' ? document.getElementById(el) : el;
        if (!mount) return;
        onState = opts.onState || null;
        onStage = opts.onStage || null;
        mount.classList.add('rsi-lattice');
        const legend = Object.entries(SPECIES_COL).map(([s, c]) => `<span title="${s}"><i class="rsi-dot" style="background:rgb(${c.join(',')})"></i>${s}</span>`).join('');
        mount.innerHTML = `
            <canvas class="rsi-lattice__canvas"></canvas>
            <div class="rsi-lattice__grain"></div>
            <div class="rsi-lattice__hud">
                <div class="rsi-hud__corner">
                    <span class="rsi-hud__eyebrow">policy</span>
                    <span class="rsi-hud__version" data-k="version">—</span>
                    <span class="rsi-hud__loop" data-k="loop">idle</span>
                </div>
                <div class="rsi-hud__corner rsi-hud__corner--right">
                    <span class="rsi-hud__eyebrow">posture</span>
                    <span class="rsi-hud__version" data-k="posture">—</span>
                    <span class="rsi-hud__loop" data-k="contra"></span>
                </div>
                <div class="rsi-hud__stage">
                    <span class="rsi-hud__stage-name" data-k="stage"></span>
                    <span class="rsi-hud__stage-hint" data-k="hint"></span>
                </div>
                <div class="rsi-hud__legend">
                    <span><i class="rsi-ring rsi-ring--leak"></i>leaking</span>
                    <span><i class="rsi-ring rsi-ring--stock"></i>stocking</span>
                    <span><i class="rsi-chord"></i>contradiction</span>
                    <span><i class="rsi-dia rsi-dia--rsi"></i>control added</span>
                    <div class="rsi-hud__species">${legend}</div>
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
        queue = []; active = null; ax = new Map(); agents = new Map(); edges = []; toc = []; chords = [];
        pulses = []; ripples = []; echoes = []; flashes = []; seen = new Set();
        version = null; iteration = 0; posture = null; clampScale = 0; trail = [];
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
        SPECIES_COL,
    };
})();

if (typeof window !== 'undefined') window.RsiLattice = RsiLattice;
