/**
 * RsiLattice: "Ouroboros Lattice" for the RSI security loop.
 *
 * A flat line drawing: hairlines, solid dots, one accent colour for the loop's
 * progress, and red / green only where they mean leak / stock.
 *
 *   • Ring   = the loop. The five looping stages (correspond, tick, heal,
 *              stock, lora) sit on a circle; ingest and scale induction
 *              spiral into it as a tail. A dot travels the ring and the
 *              accent arc shows how far the current pass has gone.
 *   • AX band (outer)   = the security policy: clauses grouped by level
 *              (principles, policies, controls, assets) and the viable control
 *              operations. Contradictory clauses are joined by red chords.
 *              Controls added by the loop appear in green.
 *   • TC spokes         = the table of correspondences: one gauge link from
 *              each clause to its agent; a twisted link (cos A <= 0) is red.
 *   • AM lattice (inner) = Agent Mechanics agents, placed on their clause's
 *              spoke at a radius set by rank (Attr near the core, Inr near the
 *              band) and drawn brighter with rank. A leaking agent is red; a
 *              stocking agent is green. Hostile (op) edges are dashed red.
 *
 * Heals draw a red line from the agent to the clashing clauses. Kept controls
 * draw a green line from the stocking agent to the band. The accent arc
 * around the core is the clamped LoRA scale. Events are replayed with a
 * minimum dwell so the offline engine, which finishes in milliseconds, still
 * reads as a loop.
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
            stages[name] = { ...(stages[name] || { done: 0 }), x: cx + Math.cos(a) * R, y: cy + Math.sin(a) * R, angle: a };
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
        stages.ingest = { ...(stages.ingest || { done: 0 }), ...at(0), angle: th1, s: 0 };
        stages.scale_induction = { ...(stages.scale_induction || { done: 0 }), ...at(0.52), angle: th1, s: 0.52 };
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
            ripples.push({ t, col: ev.decision === 'rollback' ? COL.white : COL.violet });
        } else if (ev.kind === 'brk' && ag) {
            flashes.push({ x: ag.px, y: ag.py, col: COL.white, t });
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

    // ── render: a flat line drawing (hairlines, solid dots, no glow) ─────
    const INK = a => `rgba(255,255,255,${a})`;
    const BG = '#08080a';
    const RANK_ALPHA = { Attr: 0.85, Cyc: 0.7, Res: 0.58, Seed: 0.48, Inr: 0.38 };
    const RANK_SIZE = { Attr: 4, Cyc: 3.4, Res: 2.9, Seed: 2.5, Inr: 2 };

    function markerPoint() {
        if (!comet) return null;
        if (comet.mode === 'tail') return tailPts[Math.round(Math.min(1, Math.max(0, comet.s)) * 80)];
        return { x: cx + Math.cos(comet.a) * R, y: cy + Math.sin(comet.a) * R };
    }

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
            if (!running && !active) comet.ta += 0.004;
        }
    }

    function drawBackground() {
        ctx.fillStyle = BG;
        ctx.fillRect(0, 0, W, H);
    }

    function drawLoop() {
        ctx.lineWidth = 1;
        ctx.strokeStyle = INK(0.16);
        ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.stroke();
        ctx.beginPath();
        tailPts.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
        ctx.stroke();
        // progress through the current pass, in the accent colour
        if (running && comet) {
            ctx.strokeStyle = rgba(COL.violet, 0.9);
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            if (comet.mode === 'tail') {
                const n = Math.round(Math.min(1, Math.max(0, comet.s)) * 80);
                for (let i = 0; i <= n; i++) i ? ctx.lineTo(tailPts[i].x, tailPts[i].y) : ctx.moveTo(tailPts[0].x, tailPts[0].y);
            } else {
                let p = ((comet.a + Math.PI / 2) % TAU + TAU) % TAU;
                if (p > TAU - 1e-3) p = 0; // back at correspond: a new pass starts
                if (p > 1e-3) ctx.arc(cx, cy, R, -Math.PI / 2, -Math.PI / 2 + p);
            }
            ctx.stroke();
        }
        // direction of travel: one small arrowhead between lora and correspond
        const a = stages.lora.angle + (TAU / 5) * 0.5;
        ctx.save();
        ctx.translate(cx + Math.cos(a) * R, cy + Math.sin(a) * R); ctx.rotate(a + Math.PI / 2);
        ctx.fillStyle = INK(0.45);
        ctx.beginPath(); ctx.moveTo(4, 0); ctx.lineTo(-3, -3.5); ctx.lineTo(-3, 3.5); ctx.closePath(); ctx.fill();
        ctx.restore();
    }

    function drawMarker() {
        const p = markerPoint();
        if (!p) return;
        ctx.fillStyle = running ? rgba(COL.violet, 1) : INK(0.7);
        ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, TAU); ctx.fill();
    }

    function drawStages() {
        [...TAIL, ...LOOP].forEach(name => {
            const s = stages[name];
            const isActive = active === name;
            ctx.lineWidth = 1;
            if (isActive) {
                ctx.strokeStyle = rgba(COL.violet, 0.6);
                ctx.beginPath(); ctx.arc(s.x, s.y, 9, 0, TAU); ctx.stroke();
            }
            ctx.fillStyle = isActive ? rgba(COL.violet, 1) : BG;
            ctx.strokeStyle = isActive ? rgba(COL.violet, 1) : INK(s.done ? 0.6 : 0.3);
            ctx.beginPath(); ctx.arc(s.x, s.y, 4.5, 0, TAU); ctx.fill(); ctx.stroke();
            if (s.done && !isActive) {
                ctx.fillStyle = INK(0.6);
                ctx.beginPath(); ctx.arc(s.x, s.y, 1.8, 0, TAU); ctx.fill();
            }
            if (W < 560 && !isActive) return;
            const dx = s.x - cx, dy = s.y - cy, d = Math.hypot(dx, dy) || 1;
            const label = STAGE_META[name].label.toUpperCase();
            ctx.font = '500 10px -apple-system, "Segoe UI", sans-serif';
            if ('letterSpacing' in ctx) ctx.letterSpacing = '1.5px';
            const tw = ctx.measureText(label).width;
            let lx = s.x + dx / d * 20;
            const ly = Math.max(10, Math.min(H - 10, s.y + dy / d * 18));
            const align = Math.abs(dx / d) < 0.3 ? 'center' : (dx > 0 ? 'left' : 'right');
            const left = align === 'center' ? lx - tw / 2 : align === 'left' ? lx : lx - tw;
            lx += Math.max(0, 8 - left) - Math.max(0, left + tw - (W - 8));
            ctx.textAlign = align;
            ctx.textBaseline = 'middle';
            ctx.fillStyle = isActive ? INK(0.92) : INK(0.4);
            ctx.fillText(label, lx, ly);
            if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';
        });
    }

    function drawDomains(t) {
        const rax = R * 0.8;
        ctx.lineWidth = 1;
        ctx.strokeStyle = INK(0.07);
        ctx.beginPath(); ctx.arc(cx, cy, rax, 0, TAU); ctx.stroke();
        if (!ax.size) {
            ctx.strokeStyle = INK(0.04);
            for (let i = 0; i < 4; i++) { ctx.beginPath(); ctx.arc(cx, cy, R * (0.2 + 0.13 * i), 0, TAU); ctx.stroke(); }
        }
        // group labels
        if (W >= 560) {
            GROUPS.forEach(g => {
                const nodes = [...ax.values()].filter(n => n.group === g);
                if (!nodes.length) return;
                const mid = nodes[Math.floor(nodes.length / 2)].angle;
                // outside the band unless a stage node sits there; then just inside it
                let lr = rax + 22;
                const near = st => st && Math.hypot(st.x - (cx + Math.cos(mid) * lr), st.y - (cy + Math.sin(mid) * lr)) < 36;
                if ([...TAIL, ...LOOP].some(name => near(stages[name]))) lr = rax - 16;
                ctx.font = '500 9px -apple-system, "Segoe UI", sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = INK(0.28);
                ctx.fillText(`${g === 'op' ? '' : g + ' · '}${GROUP_NAME[g]}`.toUpperCase(), cx + Math.cos(mid) * lr, cy + Math.sin(mid) * lr);
            });
        }
        // TC spokes: one gauge link per clause; a twisted link (cos A <= 0) is red
        ctx.lineWidth = 0.75;
        toc.forEach(r => {
            const n = ax.get(r.ax), ag = agents.get(r.agent);
            if (!n || !ag || ag.px === undefined) return;
            const al = Math.cos(r.A || 0);
            ctx.strokeStyle = al > 0 ? INK(0.03 + 0.03 * al) : rgba(COL.red, 0.14);
            ctx.beginPath(); ctx.moveTo(n.x, n.y); ctx.lineTo(ag.px, ag.py); ctx.stroke();
        });
        // contradiction chords between clauses
        ctx.lineWidth = 1;
        ctx.strokeStyle = rgba(COL.red, 0.55);
        chords.forEach(c => {
            const a = ax.get(c.a), b = ax.get(c.b);
            if (!a || !b) return;
            ctx.beginPath(); ctx.moveTo(a.x, a.y);
            ctx.quadraticCurveTo(cx + (a.x + b.x - 2 * cx) * 0.25, cy + (a.y + b.y - 2 * cy) * 0.25, b.x, b.y);
            ctx.stroke();
        });
        // AM edges: hostile edges dashed red, the rest in neutral ink
        edges.forEach(([s, d, typ]) => {
            const a = agents.get(s), b = agents.get(d);
            if (!a || !b || a.px === undefined || b.px === undefined) return;
            ctx.save();
            if (typ === 'op') { ctx.setLineDash([2, 3]); ctx.strokeStyle = rgba(COL.red, 0.28); }
            else if (typ === 'pr') { ctx.setLineDash([1, 3]); ctx.strokeStyle = INK(0.07); }
            else if (typ === 'du') ctx.strokeStyle = INK(0.12);
            else if (typ === 'st') ctx.strokeStyle = INK(0.07);
            else ctx.strokeStyle = INK(0.03);
            ctx.lineWidth = 0.75;
            ctx.beginPath(); ctx.moveTo(a.px, a.py); ctx.lineTo(b.px, b.py); ctx.stroke();
            ctx.restore();
        });
        // AX nodes: filled squares for clauses, hollow for moduli and operations, green when added
        hover = null;
        for (const n of ax.values()) {
            const k = ease((t - n.born) / 500);
            const isHover = mouse && Math.hypot(mouse.x - n.x, mouse.y - n.y) < 8;
            if (isHover) hover = { kind: 'ax', n };
            const s = isHover ? 3.5 : 2.3;
            const hollow = !n.rsi && (n.group === 'op' || n.ref.includes('#mod'));
            if (hollow) {
                ctx.strokeStyle = INK(0.5 * k);
                ctx.lineWidth = 1;
                ctx.strokeRect(n.x - s, n.y - s, s * 2, s * 2);
            } else {
                ctx.fillStyle = n.rsi ? rgba(COL.green, 0.95 * k) : INK(0.62 * k);
                ctx.fillRect(n.x - s, n.y - s, s * 2, s * 2);
            }
            if (W >= 700 && n.group !== 'op' && !n.ref.includes('#mod')) {
                const id = (n.ref.split('/').pop() || '').slice(0, 6);
                ctx.font = '9px Menlo, Consolas, monospace';
                ctx.textAlign = Math.cos(n.angle) > 0.2 ? 'left' : Math.cos(n.angle) < -0.2 ? 'right' : 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = n.rsi ? rgba(COL.green, 0.7 * k) : INK(0.35 * k);
                ctx.fillText(id, n.x + Math.cos(n.angle) * 9, n.y + Math.sin(n.angle) * 9);
            }
        }
        // agents: solid dots, brighter with rank; red while leaking, green while stocking
        for (const ag of agents.values()) {
            if (ag.tx === undefined) continue;
            ag.px += (ag.tx - ag.px) * 0.08;
            ag.py += (ag.ty - ag.py) * 0.08;
            const k = ease((t - ag.born) / 500);
            const size = RANK_SIZE[ag.rank] || 2.5;
            const isHover = mouse && Math.hypot(mouse.x - ag.px, mouse.y - ag.py) < size + 4;
            if (isHover) hover = { kind: 'agent', ag };
            const col = ag.state === 'leak' ? COL.red : ag.state === 'stock' ? COL.green : null;
            ctx.fillStyle = col ? rgba(col, 0.95 * k) : INK((RANK_ALPHA[ag.rank] || 0.5) * k);
            ctx.beginPath(); ctx.arc(ag.px, ag.py, size, 0, TAU); ctx.fill();
            if (isHover) {
                ctx.strokeStyle = INK(0.8);
                ctx.lineWidth = 1;
                ctx.beginPath(); ctx.arc(ag.px, ag.py, size + 3, 0, TAU); ctx.stroke();
            }
        }
        canvas.style.cursor = hover ? 'pointer' : 'default';
        renderTip();
        // travelling links: a hairline drawn from source to target, then faded
        pulses = pulses.filter(p => t - p.t < p.dur + 300);
        pulses.forEach(p => {
            const u = ease((t - p.t) / p.dur);
            const x = p.from.x + (p.to.x - p.from.x) * u;
            const y = p.from.y + (p.to.y - p.from.y) * u;
            const fade = 1 - Math.max(0, (t - p.t - p.dur) / 300);
            ctx.strokeStyle = rgba(p.col, 0.6 * fade);
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(p.from.x, p.from.y); ctx.lineTo(x, y); ctx.stroke();
            ctx.fillStyle = rgba(p.col, 0.9 * fade);
            ctx.beginPath(); ctx.arc(x, y, 1.6, 0, TAU); ctx.fill();
        });
        // event marks: one thin ring that opens and fades
        flashes = flashes.filter(f => t - f.t < 700);
        flashes.forEach(f => {
            const k = (t - f.t) / 700;
            ctx.strokeStyle = rgba(f.col, 0.6 * (1 - k));
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.arc(f.x, f.y, 5 + 9 * ease(k), 0, TAU); ctx.stroke();
        });
        // core: the local driver. The accent arc is the clamped LoRA scale.
        const cr = 8;
        ctx.lineWidth = 1;
        ctx.strokeStyle = INK(0.3);
        ctx.beginPath(); ctx.arc(cx, cy, cr, 0, TAU); ctx.stroke();
        if (clampScale > 0) {
            ctx.strokeStyle = rgba(COL.violet, 0.95);
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(cx, cy, cr, -Math.PI / 2, -Math.PI / 2 + TAU * Math.min(1, clampScale)); ctx.stroke();
        }
        ripples = ripples.filter(rp => t - rp.t < 900);
        ripples.forEach(rp => {
            const k = (t - rp.t) / 900;
            ctx.strokeStyle = rgba(rp.col, 0.6 * (1 - k));
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.arc(cx, cy, cr + 4 + 12 * ease(k), 0, TAU); ctx.stroke();
        });
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
                + `<span>${esc(a.hero)} · ${esc(a.rank)}</span>`
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
        drawMarker();
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
        mount.innerHTML = `
            <canvas class="rsi-lattice__canvas"></canvas>
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
                    <span><i class="rsi-dot rsi-dot--leak"></i>leaking</span>
                    <span><i class="rsi-dot rsi-dot--stock"></i>stocking</span>
                    <span><i class="rsi-chord"></i>contradiction</span>
                    <span><i class="rsi-sq rsi-sq--rsi"></i>control added</span>
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
        pulses = []; ripples = []; flashes = []; seen = new Set();
        version = null; iteration = 0; posture = null; clampScale = 0;
        Object.values(stages).forEach(s => { s.done = 0; });
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
