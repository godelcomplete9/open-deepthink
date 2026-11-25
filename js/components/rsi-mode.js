/**
 * RsiMode: controller for the "♾️ RSI Loop" tab (cybersecurity).
 *
 * Starts runs via POST /start_rsi, receives `RSI_EVENT:` lines from the shared
 * /stream_log SSE feed (routed here by index.html), and keeps the Ouroboros
 * Lattice, the stage rail, the instrument cards and the result panes in sync.
 */
const RsiMode = (() => {
    'use strict';

    const $ = id => document.getElementById(id);
    const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const FUNCTION = {
        Ni: 'threat forecasting', Ne: 'red-team testing', Si: 'audit logging', Se: 'monitoring & alerting',
        Ti: 'access rules', Te: 'incident response', Fi: 'identity & least privilege', Fe: 'coordination & rollout',
    };
    // Drive presets mirror deepthink/nomosloop/world.py DRIVES.
    const PRESETS = {
        leak: { s_min: 0.62, zeta: 0.3, theta_p: 0.2, q: 0.03 },
        bal: { s_min: 0.5, zeta: 0.15, theta_p: 0.3, q: 0.02 },
        stock: { s_min: 0.38, zeta: 0.08, theta_p: 0.45, q: 0.01 },
    };
    const SLIDERS = ['s_min', 'zeta', 'theta_p', 'q', 'r_scale'];

    let active = false;
    let report = '';
    let reportName = 'rsi_report.md';
    let loopLog = [];
    let drive = 'bal';
    let touched = new Set();

    function setStatus(text, type) {
        const el = $('rsi-status');
        if (!el) return;
        el.textContent = text || '';
        if (typeof setAppStatus === 'function') setAppStatus(el, type);
    }

    // ── drive + sensitivity ──────────────────────────────────────────────
    function setDrive(d) {
        drive = d;
        document.querySelectorAll('[data-rsi-drive]').forEach(b => b.classList.toggle('active', b.dataset.rsiDrive === d));
        const preset = PRESETS[d];
        Object.entries(preset).forEach(([k, v]) => {
            if (touched.has(k)) return;
            const el = $(`rsi-s-${k}`);
            if (el) { el.value = v; showSlider(k); }
        });
        const hint = $('rsi-drive-hint');
        if (hint) {
            hint.textContent = {
                leak: 'Defensive: agents that leak heal first. Expect many findings and few new controls.',
                bal: 'Balanced: heal what leaks, evolve where there is surplus.',
                stock: 'Evolving: agents with surplus write controls. Expect posture to rise and fewer findings.',
            }[d];
        }
    }

    function showSlider(k) {
        const el = $(`rsi-s-${k}`), out = $(`rsi-s-${k}-v`);
        if (el && out) out.textContent = Number(el.value).toFixed(k === 'q' ? 3 : 2);
    }

    function sensitivity() {
        const out = {};
        SLIDERS.forEach(k => {
            const el = $(`rsi-s-${k}`);
            if (el && (touched.has(k) || k in PRESETS[drive])) out[k] = parseFloat(el.value);
        });
        return out;
    }

    // ── stage rail ───────────────────────────────────────────────────────
    function buildRail() {
        const rail = $('rsi-rail');
        if (!rail) return;
        rail.innerHTML = RsiLattice.STAGES.map((name, i) => {
            const sep = i === 0 ? '' : (i === 2 ? '<span class="rsi-rail__sep">⟶</span>' : '<span class="rsi-rail__sep">·</span>');
            return `${sep}<span class="rsi-rail__step" data-stage="${name}">${RsiLattice.STAGE_META[name].label}</span>`;
        }).join('') + '<span class="rsi-rail__loopback" title="lora routes back to correspond while loops remain">↺</span>';
    }

    function markStage(name) {
        document.querySelectorAll('.rsi-rail__step').forEach(el => {
            const on = el.dataset.stage === name;
            if (el.classList.contains('is-active') && !on) el.classList.add('is-done');
            el.classList.toggle('is-active', on);
        });
    }

    // ── instrument cards ─────────────────────────────────────────────────
    function renderPosture(st) {
        const post = (st && st.posture) || [];
        const box = $('rsi-posture');
        if (!box || !post.length) return;
        const last = post[post.length - 1], first = post[0];
        const delta = last.score - first.score;
        $('rsi-posture-version').textContent = `${post.length} version${post.length === 1 ? '' : 's'}`;
        const w = 260, h = 64, pad = 6;
        const vals = post.map(p => p.score);
        const lo = Math.min(...vals) - 0.02, hi = Math.max(...vals) + 0.02;
        const x = i => pad + (post.length === 1 ? (w - 2 * pad) / 2 : i * (w - 2 * pad) / (post.length - 1));
        const y = v => h - pad - (v - lo) / (hi - lo || 1) * (h - 2 * pad);
        let d = `M ${x(0)} ${y(vals[0])}`;
        for (let i = 1; i < vals.length; i++) d += ` H ${x(i)} V ${y(vals[i])}`;
        const missing = (last.missing || []).map(s => `<span class="rsi-chip-mini" title="${esc(FUNCTION[s])}">${s}</span>`).join('') || '<span class="rsi-empty">none</span>';
        box.innerHTML = `
            <div><span class="rsi-capacity__value">${Number(last.score).toFixed(3)}</span>
            <span class="rsi-capacity__delta ${delta > 1e-9 ? 'is-up' : 'is-flat'}">${delta > 1e-9 ? '▲ +' + delta.toFixed(3) : '— baseline'}</span></div>
            <div class="rsi-capacity__meta">diversity ${Number(last.diversity).toFixed(2)} · ${last.contradictions} contradictory pair${last.contradictions === 1 ? '' : 's'}</div>
            <svg class="rsi-spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img" aria-label="Posture per policy version">
                <defs><linearGradient id="rsi-spark-fill" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0" stop-color="#34d399" stop-opacity="0.3"/><stop offset="1" stop-color="#34d399" stop-opacity="0"/>
                </linearGradient></defs>
                <path d="${d} V ${h} H ${x(0)} Z" fill="url(#rsi-spark-fill)"/>
                <path d="${d}" fill="none" stroke="#34d399" stroke-width="1.5" vector-effect="non-scaling-stroke"/>
            </svg>
            <div class="rsi-capacity__meta">missing functions: ${missing}</div>`;
    }

    function renderFunctions(st) {
        const post = (st && st.posture) || [];
        const box = $('rsi-functions');
        if (!box || !post.length || !post[post.length - 1].functions) return;
        const f = post[post.length - 1].functions;
        const max = Math.max(...Object.values(f), 0.01);
        box.innerHTML = Object.entries(f).map(([sp, share]) => {
            const c = (RsiLattice.SPECIES_COL[sp] || [255, 255, 255]).join(',');
            return `<div class="rsi-fn" title="${esc(sp)}: ${esc(FUNCTION[sp])}">
                <span class="rsi-fn__sp" style="color:rgb(${c})">${sp}</span>
                <span class="rsi-fn__name">${esc(FUNCTION[sp])}</span>
                <span class="rsi-fn__bar"><i style="width:${(share / max * 100).toFixed(1)}%;background:rgb(${c})"></i></span>
                <span class="rsi-fn__val">${(share * 100).toFixed(0)}%</span></div>`;
        }).join('');
    }

    function renderAgents(st) {
        const box = $('rsi-agents');
        if (!box || !st || !Array.isArray(st.agents)) return;
        const by = { leak: 0, stock: 0, idle: 0 };
        const ranks = {};
        st.agents.forEach(a => { by[a.state] = (by[a.state] || 0) + 1; ranks[a.rank] = (ranks[a.rank] || 0) + 1; });
        $('rsi-agents-count').textContent = `${st.agents.length} agents`;
        const ens = (st.ensembles || []).slice(0, 4).map(e =>
            `<details class="rsi-ens"><summary><b>${esc(e.title)}</b> · ${e.members.length} members · ${esc(e.climate)} · L ${e.L}</summary><pre>${esc(e.text)}</pre></details>`).join('');
        box.innerHTML = `
            <div class="rsi-states">
                <span class="rsi-state rsi-state--leak">${by.leak} leaking</span>
                <span class="rsi-state rsi-state--stock">${by.stock} stocking</span>
                <span class="rsi-state">${by.idle} idle</span>
            </div>
            <div class="rsi-capacity__meta">${['Attr', 'Cyc', 'Res', 'Seed', 'Inr'].filter(r => ranks[r]).map(r => `${ranks[r]} ${r}`).join(' · ')} · rounds/solve ${st.rounds_per_solve ?? '—'}</div>
            ${ens || '<span class="rsi-empty">No ensemble yet: agents cluster on stems and duals.</span>'}`;
    }

    function renderLists(st) {
        if (!st) return;
        const fbox = $('rsi-findings');
        const ins = (st.inspections || []).slice().sort((a, b) => (b.severity || 0) - (a.severity || 0));
        $('rsi-tab-findings-n').textContent = ins.length ? ` ${ins.length}` : '';
        if (fbox) {
            fbox.innerHTML = ins.length ? ins.map(f => `
                <div class="rsi-finding">
                    <div class="rsi-finding__head"><span class="rsi-sev" style="--sev:${Math.min(1, f.severity || 0)}">${Number(f.severity || 0).toFixed(2)}</span>
                    <b>${esc(f.agent)}</b> heal <code>${esc(f.op)}</code> · ${esc(f.cause)} on ${esc(f.species)} · Leak ${f.leak_before} → ${f.leak_after}${f.memory ? ' · <span title="reused from distilled memory">recalled</span>' : ''}</div>
                    ${(f.clashes || []).map(c => `<div class="rsi-clash"><code>${esc(c.a1)}</code> × <code>${esc(c.a2)}</code> <span>f=${Number(c.f).toFixed(2)}</span></div>`).join('')}
                    <p>${esc(f.finding)}</p>
                </div>`).join('') : '<span class="rsi-empty">No agent leaked enough to need healing.</span>';
        }
        const cbox = $('rsi-controls');
        const ctrls = st.controls || [];
        $('rsi-tab-controls-n').textContent = ctrls.length ? ` ${ctrls.length}` : '';
        if (cbox) {
            cbox.innerHTML = ctrls.length ? ctrls.slice().reverse().map(c => {
                const col = (RsiLattice.SPECIES_COL[c.function] || [255, 255, 255]).join(',');
                return `<div class="rsi-control"><span class="rsi-control__id">${esc(c.id)}</span>
                    <span class="rsi-control__fn" style="color:rgb(${col})">${esc(c.function)} · ${esc(FUNCTION[c.function])}</span>
                    <p>${esc(c.text)}</p>
                    <span class="rsi-capacity__meta">by ${esc(c.agent)} via <code>${esc(c.op)}</code> on <code>${esc(c.clause)}</code> · posture ${Number(c.posture_before).toFixed(3)} → ${Number(c.posture_after).toFixed(3)}</span></div>`;
            }).join('') : '<span class="rsi-empty">No control improved the measured posture yet.</span>';
        }
        const tbox = $('rsi-toc');
        if (tbox && st.toc) {
            tbox.innerHTML = `<table class="rsi-table"><thead><tr><th>TC</th><th>Artifact</th><th>Agent</th><th>Species projection</th><th>A</th><th>conf</th><th>mapper</th></tr></thead><tbody>${
                st.toc.slice(0, 200).map(r => `<tr><td>${r.n}</td><td><code>${esc(r.ax)}</code></td><td>${esc(r.agent)}</td><td>${Object.entries(r.p).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(', ')}</td><td>${Number(r.A).toFixed(2)}</td><td>${Number(r.conf).toFixed(2)}</td><td>${esc(r.mapper)}</td></tr>`).join('')
            }</tbody></table>`;
        }
        const lbox = $('rsi-lora');
        if (lbox && st.lora) {
            const L = st.lora;
            lbox.innerHTML = `<div class="rsi-capacity__meta">examples ${L.examples || 0} · scale ${Number(L.scale || 0).toFixed(2)} (last good ${Number(L.good_scale || 0).toFixed(2)}) · trained ${L.trained || 0}×</div>
                ${(L.history || []).map(h => `<div class="rsi-lora-row"><code>t=${h.t}</code> <b>${esc(h.decision)}</b> scale → ${Number(h.clamp?.scale ?? 0).toFixed(2)} · pushed ${h.clamp?.pushed ? 'yes' : 'no'}${h.clamp?.reason ? ` (${esc(h.clamp.reason)})` : ''} · rounds/solve ${h.metric?.rounds_per_solve} · leak ${h.metric?.leak} · posture ${h.metric?.capacity}</div>`).join('')}
                ${(L.programs || []).length ? `<div class="rsi-capacity__meta" style="margin-top:0.6rem">weight programs</div>` : ''}
                ${(L.programs || []).map(p => {
                    const t = p.train || {};
                    const mix = (p.mix || []).map(m => `${Number(m.id)}×${Number(m.scale).toFixed(2)}`).join(' + ') || 'unchanged';
                    const fixes = (p.corrections || []).length ? ` · ${p.corrections.length} correction(s): ${esc(p.corrections.slice(0, 3).join('; '))}` : '';
                    return `<div class="rsi-lora-row"><code>#${Number(p.n)}</code> <b>${esc(p.source)}</b> · examples ${Number(p.examples_in)} → ${Number(p.examples_out)} · r ${Number(t.rank)} α ${Number(t.alpha)} lr ${esc(String(t.learning_rate ?? '—'))} · ${esc((t.target_modules || []).join(','))} · mix ${esc(mix)}${fixes}${p.rationale ? `<br><span class="app-hint">${esc(p.rationale)}</span>` : ''}</div>`;
                }).join('')}`;
        }
    }

    function pushLoopLog(ev) {
        if (!ev || ev.phase !== 'exit') return;
        (ev.log || []).forEach(line => loopLog.push({ cls: 'is-node', text: `▸ ${line}` }));
        const box = $('rsi-looplog');
        const events = (ev.state && ev.state.events) || [];
        if (box) {
            const evLines = events.slice(-40).map(e => {
                const cls = e.kind === 'leak' || e.kind === 'clash' ? 'is-rollback' : e.kind === 'stock' ? 'is-elevate' : '';
                return `<div class="${cls}">${esc(e.text)}</div>`;
            }).join('');
            box.innerHTML = loopLog.map(l => `<div class="${l.cls}">${esc(l.text)}</div>`).join('') + '<div class="rsi-looplog__sep">— recent NX-1 events —</div>' + evLines;
            box.scrollTop = box.scrollHeight;
        }
    }

    function onState(st, ev) {
        if (st) { renderPosture(st); renderFunctions(st); renderAgents(st); renderLists(st); }
        if (ev && ev.type === 'complete') { markStage(null); return; }
        pushLoopLog(ev);
    }

    // ── results ──────────────────────────────────────────────────────────
    function showPane(name) {
        document.querySelectorAll('[data-rsi-pane]').forEach(b => b.classList.toggle('active', b.dataset.rsiPane === name));
        document.querySelectorAll('[data-rsi-pane-body]').forEach(p => p.classList.toggle('hidden', p.dataset.rsiPaneBody !== name));
    }

    function renderResult(result) {
        $('rsi-results').classList.remove('hidden');
        if (result.error) {
            $('rsi-report').innerHTML = `<p style="color: var(--app-accent-danger)">RSI run failed: ${esc(result.error)}</p>`;
            showPane('report');
            return;
        }
        report = result.artifact_code || '';
        reportName = (result.artifact_path || 'rsi_report.md').split(/[\\/]/).pop();
        // The report quotes policy text and model output: never inject it unsanitised.
        $('rsi-report').innerHTML = (window.marked && marked.parse && window.DOMPurify)
            ? DOMPurify.sanitize(marked.parse(report))
            : `<pre>${esc(report)}</pre>`;
        $('rsi-download-btn').disabled = !report;
        if (result.snapshot) renderLists(result.snapshot);
        showPane((result.findings || []).length ? 'findings' : 'controls');
    }

    function download() {
        if (!report) return;
        const blob = new Blob([report], { type: 'text/markdown;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = reportName;
        document.body.appendChild(a); a.click(); a.remove();
        URL.revokeObjectURL(url);
    }

    // ── run lifecycle ────────────────────────────────────────────────────
    function setRunning(on) {
        active = on;
        const btn = $('rsi-start-btn');
        if (btn) { btn.disabled = on; btn.textContent = on ? 'Looping…' : 'Start RSI Loop'; }
    }

    async function start(e) {
        e.preventDefault();
        const text = $('rsi-text').value;
        if (!text.trim()) { setStatus('Paste a security policy first.', 'warn'); return; }
        const val = id => ($(id) ? $(id).value.trim() : '');
        const num = (id, d) => { const v = parseFloat(val(id)); return Number.isFinite(v) ? v : d; };
        const payload = {
            text,
            law_title: val('rsi-title') || 'policy',
            objective: val('rsi-objective'),
            drive,
            iterations: Math.round(num('rsi-iterations', 2)),
            ticks: Math.round(num('rsi-ticks', 3)),
            mutators: Math.round(num('rsi-mutators', 8)),
            lambda_am: num('rsi-lambda', 0.7),
            sensitivity: sensitivity(),
            offline: $('rsi-offline').checked,
            llamacpp_url: val('llamacpp-url'),
            llamacpp_model: val('llamacpp-model'),
            api_key: val('openrouter-api-key'),
            mapper_model: val('rsi-mapper-model'),
            mapper_arc_agi2: val('rsi-mapper-arc'),
            lora: {
                enabled: $('rsi-lora-on').checked,
                adapter_id: Math.round(num('rsi-lora-id', 0)),
                clamp_max: num('rsi-lora-max', 1.0),
            },
        };
        if ($('rsi-wsp-on') && $('rsi-wsp-on').checked) {
            payload.lora.program = {
                enabled: true,
                model: val('rsi-editor-model'),
                adapter_ids: val('rsi-wsp-adapters').split(',').map(s => parseInt(s, 10)).filter(Number.isFinite),
            };
        }
        loopLog = [];
        $('rsi-looplog').innerHTML = '';
        $('rsi-results').classList.add('hidden');
        document.querySelectorAll('.rsi-rail__step').forEach(el => el.classList.remove('is-active', 'is-done'));
        RsiLattice.reset();
        setRunning(true);
        setStatus(payload.offline ? 'Running offline engine…' : 'Running on the local driver…');
        if (typeof addLogMessage === 'function') addLogMessage(`--- ♾️ RSI security loop (${drive}, ${payload.iterations} loops, λ_AM ${payload.lambda_am}, offline=${payload.offline}) ---`, { color: 'cyan' });
        try {
            const resp = await fetch('/start_rsi', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(data.message || `HTTP ${resp.status}`);
            setStatus(`Looping “${data.law_title}” · mapper: ${data.mapper}`, null);
        } catch (err) {
            setRunning(false);
            RsiLattice.stop();
            setStatus(err.message, 'error');
        }
    }

    function handleEvent(jsonStr) {
        let data;
        try { data = JSON.parse(jsonStr.trim()); } catch (e) { console.error('RSI event parse', e); return; }
        if (data.type === 'rsi_event') {
            if (!RsiLattice.isRunning()) { RsiLattice.reset(); setRunning(true); }
            RsiLattice.event(data);
        } else if (data.type === 'rsi_complete') {
            const result = data.result || {};
            RsiLattice.complete(result.snapshot || null);
            const wait = () => {
                if (RsiLattice.isRunning()) { setTimeout(wait, 200); return; }
                setRunning(false);
                renderResult(result);
                if (result.error) setStatus('Run failed.', 'error');
                else setStatus(`Done · ${result.findings.length} findings · ${result.controls.length} controls · posture ${Number(result.posture_baseline).toFixed(3)} → ${Number(result.posture_final).toFixed(3)}`, 'success');
            };
            wait();
        }
    }

    // ── inputs ───────────────────────────────────────────────────────────
    async function loadExamples() {
        const box = $('rsi-examples');
        if (!box) return;
        try {
            const data = await (await fetch('/rsi_examples')).json();
            (data.examples || []).forEach((ex, i) => {
                const b = document.createElement('button');
                b.type = 'button';
                b.className = 'rsi-chip';
                b.textContent = ex.name.replace(/\.(txt|md)$/, '').replace(/_/g, ' ');
                b.addEventListener('click', () => {
                    $('rsi-text').value = ex.text;
                    $('rsi-title').value = ex.name.replace(/\.(txt|md)$/, '');
                });
                box.appendChild(b);
                if (ex.name.startsWith('institution') && !$('rsi-text').value) b.click();
                if (i === (data.examples.length - 1) && !$('rsi-text').value) box.querySelector('.rsi-chip')?.click();
            });
            if (!(data.examples || []).length) box.classList.add('hidden');
        } catch (e) {
            box.classList.add('hidden');
        }
    }

    function init() {
        if (!$('rsi-mode')) return;
        RsiLattice.init('rsi-lattice', { onState, onStage: markStage });
        buildRail();
        $('rsi-form').addEventListener('submit', start);
        $('rsi-download-btn').addEventListener('click', download);
        $('rsi-wsp-on')?.addEventListener('change', ev => $('rsi-wsp-fields')?.classList.toggle('hidden', !ev.target.checked));
        document.querySelectorAll('[data-rsi-pane]').forEach(b => b.addEventListener('click', () => showPane(b.dataset.rsiPane)));
        document.querySelectorAll('[data-rsi-drive]').forEach(b => b.addEventListener('click', () => setDrive(b.dataset.rsiDrive)));
        SLIDERS.forEach(k => {
            const el = $(`rsi-s-${k}`);
            if (!el) return;
            el.addEventListener('input', () => { touched.add(k); showSlider(k); });
            showSlider(k);
        });
        const lam = $('rsi-lambda');
        if (lam) {
            const show = () => { $('rsi-lambda-v').textContent = Number(lam.value).toFixed(2); };
            lam.addEventListener('input', show);
            show();
        }
        $('rsi-reset-sens')?.addEventListener('click', () => { touched = new Set(); setDrive(drive); $('rsi-s-r_scale').value = 0.1; showSlider('r_scale'); });
        $('rsi-file-input').addEventListener('change', ev => {
            const f = ev.target.files && ev.target.files[0];
            if (!f) return;
            const reader = new FileReader();
            reader.onload = () => {
                $('rsi-text').value = String(reader.result || '');
                $('rsi-title').value = f.name.replace(/\.[^.]+$/, '');
                setStatus(`Loaded ${f.name} (${(f.size / 1024).toFixed(1)} KB)`, 'success');
            };
            reader.readAsText(f);
            ev.target.value = '';
        });
        setDrive('bal');
        loadExamples();
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();

    return { handleEvent, isActive: () => active };
})();

if (typeof window !== 'undefined') window.RsiMode = RsiMode;
