/**
 * RsiMode — controller for the "♾️ RSI Loop" tab.
 *
 * Starts runs via POST /start_rsi, receives `RSI_EVENT:` lines from the shared
 * /stream_log SSE feed (routed here by index.html), and keeps the Ouroboros
 * Lattice canvas, stage rail, instrument cards, and result panes in sync.
 */
const RsiMode = (() => {
    'use strict';

    const $ = id => document.getElementById(id);
    const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const LEVEL_COLORS = ['#8b5cf6', '#6d74f4', '#3b82f6', '#2196e0', '#06b6d4'];

    let active = false;
    let artifactCode = '';
    let artifactName = 'rsi_artifact.py';
    let loopLog = [];

    function levelColor(i, n) {
        if (n <= 1) return LEVEL_COLORS[0];
        return LEVEL_COLORS[Math.round(i / (n - 1) * (LEVEL_COLORS.length - 1))];
    }

    function setStatus(text, type) {
        const el = $('rsi-status');
        if (!el) return;
        el.textContent = text || '';
        if (typeof setAppStatus === 'function') setAppStatus(el, type);
    }

    // ── stage rail ───────────────────────────────────────────────────────
    function buildRail() {
        const rail = $('rsi-rail');
        if (!rail) return;
        const parts = RsiLattice.STAGES.map((name, i) => {
            const sep = i === 0 ? '' : (i === 2 ? '<span class="rsi-rail__sep">⟶</span>' : '<span class="rsi-rail__sep">·</span>');
            return `${sep}<span class="rsi-rail__step" data-stage="${name}">${RsiLattice.STAGE_META[name].label}</span>`;
        });
        rail.innerHTML = parts.join('') + '<span class="rsi-rail__loopback" title="rectify routes back to mutate while loops remain">↺</span>';
    }

    function markStage(name) {
        document.querySelectorAll('.rsi-rail__step').forEach(el => {
            const isActive = el.dataset.stage === name;
            if (el.classList.contains('is-active') && !isActive) el.classList.add('is-done');
            el.classList.toggle('is-active', isActive);
        });
    }

    // ── instrument cards ─────────────────────────────────────────────────
    function renderCapacity(st) {
        const caps = (st && st.capacity) || [];
        const box = $('rsi-capacity');
        if (!box || !caps.length) return;
        const last = caps[caps.length - 1];
        const base = st.baseline_capacity || caps[0].capacity;
        const delta = last.capacity - base;
        $('rsi-cap-instances').textContent = `${last.instances} instances`;

        const w = 260, h = 70, pad = 6;
        const vals = caps.map(c => c.capacity);
        const lo = Math.min(...vals, base) - 0.05, hi = Math.max(...vals, base) + 0.05;
        const x = i => pad + (caps.length === 1 ? (w - 2 * pad) / 2 : i * (w - 2 * pad) / (caps.length - 1));
        const y = v => h - pad - (v - lo) / (hi - lo || 1) * (h - 2 * pad);
        // step line: capacity holds until the next elevation
        let d = `M ${x(0)} ${y(vals[0])}`;
        for (let i = 1; i < vals.length; i++) d += ` H ${x(i)} V ${y(vals[i])}`;
        const area = `${d} V ${h} H ${x(0)} Z`;
        const dots = caps.map((c, i) =>
            `<circle cx="${x(i)}" cy="${y(c.capacity)}" r="${i === caps.length - 1 ? 3 : 1.8}" fill="${i === caps.length - 1 ? '#fff' : '#8b5cf6'}"><title>${esc(c.version)} · ${c.capacity.toFixed(3)} · ${esc(c.source)}</title></circle>`
        ).join('');
        box.innerHTML = `
            <div><span class="rsi-capacity__value">${last.capacity.toFixed(3)}</span>
            <span class="rsi-capacity__delta ${delta > 1e-9 ? 'is-up' : 'is-flat'}">${delta > 1e-9 ? '▲ +' + delta.toFixed(3) : '— baseline'}</span></div>
            <div class="rsi-capacity__meta">baseline ${base.toFixed(3)} · coverage ${(last.coverage * 100).toFixed(0)}% · ${caps.length} version${caps.length === 1 ? '' : 's'}</div>
            <svg class="rsi-spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img" aria-label="Capacity per lattice version">
                <defs><linearGradient id="rsi-spark-fill" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0" stop-color="#8b5cf6" stop-opacity="0.35"/><stop offset="1" stop-color="#8b5cf6" stop-opacity="0"/>
                </linearGradient></defs>
                <line x1="0" x2="${w}" y1="${y(base)}" y2="${y(base)}" stroke="rgba(255,255,255,0.18)" stroke-dasharray="3 4"/>
                <path d="${area}" fill="url(#rsi-spark-fill)"/>
                <path d="${d}" fill="none" stroke="#8b5cf6" stroke-width="1.5" vector-effect="non-scaling-stroke"/>
                ${dots}
            </svg>`;
    }

    function renderLevels(st) {
        const lat = st && st.lattice;
        const box = $('rsi-levels');
        if (!box || !lat || !(lat.levels || []).length) return;
        $('rsi-lattice-version').textContent = lat.version || '';
        const n = lat.levels.length;
        box.innerHTML = lat.levels.map((lv, i) => `
            <div class="rsi-level" style="--lvl:${levelColor(i, n)}">
                <span class="rsi-level__id">${esc(lv.id)}</span>
                <span class="rsi-level__name" title="${esc(lv.name)}">${esc(lv.name)} · ${lv.rules.length} rule${lv.rules.length === 1 ? '' : 's'}</span>
                ${lv.base ? `<span class="rsi-level__meta"><b>base</b> ${esc(lv.base)}</span>` : ''}
                ${lv.modulus ? `<span class="rsi-level__meta"><b>mod</b> ${esc(lv.modulus)}</span>` : ''}
            </div>`).join('');
    }

    function renderSwarm(st) {
        const props = (st && st.proposals) || [];
        const box = $('rsi-swarm');
        if (!box || !props.length) return;
        const acc = props.filter(p => p.status === 'accepted').length;
        $('rsi-swarm-count').textContent = `${acc} / ${props.length} accepted`;
        box.innerHTML = props.slice().reverse().map(p => `
            <div class="rsi-prop rsi-prop--${esc(p.status)}" title="${esc(p.id)} · ${esc(p.author)}">
                <span class="rsi-prop__mark"></span>
                <span class="rsi-prop__op">${esc(p.op)} <small>@${esc(p.level)}</small></span>
                <span class="rsi-prop__score">${p.score == null ? '…' : Number(p.score).toFixed(2)}</span>
            </div>`).join('');
    }

    function pushLoopLog(ev) {
        if (!ev || ev.phase !== 'exit') return;
        const loopNo = ev.node === 'rectify' ? ev.iteration : ev.iteration + 1;
        loopLog.push({ cls: 'is-node', text: `▸ ${ev.node}  (loop ${loopNo})` });
        (ev.log || []).forEach(line => {
            const cls = /^elevated/.test(line) ? 'is-elevate' : (/rolled back|rollback/.test(line) ? 'is-rollback' : '');
            loopLog.push({ cls, text: '   ' + line });
        });
        const box = $('rsi-looplog');
        if (box) {
            box.innerHTML = loopLog.map(l => `<div class="${l.cls}">${esc(l.text)}</div>`).join('');
            box.scrollTop = box.scrollHeight;
        }
    }

    function onState(st, ev) {
        if (st) { renderCapacity(st); renderLevels(st); renderSwarm(st); }
        if (ev && ev.type === 'complete') { markStage(null); return; }
        pushLoopLog(ev);
    }

    // ── results ──────────────────────────────────────────────────────────
    function showPane(name) {
        document.querySelectorAll('[data-rsi-pane]').forEach(b => b.classList.toggle('active', b.dataset.rsiPane === name));
        document.querySelectorAll('[data-rsi-pane-body]').forEach(p => p.classList.toggle('hidden', p.dataset.rsiPaneBody !== name));
    }

    function renderResult(result) {
        const box = $('rsi-results');
        if (!box) return;
        box.classList.remove('hidden');
        if (result.error) {
            $('rsi-summary').innerHTML = `<p style="color: var(--app-accent-danger)">RSI run failed: ${esc(result.error)}</p>`;
            showPane('summary');
            return;
        }
        const md = result.proposed_solution || '';
        $('rsi-summary').innerHTML = (window.marked && marked.parse) ? marked.parse(md) : `<pre>${esc(md)}</pre>`;
        artifactCode = result.artifact_code || '';
        artifactName = (result.artifact_path || 'rsi_artifact.py').split(/[\\/]/).pop();
        const code = $('rsi-artifact-code');
        code.textContent = artifactCode;
        if (window.Prism && Prism.highlightElement) Prism.highlightElement(code);
        $('rsi-download-btn').disabled = !artifactCode;
        showPane('summary');
    }

    function download() {
        if (!artifactCode) return;
        const blob = new Blob([artifactCode], { type: 'text/x-python;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = artifactName;
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
        if (!text.trim()) { setStatus('Paste a formal artifact first.', 'warn'); return; }
        const val = id => ($(id) ? $(id).value.trim() : '');
        const payload = {
            text,
            law_title: val('rsi-title') || 'rsi-artifact',
            iterations: parseInt(val('rsi-iterations'), 10) || 2,
            mutators: parseInt(val('rsi-mutators'), 10),
            offline: $('rsi-offline').checked,
            provider: val('llm-provider') || 'openrouter',
            api_key: val('openrouter-api-key'),
            openrouter_model: val('openrouter-model'),
            llamacpp_url: val('llamacpp-url'),
            llamacpp_model: val('llamacpp-model'),
            synthesis_model: val('synthesis-model'),
        };
        if (Number.isNaN(payload.mutators)) payload.mutators = 6;

        loopLog = [];
        $('rsi-looplog').innerHTML = '';
        $('rsi-results').classList.add('hidden');
        document.querySelectorAll('.rsi-rail__step').forEach(el => el.classList.remove('is-active', 'is-done'));
        RsiLattice.reset();
        setRunning(true);
        setStatus(payload.offline ? 'Running offline engine…' : `Running with ${payload.provider}…`);
        if (typeof addLogMessage === 'function') addLogMessage(`--- ♾️ RSI loop starting (${payload.iterations} loops, ${payload.mutators} mutators, offline=${payload.offline}) ---`, { color: 'cyan' });

        try {
            const resp = await fetch('/start_rsi', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(data.message || `HTTP ${resp.status}`);
            setStatus(`Looping “${data.law_title}”`, null);
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
            // results land after the replay queue drains so the loop finishes first
            const wait = () => {
                if (RsiLattice.isRunning()) { setTimeout(wait, 200); return; }
                setRunning(false);
                renderResult(result);
                if (result.error) setStatus('Run failed.', 'error');
                else setStatus(`Converged · ${result.lattice_version} · capacity ${Number(result.baseline_capacity).toFixed(3)} → ${Number(result.final_capacity).toFixed(3)}`, 'success');
            };
            wait();
        }
    }

    // ── inputs ───────────────────────────────────────────────────────────
    async function loadExamples() {
        const box = $('rsi-examples');
        if (!box) return;
        try {
            const resp = await fetch('/rsi_examples');
            const data = await resp.json();
            (data.examples || []).forEach((ex, i) => {
                const b = document.createElement('button');
                b.type = 'button';
                b.className = 'rsi-chip';
                b.textContent = ex.name.replace(/\.(txt|md)$/, '');
                b.addEventListener('click', () => {
                    $('rsi-text').value = ex.text;
                    $('rsi-title').value = b.textContent.replace(/_fragment$/, '');
                });
                box.appendChild(b);
                if (i === 0 && !$('rsi-text').value) b.click();
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
        document.querySelectorAll('[data-rsi-pane]').forEach(b => b.addEventListener('click', () => showPane(b.dataset.rsiPane)));
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
        loadExamples();
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();

    return { handleEvent, isActive: () => active };
})();

if (typeof window !== 'undefined') window.RsiMode = RsiMode;
