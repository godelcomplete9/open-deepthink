/**
 * Theory Synthesis swarm panel.
 *
 * A quiet, live view of one run: every QNN as a square (coloured by the answer
 * it currently backs, pulsing while it works), the vote as bars against the
 * quorum, and the body of work as counts. Fed by SWARM_EVENT lines from the
 * server (see deepthink/theory/pipeline.py).
 */
const TheorySwarm = (() => {
    const PALETTE = ['#a78bfa', '#06b6d4', '#34d399', '#fbbf24', '#f472b6', '#60a5fa', '#f97316', '#e879f9'];
    let root = null;
    let els = {};
    let state = null;

    function el(tag, cls, text) {
        const node = document.createElement(tag);
        if (cls) node.className = cls;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function freshState() {
        return {
            members: 0,
            rounds: 0,
            round: 0,
            quorum: 0.5,
            vote: 'weighted',
            ledger: 'shared',
            domain: '',
            colors: {},
            assignments: {},
            status: {},
            converged: false,
            done: false,
        };
    }

    function mount(containerId) {
        root = document.getElementById(containerId);
        if (!root) return null;
        root.innerHTML = '';
        root.classList.add('swarm-panel');
        els.head = el('div', 'swarm-head');
        els.title = el('span', 'swarm-title', '∑ Swarm');
        els.round = el('span', 'swarm-round', 'idle');
        els.head.append(els.title, els.round);
        els.meta = el('div', 'swarm-meta', 'The vote and the body of work appear here when a run starts.');
        els.grid = el('div', 'swarm-grid');
        els.bars = el('div', 'swarm-bars');
        els.bodyTitle = el('div', 'swarm-section', 'Body of work');
        els.body = el('div', 'swarm-body', '—');
        els.status = el('div', 'swarm-status', '');
        root.append(els.head, els.meta, els.grid, els.bars, els.bodyTitle, els.body, els.status);
        state = freshState();
        return api;
    }

    function start() {
        if (!root) return;
        state = freshState();
        els.round.textContent = 'starting…';
        els.meta.textContent = 'Sizing the problem and spanning personas…';
        els.grid.innerHTML = '';
        els.bars.innerHTML = '';
        els.body.textContent = '—';
        els.status.textContent = '';
        root.classList.add('is-running');
    }

    function colorFor(key) {
        if (!key) return '';
        if (!(key in state.colors)) {
            const used = Object.keys(state.colors).length;
            state.colors[key] = PALETTE[used % PALETTE.length];
        }
        return state.colors[key];
    }

    function buildGrid(n) {
        els.grid.innerHTML = '';
        for (let i = 0; i < n; i++) {
            const sq = el('span', 'swarm-cell');
            sq.dataset.member = String(i);
            sq.title = `QNN ${i}: waiting`;
            els.grid.append(sq);
        }
    }

    function cell(i) {
        return els.grid.querySelector(`[data-member="${i}"]`);
    }

    function paintMember(i) {
        const sq = cell(i);
        if (!sq) return;
        const st = state.status[i] || {};
        const key = state.assignments[String(i)] || '';
        const color = colorFor(key);
        sq.style.background = color || '';
        sq.classList.toggle('is-busy', ['descent', 'ascent', 'verify', 'evolve', 'respawn'].includes(st.status));
        sq.classList.toggle('is-abstain', st.status === 'done' && !key && state.round > 0);
        const bits = [`QNN ${i}`];
        if (st.status) bits.push(st.status);
        if (st.answer) bits.push(`answer ${st.answer}`);
        if (typeof st.verify === 'number') bits.push(`check ${st.verify.toFixed(2)}`);
        if (typeof st.weight === 'number') bits.push(`weight ${st.weight.toFixed(3)}`);
        sq.title = bits.join(' · ');
    }

    function renderBars(clusters) {
        els.bars.innerHTML = '';
        (clusters || []).slice(0, 6).forEach((c) => {
            const row = el('div', 'swarm-bar');
            const label = el('span', 'swarm-bar__label', c.label || c.key || '—');
            label.title = c.label || c.key || '';
            const track = el('span', 'swarm-bar__track');
            const fill = el('span', 'swarm-bar__fill');
            fill.style.width = `${Math.max(2, Math.round((c.support || 0) * 100))}%`;
            fill.style.background = colorFor(c.key);
            const quorum = el('span', 'swarm-bar__quorum');
            quorum.style.left = `${Math.round(state.quorum * 100)}%`;
            track.append(fill, quorum);
            const num = el('span', 'swarm-bar__num', `${Math.round((c.support || 0) * 100)}% · ${c.count}`);
            row.append(label, track, num);
            els.bars.append(row);
        });
    }

    function renderBody(stats) {
        if (!stats) return;
        const parts = [
            `${stats.entries} results`,
            `${stats.verified} verified`,
            `${stats.corroborated} corroborated`,
        ];
        if (stats.disputed) parts.push(`${stats.disputed} disputed`);
        if (stats.suspect) parts.push(`${stats.suspect} suspect`);
        if (stats.refuted) parts.push(`${stats.refuted} refuted`);
        parts.push(`${stats.conflicts_settled} conflicts settled`);
        if (stats.conflicts_open) parts.push(`${stats.conflicts_open} open`);
        parts.push(`${stats.reuses} reused`);
        els.body.textContent = parts.join(' · ');
        els.bodyTitle.textContent = stats.shared === false ? 'Body of work (private)' : 'Body of work (shared)';
    }

    function handle(evt) {
        if (!root || !evt || typeof evt !== 'object') return;
        switch (evt.type) {
            case 'init':
                state.members = evt.members || 0;
                state.rounds = evt.rounds || 0;
                state.quorum = typeof evt.quorum === 'number' ? evt.quorum : 0.5;
                state.vote = evt.vote || 'weighted';
                state.ledger = evt.ledger || 'shared';
                state.domain = evt.domain || '';
                buildGrid(state.members);
                els.round.textContent = `round 1/${state.rounds}`;
                els.meta.textContent = `${state.members} QNNs · ${state.domain} · ${state.vote} vote · quorum ${Math.round(state.quorum * 100)}%`;
                break;
            case 'member': {
                const i = evt.member;
                const st = state.status[i] || {};
                st.status = evt.status;
                if (evt.round) state.round = Math.max(state.round, evt.round);
                if (evt.answer !== undefined) st.answer = evt.answer;
                if (evt.verify !== undefined && evt.verify !== null) st.verify = evt.verify;
                state.status[i] = st;
                if (evt.status === 'done' && evt.canonical !== undefined) {
                    // Provisional colour until the round's vote assigns clusters.
                    state.assignments[String(i)] = evt.canonical || '';
                }
                if (state.round) els.round.textContent = `round ${state.round}/${state.rounds}`;
                paintMember(i);
                break;
            }
            case 'ledger':
                renderBody(evt.stats);
                break;
            case 'vote':
                state.round = evt.round || state.round;
                state.converged = !!evt.converged;
                state.assignments = evt.assignments || {};
                (evt.clusters || []).forEach((c) => colorFor(c.key));
                (evt.weights || []).forEach((w, i) => {
                    const st = state.status[i] || {};
                    st.weight = w;
                    state.status[i] = st;
                });
                for (let i = 0; i < state.members; i++) paintMember(i);
                renderBars(evt.clusters);
                els.round.textContent = `round ${state.round}/${state.rounds}`;
                els.status.textContent = evt.converged
                    ? 'Converged: the leading answer reached the quorum.'
                    : (evt.final ? 'Rounds used up: the leading answer did not reach the quorum.' : 'No quorum yet: dissenting QNNs evolve and try again.');
                break;
            case 'done':
                state.done = true;
                root.classList.remove('is-running');
                for (let i = 0; i < state.members; i++) {
                    const st = state.status[i] || {};
                    if (st.status !== 'done') st.status = 'done';
                    state.status[i] = st;
                    paintMember(i);
                }
                renderBody(evt.ledger);
                els.round.textContent = `${evt.rounds} round${evt.rounds === 1 ? '' : 's'} · ${evt.calls} calls`;
                els.status.textContent = `${evt.converged ? 'Converged' : 'Not converged'} · answer ${evt.final_answer || '(none)'} · support ${Math.round((evt.support || 0) * 100)}%`;
                break;
            default:
                break;
        }
    }

    function stop() {
        if (root) root.classList.remove('is-running');
    }

    const api = { mount, start, handle, stop };
    return api;
})();
