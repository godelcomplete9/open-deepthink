/**
 * VortexAnimation — panel whirlpool that runs until stop().
 * Tuned for long graph runs (light per-frame cost, no auto-timeout).
 */
const VortexAnimation = (() => {
    'use strict';

    const PARTICLE_COUNT = 260;
    const NOISE_DOTS = 90;
    const FADE_MS = 600;
    // A compact wheel: its radius follows the panel but never exceeds MAX_R,
    // and the label sits just below it.
    const MAX_R = 54;
    const R_FRACTION = 0.3;
    const LABEL_GAP = 16;

    let rootEl = null;
    let canvasEl = null;
    let labelEl = null;
    let mountEl = null;
    let animationFrameId = null;
    let resizeHandler = null;
    let resizeObserver = null;
    let hideTimer = null;
    let playing = false;
    let startTime = 0;
    let particles = [];
    let generation = 0; // invalidate stale rAF chains

    function resolveMount(options) {
        if (options && options.container instanceof HTMLElement) return options.container;
        if (options && typeof options.container === 'string') {
            const el = document.getElementById(String(options.container).replace(/^#/, ''));
            if (el) return el;
        }
        return (
            document.getElementById('qdad-vortex-panel')
            || document.getElementById('brainstorm-vortex-panel')
            || null
        );
    }

    function idleSelector() {
        return '.mode-vortex-idle';
    }

    function ensureDOM(mount) {
        if (!mount) return false;

        if (rootEl && mountEl === mount && mount.contains(rootEl)) {
            return true;
        }

        if (rootEl && rootEl.parentNode) {
            rootEl.parentNode.removeChild(rootEl);
        }

        mountEl = mount;
        rootEl = document.createElement('div');
        rootEl.className = 'vortex-panel';
        rootEl.setAttribute('aria-hidden', 'true');

        canvasEl = document.createElement('canvas');
        canvasEl.className = 'vortex-panel__canvas';

        const grain = document.createElement('div');
        grain.className = 'vortex-panel__grain';

        const scanlines = document.createElement('div');
        scanlines.className = 'vortex-panel__scanlines';

        labelEl = document.createElement('div');
        labelEl.className = 'vortex-panel__label';
        labelEl.innerHTML = '<span class="vortex-panel__label-text">Running</span>';

        rootEl.appendChild(canvasEl);
        rootEl.appendChild(grain);
        rootEl.appendChild(scanlines);
        rootEl.appendChild(labelEl);

        const idle = mount.querySelector(idleSelector());
        if (idle) idle.classList.add('is-hidden');

        mount.appendChild(rootEl);
        return true;
    }

    function initParticles() {
        particles = [];
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            const isRed = Math.random() > 0.5;
            particles.push({
                // ~half the particles sit on the rim, the rest fill the disc
                frac: Math.min(1, 0.07 + Math.random() * 1.9),
                angle: Math.random() * Math.PI * 2,
                baseSpeed: Math.random() * 0.006 + 0.002,
                size: Math.random() * 0.9 + 0.45,
                color: isRed ? [220, 38, 38] : [59, 130, 246],
            });
        }
    }

    function geometry(width, height) {
        const maxR = Math.min(MAX_R, Math.min(width, height) * R_FRACTION);
        return { maxR, cx: width / 2, cy: height / 2 - LABEL_GAP };
    }

    function resizeCanvas(ctx) {
        if (!rootEl || !canvasEl) return;
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(1, rootEl.clientWidth || mountEl?.clientWidth || 320);
        const h = Math.max(1, rootEl.clientHeight || mountEl?.clientHeight || 320);
        canvasEl.width = Math.floor(w * dpr);
        canvasEl.height = Math.floor(h * dpr);
        canvasEl.style.width = w + 'px';
        canvasEl.style.height = h + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const g = geometry(w, h);
        rootEl.style.setProperty('--vortex-label-top', `${Math.round(g.cy + g.maxR + LABEL_GAP)}px`);
    }

    function renderFrame(ctx, gen) {
        if (!playing || gen !== generation) return;

        const elapsed = (performance.now() - startTime) / 1000;
        // Continuous spin + gentle pulse (never ends on its own)
        const pulse = 0.5 + 0.5 * Math.sin(elapsed * 0.7);
        const speedMultiplier = 1.6 + pulse * 2.2;

        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const width = canvasEl.width / dpr;
        const height = canvasEl.height / dpr;

        const flicker = 1 - Math.random() * 0.08 * pulse;
        ctx.fillStyle = `rgb(${Math.round(4 * flicker)}, ${Math.round(4 * flicker)}, ${Math.round(4 * flicker)})`;
        ctx.fillRect(0, 0, width, height);

        const { maxR, cx: centerX, cy: centerY } = geometry(width, height);

        // Sparse film grain
        ctx.fillStyle = `rgba(255, 255, 255, ${0.08 + 0.12 * pulse})`;
        for (let i = 0; i < NOISE_DOTS; i++) {
            ctx.fillRect(Math.random() * width, Math.random() * height, Math.random() * 1.4, Math.random() * 1.4);
        }

        const radiusScale = 0.82 + 0.18 * pulse;

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            p.angle += p.baseSpeed * speedMultiplier;
            const r0 = p.frac * maxR * radiusScale;
            const x = centerX + Math.cos(p.angle) * r0;
            const y = centerY + Math.sin(p.angle) * r0;

            const t = 0.35 * pulse;
            const r = Math.round(p.color[0] * (1 - t) + 255 * t);
            const g = Math.round(p.color[1] * (1 - t) + 255 * t);
            const b = Math.round(p.color[2] * (1 - t) + 255 * t);
            const alpha = 0.35 + Math.random() * 0.45;

            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
            ctx.beginPath();
            ctx.arc(x, y, p.size * (1 + 0.4 * pulse), 0, Math.PI * 2);
            ctx.fill();
        }

        const bloomR = maxR * (0.4 + 0.4 * pulse);
        const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, bloomR);
        gradient.addColorStop(0, `rgba(255, 255, 255, ${0.18 + 0.28 * pulse})`);
        gradient.addColorStop(0.55, `rgba(255, 255, 255, ${0.06 * pulse})`);
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);

        const vignette = ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, Math.max(centerX, centerY) * 1.45
        );
        vignette.addColorStop(0, 'transparent');
        vignette.addColorStop(0.85, 'rgba(0, 0, 0, 0.25)');
        vignette.addColorStop(1, 'rgba(0, 0, 0, 0.65)');
        ctx.fillStyle = vignette;
        ctx.fillRect(0, 0, width, height);

        animationFrameId = requestAnimationFrame(() => renderFrame(ctx, gen));
    }

    function cleanupListeners() {
        if (animationFrameId != null) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        if (resizeHandler) {
            window.removeEventListener('resize', resizeHandler);
            resizeHandler = null;
        }
        if (resizeObserver) {
            resizeObserver.disconnect();
            resizeObserver = null;
        }
        if (hideTimer) {
            clearTimeout(hideTimer);
            hideTimer = null;
        }
    }

    function showIdle() {
        if (!mountEl) return;
        const idle = mountEl.querySelector(idleSelector());
        if (idle) idle.classList.remove('is-hidden');
        if (rootEl) {
            rootEl.classList.remove('vortex-panel--active', 'vortex-panel--fading');
            rootEl.setAttribute('aria-hidden', 'true');
        }
    }

    function stop(opts) {
        const fade = !opts || opts.fade !== false;
        generation += 1; // kill any in-flight frames
        if (!playing && !rootEl) return;

        playing = false;
        cleanupListeners();

        if (rootEl && fade) {
            rootEl.classList.add('vortex-panel--fading');
            hideTimer = setTimeout(() => {
                showIdle();
                hideTimer = null;
            }, FADE_MS);
        } else {
            showIdle();
        }
    }

    function play(opts) {
        const options = opts || {};
        const mount = resolveMount(options);
        if (!mount) {
            console.warn('VortexAnimation: no mount container found');
            return;
        }

        generation += 1;
        const gen = generation;
        playing = false;
        cleanupListeners();
        if (rootEl) rootEl.classList.remove('vortex-panel--fading');

        if (!ensureDOM(mount)) return;

        if (labelEl) {
            labelEl.querySelector('.vortex-panel__label-text').textContent =
                options.label || 'Running';
        }

        const idle = mount.querySelector(idleSelector());
        if (idle) idle.classList.add('is-hidden');

        rootEl.classList.remove('vortex-panel--fading');
        rootEl.classList.add('vortex-panel--active');
        rootEl.setAttribute('aria-hidden', 'false');

        const ctx = canvasEl.getContext('2d', { alpha: false });
        if (!ctx) return;

        // Defer first paint so layout has size (panel may just become visible)
        const startLoop = () => {
            if (gen !== generation) return;
            resizeCanvas(ctx);
            initParticles();
            startTime = performance.now();
            playing = true;
            animationFrameId = requestAnimationFrame(() => renderFrame(ctx, gen));
        };

        resizeHandler = () => {
            if (playing) resizeCanvas(ctx);
        };
        window.addEventListener('resize', resizeHandler);

        if (typeof ResizeObserver !== 'undefined') {
            resizeObserver = new ResizeObserver(() => {
                if (playing) resizeCanvas(ctx);
            });
            resizeObserver.observe(mount);
        }

        // Double-rAF ensures layout after display:block / mode switch
        requestAnimationFrame(() => requestAnimationFrame(startLoop));
    }

    function isPlaying() {
        return playing;
    }

    function setLabel(text) {
        if (labelEl) {
            const el = labelEl.querySelector('.vortex-panel__label-text');
            if (el) el.textContent = text || 'Running';
        }
    }

    return { play, stop, isPlaying, setLabel };
})();

if (typeof window !== 'undefined') {
    window.VortexAnimation = VortexAnimation;
}
