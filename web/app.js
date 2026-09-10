/**
 * say that sound — Voice Pronunciation Coach Web Client
 *
 * Integrated UI with:
 *   - Ambient interactive WaveField canvas (matching pronounce-live aesthetics)
 *   - LiveKit Cloud WebRTC transport with echo cancellation and noise suppression
 *   - Real-time Rime Mist v3 corrective speech playback
 *   - Real-time 2x2 metrics card (Target Word, Score, Weak Phoneme, Speed)
 *   - Dynamic 5-attempt session history strip (✓ / ✗)
 *   - Live microphone frequency waveform visualizer
 */

import {
    Room,
    RoomEvent,
    Track,
    ConnectionState,
} from 'https://cdn.jsdelivr.net/npm/livekit-client@2/dist/livekit-client.esm.mjs';

// =============================================================================
// 1. Ambient Interactive WaveField Canvas
// =============================================================================

function initWaveField() {
    const canvas = document.getElementById('hero-wave-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const COLORS = ['#1F4E3D', '#C85A32', '#E2A85C'];
    const pointer = { x: 0.5, y: 0.5, energy: 0.35 };

    let t = 0;

    function resize() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const rect = canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    resize();
    window.addEventListener('resize', resize);

    window.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        pointer.x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        pointer.y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
        pointer.energy = 1.0;
    });

    function draw() {
        const rect = canvas.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;

        if (w > 0 && h > 0) {
            ctx.clearRect(0, 0, w, h);
            t += 0.006;
            pointer.energy += (0.4 - pointer.energy) * 0.02;

            const mid = h / 2;
            const lines = 9;

            for (let i = 0; i < lines; i++) {
                const p = i / (lines - 1);
                const color = COLORS[i % COLORS.length];

                ctx.beginPath();
                ctx.lineWidth = 1.4;
                ctx.strokeStyle = color;
                ctx.globalAlpha = 0.18 + 0.35 * Math.pow(1 - Math.abs(p - 0.5) * 2, 1.2);

                for (let x = 0; x <= w; x += 6) {
                    const nx = x / w;
                    const dist = Math.abs(nx - pointer.x);
                    const focus = Math.exp(-(dist * dist) / 0.02);
                    const amp = h * 0.22 * (0.35 + focus * 1.5 * pointer.energy) * Math.sin(Math.PI * nx);
                    const y = mid + (p - 0.5) * h * 0.42 +
                              Math.sin(nx * 9 + t * 2.2 + i * 0.55) * amp * 0.5 +
                              Math.sin(nx * 21 - t * 3.1 + i) * amp * 0.22;

                    if (x === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
            }
            ctx.globalAlpha = 1;
        }
        requestAnimationFrame(draw);
    }

    draw();
}

// =============================================================================
// 2. DOM References & State
// =============================================================================

const connectBtn = document.getElementById('connect-btn');
const connectBtnText = document.getElementById('connect-btn-text');
const muteBtn = document.getElementById('mute-btn');

const connectionDot = document.getElementById('connection-dot');
const connectionText = document.getElementById('connection-text');
const drillStateBadge = document.getElementById('drill-state-badge');

const targetWordEl = document.getElementById('target-word');
const targetPhonemesEl = document.getElementById('target-phonemes');
const metricTargetWordEl = document.getElementById('metric-target-word');

const confidenceEl = document.getElementById('confidence');
const scoreStatusEl = document.getElementById('score-status');
const weakPhonemeEl = document.getElementById('weak-phoneme');
const speedTierEl = document.getElementById('speed-tier');
const speedLabelEl = document.getElementById('speed-label');

const micMeterCanvas = document.getElementById('mic-meter');
const micCtx = micMeterCanvas ? micMeterCanvas.getContext('2d') : null;
const micStatusLabel = document.getElementById('mic-status-label');
const historyStrip = document.getElementById('history-strip');
const transcriptLog = document.getElementById('transcript-log');
const vocabPills = document.querySelectorAll('.vocab-pill');

// State
let room = null;
let audioContext = null;
let analyser = null;
let meterAnimId = null;
let isMicMuted = false;
let lastCoachMsg = '';

const PHONEME_MAP = {
    three: '/θɹi/',
    think: '/θɪŋk/',
    this: '/ðɪs/',
    sheep: '/ʃip/',
    ship: '/ʃɪp/',
    rice: '/ɹaɪs/',
    light: '/laɪt/',
    right: '/ɹaɪt/',
};

// =============================================================================
// 3. LiveKit Connection Handling
// =============================================================================

async function toggleConnection() {
    if (room && room.state === ConnectionState.Connected) {
        await disconnect();
    } else {
        await connect();
    }
}

async function connect() {
    setStatus('connecting', 'Connecting…');
    connectBtn.disabled = true;
    connectBtnText.textContent = 'Connecting…';

    try {
        const identity = 'web-user-' + Math.random().toString(36).substring(2, 8);
        const res = await fetch(`/token?room=say-that-sound&identity=${identity}`);
        if (!res.ok) {
            const errText = await res.text();
            throw new Error(`Token request failed: ${errText}`);
        }
        const data = await res.json();
        const { token, url } = data;

        if (!url || !token) {
            throw new Error('Missing token or url in server response');
        }

        appendTranscript('system', `Connecting to voice session (${identity})…`);

        room = new Room({
            adaptiveStream: true,
            dynacast: true,
        });

        // Connected
        room.on(RoomEvent.Connected, () => {
            setStatus('connected', 'Connected');
            updateDrillBadge('LISTENING');
            connectBtn.disabled = false;
            connectBtn.classList.add('active-disconnect');
            connectBtnText.textContent = 'Disconnect';
            muteBtn.disabled = false;
            appendTranscript('system', 'Connected to room. Speak a target word clearly into your microphone.');
        });

        // Disconnected
        room.on(RoomEvent.Disconnected, () => {
            setStatus('disconnected', 'Disconnected');
            updateDrillBadge('IDLE');
            cleanup();
        });

        // Subscribed to Agent Audio Track (Rime TTS playback)
        room.on(RoomEvent.TrackSubscribed, (track) => {
            if (track.kind === Track.Kind.Audio) {
                const el = track.attach();
                el.id = 'agent-audio-' + Date.now();
                el.style.display = 'none';
                document.body.appendChild(el);
                appendTranscript('system', 'Agent voice stream active (Rime Mist v3)');
            }
        });

        room.on(RoomEvent.TrackUnsubscribed, (track) => {
            if (track.kind === Track.Kind.Audio) {
                const els = track.detach();
                els.forEach((el) => el.remove());
            }
        });

        // Transcription feed
        room.on(RoomEvent.TranscriptionReceived, (segments, participant) => {
            if (!segments || segments.length === 0) return;
            for (const seg of segments) {
                const isAgent = participant && !participant.isLocal;
                if (seg.final && seg.text) {
                    const clean = seg.text.trim();
                    if (isAgent) {
                        if (clean && clean !== lastCoachMsg) {
                            lastCoachMsg = clean;
                            appendTranscript('agent', clean);
                        }
                    } else {
                        appendTranscript('user', clean);
                    }
                }
            }
        });

        // Realtime Data Channel for drill state updates
        room.on(RoomEvent.DataReceived, (payload) => {
            try {
                const str = new TextDecoder().decode(payload);
                const data = JSON.parse(str);
                if (data.type === 'drill_state') {
                    updateDrillMetrics(data);
                    if (data.coach_text) {
                        const clean = data.coach_text.trim();
                        if (clean && clean !== lastCoachMsg) {
                            lastCoachMsg = clean;
                            appendTranscript('agent', clean);
                        }
                    }
                }
            } catch (err) {
                // Ignore non-JSON
            }
        });

        // Connect room
        await room.connect(url, token);

        // Publish mic with AEC + NS
        await room.localParticipant.setMicrophoneEnabled(true, {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
        });

        // Setup live audio visualizer
        setupMicVisualizer();

    } catch (err) {
        console.error('Connection failed:', err);
        setStatus('disconnected', 'Disconnected');
        appendTranscript('system', `Error: ${err.message}`);
        cleanup();
    }
}

async function disconnect() {
    if (room) {
        await room.disconnect();
    }
    cleanup();
}

function cleanup() {
    if (meterAnimId) {
        cancelAnimationFrame(meterAnimId);
        meterAnimId = null;
    }
    if (audioContext) {
        audioContext.close().catch(() => {});
        audioContext = null;
    }
    analyser = null;
    room = null;

    // Detach audio elements
    document.querySelectorAll('[id^="agent-audio"]').forEach((el) => el.remove());

    connectBtn.disabled = false;
    connectBtn.classList.remove('active-disconnect');
    connectBtnText.textContent = 'Start Practicing';
    muteBtn.disabled = true;

    if (micCtx && micMeterCanvas) {
        micCtx.clearRect(0, 0, micMeterCanvas.width, micMeterCanvas.height);
    }
}

connectBtn.addEventListener('click', toggleConnection);

// Mute / Unmute
muteBtn.addEventListener('click', async () => {
    if (!room || !room.localParticipant) return;
    isMicMuted = !isMicMuted;
    await room.localParticipant.setMicrophoneEnabled(!isMicMuted);
    muteBtn.textContent = isMicMuted ? 'Unmute Mic' : 'Mute Mic';
    if (micStatusLabel) {
        micStatusLabel.textContent = isMicMuted ? 'Microphone Muted' : 'AEC + Noise Suppression Active';
    }
});

// =============================================================================
// 4. Status & Metrics Updating
// =============================================================================

function setStatus(status, text) {
    if (connectionDot) connectionDot.className = `dot ${status}`;
    if (connectionText) connectionText.textContent = text;
}

function updateDrillBadge(state) {
    if (!drillStateBadge) return;
    const s = (state || 'IDLE').toUpperCase();
    drillStateBadge.textContent = s;
    drillStateBadge.className = `badge badge-${s.toLowerCase()} font-mono-label`;
}

function updateDrillMetrics(meta) {
    if (!meta) return;

    // 1. Target word & phoneme notation
    if (meta.word) {
        const w = meta.word.toLowerCase();
        targetWordEl.textContent = w;
        if (metricTargetWordEl) metricTargetWordEl.textContent = w;
        targetPhonemesEl.textContent = PHONEME_MAP[w] || '';

        // Update pill active state
        vocabPills.forEach((p) => {
            if (p.getAttribute('data-word') === w) p.classList.add('active');
            else p.classList.remove('active');
        });
    }

    // 2. Pronunciation score / confidence
    if (meta.confidence !== undefined && meta.confidence !== null) {
        const pct = Math.round(meta.confidence * 100);
        confidenceEl.textContent = `${pct}%`;
        confidenceEl.className = 'metric-main-val font-display';

        if (pct >= 75) {
            confidenceEl.classList.add('conf-high');
            scoreStatusEl.textContent = 'PASSED (>= 75%)';
            scoreStatusEl.style.color = '#15803d';
        } else if (pct >= 50) {
            confidenceEl.classList.add('conf-mid');
            scoreStatusEl.textContent = 'NEEDS DRILL (50-74%)';
            scoreStatusEl.style.color = '#b45309';
        } else {
            confidenceEl.classList.add('conf-low');
            scoreStatusEl.textContent = 'LOW ACCURACY (< 50%)';
            scoreStatusEl.style.color = '#b91c1c';
        }
    }

    // 3. Weak phoneme substitution detail
    const detail = meta.phoneme_detail || meta.weak_phoneme || meta.phoneme;
    if (detail) {
        weakPhonemeEl.textContent = detail;
    } else {
        weakPhonemeEl.textContent = 'None (Clear!)';
    }

    // 4. Speed tier
    if (meta.speed_alpha !== undefined && meta.speed_alpha !== null) {
        speedTierEl.textContent = `${meta.speed_alpha.toFixed(2)}x`;
        speedLabelEl.textContent = meta.speed_alpha < 0.7 ? 'Slowed Corrective Model' : 'Natural Praise Speed';
    } else if (meta.speed_tier) {
        speedTierEl.textContent = meta.speed_tier === 'slow' ? '0.50x' : '1.00x';
        speedLabelEl.textContent = meta.speed_tier === 'slow' ? 'Slowed Corrective Model' : 'Natural Praise Speed';
    }

    // 5. State badge
    if (meta.drill_state) {
        updateDrillBadge(meta.drill_state);
    }

    // 6. Session history strip (last 5 attempts)
    if (historyStrip && Array.isArray(meta.history)) {
        if (meta.history.length === 0) {
            historyStrip.innerHTML = '<span class="history-empty font-mono-label">No attempts yet. Click Start Practicing and speak your first word!</span>';
        } else {
            historyStrip.innerHTML = '';
            meta.history.slice(-5).forEach((item) => {
                const pill = document.createElement('div');
                const isPass = item.passed;
                pill.className = `history-pill ${isPass ? 'pass' : 'fail'}`;
                const icon = isPass ? '✓' : '✕';
                const pct = item.confidence !== undefined ? `${Math.round(item.confidence * 100)}%` : '';
                pill.innerHTML = `
                    <span class="history-word">${item.word}</span>
                    <span class="history-icon">${icon}</span>
                    <span class="history-conf">${pct}</span>
                `;
                historyStrip.appendChild(pill);
            });
        }
    }
}

// =============================================================================
// 5. Live Microphone Frequency Visualizer
// =============================================================================

function setupMicVisualizer() {
    if (!room || !room.localParticipant) return;

    try {
        const audioTracks = Array.from(room.localParticipant.audioTrackPublications.values());
        if (audioTracks.length === 0 || !audioTracks[0].track) return;

        const mediaStreamTrack = audioTracks[0].track.mediaStreamTrack;
        const stream = new MediaStream([mediaStreamTrack]);

        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 64;

        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        function renderVisualizer() {
            meterAnimId = requestAnimationFrame(renderVisualizer);

            if (!micCtx || !micMeterCanvas) return;
            analyser.getByteFrequencyData(dataArray);

            const w = micMeterCanvas.width;
            const h = micMeterCanvas.height;

            micCtx.clearRect(0, 0, w, h);

            const barWidth = (w / bufferLength) * 1.5;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const barHeight = (dataArray[i] / 255) * h * 0.9;
                const ratio = i / bufferLength;

                // Gradient from rich moss green to terracotta clay
                micCtx.fillStyle = `rgb(${31 + ratio * 170}, ${78 + ratio * 20}, ${61 - ratio * 10})`;
                micCtx.fillRect(x, h - barHeight, barWidth - 2, barHeight);
                x += barWidth;
            }
        }

        renderVisualizer();

    } catch (err) {
        console.warn('Mic visualizer setup failed:', err);
    }
}

// =============================================================================
// 6. Transcript Logging
// =============================================================================

function appendTranscript(type, text) {
    if (!transcriptLog) return;
    const msg = document.createElement('div');
    msg.className = `transcript-msg ${type} font-mono-label`;

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const tag = type === 'agent' ? '[Coach • Rime]' : type === 'user' ? '[You]' : '[System]';

    msg.textContent = `${time} ${tag} ${text}`;
    transcriptLog.appendChild(msg);
    transcriptLog.scrollTop = transcriptLog.scrollHeight;
}

// =============================================================================
// 7. Vocabulary Selector Pills
// =============================================================================

vocabPills.forEach((btn) => {
    btn.addEventListener('click', () => {
        vocabPills.forEach((p) => p.classList.remove('active'));
        btn.classList.add('active');

        const word = btn.getAttribute('data-word');
        targetWordEl.textContent = word;
        if (metricTargetWordEl) metricTargetWordEl.textContent = word;
        targetPhonemesEl.textContent = PHONEME_MAP[word] || '';
        appendTranscript('system', `Target word set to "${word}". Speak it when ready.`);
    });
});

// =============================================================================
// 8. Auto-connect if ?coach=true or #coach
// =============================================================================

window.addEventListener('DOMContentLoaded', () => {
    initWaveField();

    if (window.location.search.includes('coach=') || window.location.hash === '#coach') {
        setTimeout(connect, 400);
    }
});
