/**
 * say that sound — Interactive Landing Page & Live Pronunciation Coach Client
 *
 * Integrates:
 *   1. Interactive multi-harmonic WaveField canvas animation (from WaveField.jsx)
 *   2. LiveKit Cloud WebRTC transport for continuous dual-stream audio
 *   3. Real-time Rime Mist v3 corrective TTS audio subscription & playback
 *   4. Real-time 2x2 metrics card (Word, Weak Phoneme, Confidence Score, Speed)
 *   5. Session history strip (Last 5 attempts with ✓ / ✗)
 *   6. Real-time microphone waveform visualizer
 */

import {
    Room,
    RoomEvent,
    Track,
    ConnectionState,
} from 'https://cdn.jsdelivr.net/npm/livekit-client@2/dist/livekit-client.esm.mjs';

// =============================================================================
// 1. Interactive WaveField Canvas Animation (Faithful to WaveField.jsx)
// =============================================================================

function initWaveField() {
    const canvas = document.getElementById('hero-wave-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const COLORS = ['#1F4E3D', '#C85A32', '#E2A85C'];
    const pointer = { x: 0.5, y: 0.5, energy: 0.35 };

    let rafId;
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
                ctx.globalAlpha = 0.22 + 0.4 * Math.pow(1 - Math.abs(p - 0.5) * 2, 1.2);

                for (let x = 0; x <= w; x += 5) {
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
        rafId = requestAnimationFrame(draw);
    }

    draw();
}

// =============================================================================
// 2. DOM Elements & State
// =============================================================================

const coachModal = document.getElementById('coach-modal');
const closeModalBtn = document.getElementById('close-modal-btn');
const connectBtn = document.getElementById('connect-btn');
const muteBtn = document.getElementById('mute-btn');
const disconnectBtn = document.getElementById('disconnect-btn');

const connectionDot = document.getElementById('connection-dot');
const connectionText = document.getElementById('connection-text');
const drillStateBadge = document.getElementById('drill-state-badge');

const targetWordEl = document.getElementById('target-word');
const targetPhonemesEl = document.getElementById('target-phonemes');
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

// CTA Buttons
const navStartBtn = document.getElementById('nav-start-btn');
const heroStartBtn = document.getElementById('hero-start-btn');
const closingStartBtn = document.getElementById('closing-start-btn');

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
// 3. Modal Opening / Closing
// =============================================================================

function openCoachStudio() {
    coachModal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    // Auto-connect if not currently connected
    if (!room || room.state !== ConnectionState.Connected) {
        connectToLiveKit();
    }
}

function closeCoachStudio() {
    coachModal.classList.add('hidden');
    document.body.style.overflow = '';
}

// CTA listeners
if (navStartBtn) navStartBtn.addEventListener('click', openCoachStudio);
if (heroStartBtn) heroStartBtn.addEventListener('click', openCoachStudio);
if (closingStartBtn) closingStartBtn.addEventListener('click', openCoachStudio);
if (closeModalBtn) closeModalBtn.addEventListener('click', closeCoachStudio);

// Close on backdrop click (outside card)
if (coachModal) {
    coachModal.addEventListener('click', (e) => {
        if (e.target === coachModal) {
            closeCoachStudio();
        }
    });
}

// Close on Escape key
window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !coachModal.classList.contains('hidden')) {
        closeCoachStudio();
    }
});

// Auto-open if query or hash has coach
if (window.location.search.includes('coach=') || window.location.hash === '#coach') {
    setTimeout(openCoachStudio, 400);
}

// =============================================================================
// 4. LiveKit WebRTC Engine
// =============================================================================

async function connectToLiveKit() {
    if (room && room.state === ConnectionState.Connected) return;

    setStatus('connecting', 'Requesting LiveKit access token…');
    connectBtn.disabled = true;
    connectBtn.textContent = 'Connecting…';

    try {
        const identity = 'web-learner-' + Math.random().toString(36).substring(2, 8);
        const res = await fetch(`/token?room=say-that-sound&identity=${identity}`);
        if (!res.ok) {
            const errText = await res.text();
            throw new Error(`Token endpoint failed: ${errText}`);
        }
        const data = await res.json();
        const { token, url } = data;

        if (!url || !token) {
            throw new Error('Invalid token response from server');
        }

        setStatus('connecting', 'Connecting to WebRTC voice room…');
        appendTranscript('system', `Access token granted for identity [${identity}]. Joining LiveKit room…`);

        room = new Room({
            adaptiveStream: true,
            dynacast: true,
        });

        // Connected
        room.on(RoomEvent.Connected, () => {
            setStatus('connected', 'Live dual-stream connected');
            updateDrillBadge('LISTENING');
            connectBtn.style.display = 'none';
            muteBtn.disabled = false;
            disconnectBtn.disabled = false;
            appendTranscript('system', 'Room connected! Speak clearly into your mic to begin practice.');
        });

        // Disconnected
        room.on(RoomEvent.Disconnected, () => {
            setStatus('disconnected', 'Disconnected');
            updateDrillBadge('IDLE');
            cleanupSession();
        });

        // Agent Audio Track Subscribed (Rime TTS playback)
        room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
            if (track.kind === Track.Kind.Audio) {
                const el = track.attach();
                el.id = 'agent-audio-' + Date.now();
                el.style.display = 'none';
                document.body.appendChild(el);
                appendTranscript('system', 'Agent audio stream active (Rime Mist v3)');
            }
        });

        room.on(RoomEvent.TrackUnsubscribed, (track) => {
            if (track.kind === Track.Kind.Audio) {
                const els = track.detach();
                els.forEach((el) => el.remove());
            }
        });

        // Live transcription stream
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

        // Realtime Drill State via Data Channel
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
                // Not JSON data packet
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
        console.error('LiveKit connection failed:', err);
        setStatus('disconnected', `Error: ${err.message}`);
        appendTranscript('system', `Connection failed: ${err.message}`);
        cleanupSession();
    }
}

async function disconnectFromLiveKit() {
    if (room) {
        await room.disconnect();
    }
    cleanupSession();
}

function cleanupSession() {
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

    // Remove any attached agent audio elements
    document.querySelectorAll('[id^="agent-audio"]').forEach((el) => el.remove());

    connectBtn.style.display = '';
    connectBtn.disabled = false;
    connectBtn.textContent = 'Connect & Start Speaking';
    muteBtn.disabled = true;
    disconnectBtn.disabled = true;

    if (micCtx && micMeterCanvas) {
        micCtx.clearRect(0, 0, micMeterCanvas.width, micMeterCanvas.height);
    }
}

// Connect / Disconnect Buttons
connectBtn.addEventListener('click', connectToLiveKit);
disconnectBtn.addEventListener('click', disconnectFromLiveKit);

// Mute / Unmute
muteBtn.addEventListener('click', async () => {
    if (!room || !room.localParticipant) return;
    isMicMuted = !isMicMuted;
    await room.localParticipant.setMicrophoneEnabled(!isMicMuted);
    muteBtn.textContent = isMicMuted ? 'Unmute Mic' : 'Mute Mic';
    if (micStatusLabel) {
        micStatusLabel.textContent = isMicMuted ? 'Microphone Muted' : 'AEC + NS Enabled';
    }
});

// =============================================================================
// 5. State & Metrics Updating
// =============================================================================

function setStatus(status, text) {
    if (connectionDot) {
        connectionDot.className = `dot ${status}`;
    }
    if (connectionText) {
        connectionText.textContent = text;
    }
}

function updateDrillBadge(state) {
    if (!drillStateBadge) return;
    const s = (state || 'IDLE').toUpperCase();
    drillStateBadge.textContent = s;
    drillStateBadge.className = `badge badge-${s.toLowerCase()} font-mono-label`;
}

function updateDrillMetrics(meta) {
    if (!meta) return;

    // 1. Target word & phonetic representation
    if (meta.word) {
        targetWordEl.textContent = meta.word;
        const phonemes = PHONEME_MAP[meta.word.toLowerCase()] || '';
        targetPhonemesEl.textContent = phonemes;

        // Highlight corresponding vocab pill
        vocabPills.forEach((p) => {
            if (p.getAttribute('data-word') === meta.word.toLowerCase()) {
                p.classList.add('active');
            } else {
                p.classList.remove('active');
            }
        });
    }

    // 2. Score / Confidence
    if (meta.confidence !== undefined && meta.confidence !== null) {
        const pct = Math.round(meta.confidence * 100);
        confidenceEl.textContent = `${pct}%`;
        confidenceEl.className = 'score-val font-display';

        if (pct >= 75) {
            confidenceEl.classList.add('conf-high');
            scoreStatusEl.textContent = 'PASSED';
            scoreStatusEl.style.color = '#15803d';
        } else if (pct >= 50) {
            confidenceEl.classList.add('conf-mid');
            scoreStatusEl.textContent = 'NEEDS DRILL';
            scoreStatusEl.style.color = '#b45309';
        } else {
            confidenceEl.classList.add('conf-low');
            scoreStatusEl.textContent = 'LOW CONF';
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

    // 4. Speed tier & Rime speed
    if (meta.speed_alpha !== undefined && meta.speed_alpha !== null) {
        speedTierEl.textContent = `${meta.speed_alpha.toFixed(2)}x`;
        speedLabelEl.textContent = meta.speed_alpha < 0.7 ? 'Slow Corrective Model' : 'Natural Speed';
    } else if (meta.speed_tier) {
        speedTierEl.textContent = meta.speed_tier === 'slow' ? '0.50x' : '1.00x';
        speedLabelEl.textContent = meta.speed_tier === 'slow' ? 'Slow Corrective Model' : 'Natural Speed';
    }

    // 5. State badge
    if (meta.drill_state) {
        updateDrillBadge(meta.drill_state);
    }

    // 6. Session history strip (last 5 attempts)
    if (historyStrip && Array.isArray(meta.history)) {
        if (meta.history.length === 0) {
            historyStrip.innerHTML = '<span class="history-empty font-mono-label">No attempts yet. Speak your first word!</span>';
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
// 6. Microphone Waveform Visualizer
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

            const barWidth = (w / bufferLength) * 1.8;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const barHeight = (dataArray[i] / 255) * h * 0.9;

                // Color gradient from moss green to gold
                const ratio = i / bufferLength;
                micCtx.fillStyle = `rgb(${31 + ratio * 180}, ${78 + ratio * 80}, ${61 + ratio * 40})`;

                micCtx.fillRect(x, h - barHeight, barWidth - 2, barHeight);
                x += barWidth;
            }
        }

        renderVisualizer();

    } catch (err) {
        console.warn('Microphone visualizer unavailable:', err);
    }
}

// =============================================================================
// 7. Transcript Logging
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
// 8. Practice Words Pill Interactivity
// =============================================================================

vocabPills.forEach((btn) => {
    btn.addEventListener('click', () => {
        vocabPills.forEach((p) => p.classList.remove('active'));
        btn.classList.add('active');

        const word = btn.getAttribute('data-word');
        targetWordEl.textContent = word;
        targetPhonemesEl.textContent = PHONEME_MAP[word] || '';
        appendTranscript('system', `Practicing target word: "${word}"`);
    });
});

// =============================================================================
// 9. Initialize Page
// =============================================================================

window.addEventListener('DOMContentLoaded', () => {
    initWaveField();
});
