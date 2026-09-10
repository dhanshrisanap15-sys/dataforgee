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
// 2. DOM References & Global State
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

// Custom Word Bar elements
const customWordForm = document.getElementById('custom-word-form');
const customWordInput = document.getElementById('custom-word-input');
const customWordSubmit = document.getElementById('custom-word-submit');
const suggChips = document.querySelectorAll('.sugg-chip');

// Phoneme Heatmap & Mouth Guide elements
const phonemeHeatmapEl = document.getElementById('phoneme-heatmap');
const articulationSoundBadge = document.getElementById('articulation-sound-badge');
const artTongueEl = document.getElementById('art-tongue');
const artLipsEl = document.getElementById('art-lips');
const artAirflowEl = document.getElementById('art-airflow');
const artVoicingEl = document.getElementById('art-voicing');
const artTipEl = document.getElementById('art-tip');

// Minimal Pairs Tracks elements
const trackTabs = document.querySelectorAll('.track-tab');
const trackWordsContainer = document.getElementById('vocab-pills');

// State
let room = null;
let audioContext = null;
let analyser = null;
let meterAnimId = null;
let isMicMuted = false;
let lastCoachMsg = '';
let currentTargetWord = 'three';
let currentBreakdown = [];
let activeTrackId = 'all';

// Pre-defined practice tracks (fast client-side lookup)
const TRACKS_DATA = {
    'all': [
        { word: 'three', ipa: '/θɹi/' },
        { word: 'ship', ipa: '/ʃɪp/' },
        { word: 'sheep', ipa: '/ʃip/' },
        { word: 'rice', ipa: '/ɹaɪs/' },
        { word: 'light', ipa: '/laɪt/' },
        { word: 'right', ipa: '/ɹaɪt/' },
        { word: 'think', ipa: '/θɪŋk/' },
        { word: 'this', ipa: '/ðɪs/' },
    ],
    'th-vs-t': [
        { word: 'three', ipa: '/θɹi/' },
        { word: 'think', ipa: '/θɪŋk/' },
        { word: 'that', ipa: '/ðæt/' },
        { word: 'this', ipa: '/ðɪs/' },
        { word: 'tree', ipa: '/tɹi/' },
        { word: 'tank', ipa: '/tæŋk/' },
    ],
    'r-vs-l': [
        { word: 'rice', ipa: '/ɹaɪs/' },
        { word: 'light', ipa: '/laɪt/' },
        { word: 'right', ipa: '/ɹaɪt/' },
        { word: 'lake', ipa: '/leɪk/' },
        { word: 'rake', ipa: '/ɹeɪk/' },
        { word: 'lead', ipa: '/lid/' },
    ],
    'sh-vs-s': [
        { word: 'ship', ipa: '/ʃɪp/' },
        { word: 'sheep', ipa: '/ʃip/' },
        { word: 'sip', ipa: '/sɪp/' },
        { word: 'seat', ipa: '/sit/' },
        { word: 'sea', ipa: '/si/' },
        { word: 'shine', ipa: '/ʃaɪn/' },
    ],
    'v-vs-w': [
        { word: 'voice', ipa: '/vɔɪs/' },
        { word: 'wave', ipa: '/weɪv/' },
        { word: 'vine', ipa: '/vaɪn/' },
        { word: 'wine', ipa: '/waɪn/' },
        { word: 'vest', ipa: '/vɛst/' },
        { word: 'west', ipa: '/wɛst/' },
    ],
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
            notifyAgentOfWord(currentTargetWord);
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

        // Send active track words & initial target word to agent
        notifyAgentOfWord(currentTargetWord, false);

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
// 4. Send Word Selection to LiveKit Agent
// =============================================================================

function notifyAgentOfWord(word, custom = false) {
    if (room && room.state === ConnectionState.Connected && room.localParticipant) {
        try {
            let trackWords = undefined;
            if (!custom && TRACKS_DATA[activeTrackId]) {
                trackWords = TRACKS_DATA[activeTrackId].map((item) => item.word);
            }
            const payload = JSON.stringify({
                type: 'select_word',
                word: word,
                track_words: trackWords,
            });
            room.localParticipant.publishData(new TextEncoder().encode(payload), { reliable: true });
        } catch (err) {
            console.debug('Failed to publish select_word to agent:', err);
        }
    }
}

// =============================================================================
// 5. Target Word Selection & Phoneme Breakdown
// =============================================================================

async function selectTargetWord(word, notify = true, custom = false) {
    if (!word) return;
    const cleanWord = word.trim().toLowerCase().replace(/[^a-z]/g, '');
    if (!cleanWord) return;

    currentTargetWord = cleanWord;
    targetWordEl.textContent = cleanWord;
    if (metricTargetWordEl) metricTargetWordEl.textContent = cleanWord;

    // Highlight pill in active track if present
    if (trackWordsContainer) {
        const pills = trackWordsContainer.querySelectorAll('.vocab-pill');
        pills.forEach((p) => {
            if (p.getAttribute('data-word') === cleanWord) p.classList.add('active');
            else p.classList.remove('active');
        });
    }

    try {
        const res = await fetch(`/api/phonemes?word=${encodeURIComponent(cleanWord)}`);
        if (res.ok) {
            const data = await res.json();
            currentBreakdown = data.breakdown || [];

            if (data.ipa) {
                targetPhonemesEl.textContent = data.ipa;
            }

            // Render interactive phoneme heatmap
            renderPhonemeHeatmap(data.phonemes, null, null);

            // Update articulation guide with initial phoneme
            if (currentBreakdown.length > 0) {
                const first = currentBreakdown[0];
                updateArticulationGuide(first.articulation, `${first.phoneme} (/${first.ipa}/)`);
            }
        }
    } catch (err) {
        console.warn('Phoneme lookup failed:', err);
    }

    if (notify) {
        notifyAgentOfWord(cleanWord, custom);
        appendTranscript('system', `Target word set to "${cleanWord}". Speak it when ready.`);
    }
}

// =============================================================================
// 6. Interactive Syllable & Phoneme Heatmap
// =============================================================================

function renderPhonemeHeatmap(phonemes, alignment, weakPhoneme) {
    if (!phonemeHeatmapEl) return;
    if (!phonemes || phonemes.length === 0) {
        phonemeHeatmapEl.innerHTML = '<span class="font-mono-label text-ink-3">No phoneme sequence available</span>';
        return;
    }

    phonemeHeatmapEl.innerHTML = '';

    // Build alignment lookup if available
    const alignMap = {};
    if (Array.isArray(alignment)) {
        alignment.forEach((item) => {
            if (item.expected) alignMap[item.expected.toUpperCase()] = item;
        });
    }

    phonemes.forEach((ph, idx) => {
        const norm = ph.toUpperCase();
        const pill = document.createElement('div');
        pill.className = 'phoneme-pill';
        pill.setAttribute('data-phoneme', norm);

        // Find breakdown details
        const itemInfo = currentBreakdown.find((b) => b.phoneme.toUpperCase() === norm) || {};
        const ipa = itemInfo.ipa || norm.toLowerCase();

        let statusTag = '';

        if (alignMap[norm]) {
            const al = alignMap[norm];
            const confPct = Math.round((al.confidence || 0) * 100);
            if (al.is_match && confPct >= 70) {
                statusTag = `✓ ${confPct}%`;
                pill.classList.add('pass');
            } else {
                const obs = al.observed || '?';
                statusTag = `${obs} ➔ ${norm}`;
                pill.classList.add('drift');
            }
        } else if (weakPhoneme && weakPhoneme.toUpperCase() === norm) {
            statusTag = 'WEAK';
            pill.classList.add('drift');
        } else {
            statusTag = 'TARGET';
        }

        pill.innerHTML = `
            <span class="pill-token">${norm}</span>
            <span class="pill-ipa">/${ipa}/</span>
            <span class="pill-status-tag">${statusTag}</span>
        `;

        // Click to inspect mouth & tongue placement
        pill.addEventListener('click', () => {
            phonemeHeatmapEl.querySelectorAll('.phoneme-pill').forEach((p) => p.classList.remove('selected-pill'));
            pill.classList.add('selected-pill');

            if (itemInfo.articulation) {
                updateArticulationGuide(itemInfo.articulation, `${norm} (/${ipa}/)`);
            } else {
                fetchPhonemeArticulation(norm, ipa);
            }
        });

        // Auto-select weak phoneme if diagnosed
        if (weakPhoneme && weakPhoneme.toUpperCase() === norm) {
            pill.classList.add('selected-pill');
        }

        phonemeHeatmapEl.appendChild(pill);
    });
}

async function fetchPhonemeArticulation(phoneme, ipa) {
    try {
        const res = await fetch(`/api/phonemes?word=${encodeURIComponent(phoneme)}`);
        if (res.ok) {
            const data = await res.json();
            if (data.breakdown && data.breakdown.length > 0) {
                updateArticulationGuide(data.breakdown[0].articulation, `${phoneme} (/${ipa}/)`);
            }
        }
    } catch (e) {
        console.debug('Failed to fetch individual articulation:', e);
    }
}

// =============================================================================
// 7. Dynamic Mouth & Tongue Placement Guide
// =============================================================================

function updateArticulationGuide(guide, soundLabel) {
    if (!guide) return;

    if (articulationSoundBadge && soundLabel) {
        articulationSoundBadge.textContent = soundLabel;
    }
    if (artTongueEl && guide.tongue) {
        artTongueEl.textContent = guide.tongue;
    }
    if (artLipsEl && guide.lips) {
        artLipsEl.textContent = guide.lips;
    }
    if (artAirflowEl && guide.airflow) {
        artAirflowEl.textContent = guide.airflow;
    }
    if (artVoicingEl && guide.voicing) {
        artVoicingEl.textContent = guide.voicing;
    }
    if (artTipEl) {
        const tipText = [guide.drill_tip, guide.drift_trap].filter(Boolean).join(' ');
        artTipEl.textContent = tipText || 'Listen carefully to the slowed corrective model spoken by Rime Mist v3.';
    }
}

// =============================================================================
// 8. Categorized Practice Tracks (Minimal Pairs)
// =============================================================================

function renderTrack(trackId) {
    activeTrackId = trackId;
    const words = TRACKS_DATA[trackId] || TRACKS_DATA['all'];

    trackTabs.forEach((tab) => {
        if (tab.getAttribute('data-track') === trackId) tab.classList.add('active');
        else tab.classList.remove('active');
    });

    if (!trackWordsContainer) return;
    trackWordsContainer.innerHTML = '';
    words.forEach((item) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `vocab-pill ${item.word === currentTargetWord ? 'active' : ''}`;
        btn.setAttribute('data-word', item.word);
        btn.textContent = `${item.word} (${item.ipa})`;
        btn.addEventListener('click', () => {
            selectTargetWord(item.word, true, false);
        });
        trackWordsContainer.appendChild(btn);
    });
}

trackTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
        const trackId = tab.getAttribute('data-track');
        renderTrack(trackId);
    });
});

// =============================================================================
// 9. Custom Word Input & Suggestion Chips
// =============================================================================

if (customWordForm) {
    customWordForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const val = customWordInput ? customWordInput.value.trim() : '';
        if (val) {
            selectTargetWord(val, true, true);
            if (customWordInput) customWordInput.blur();
        }
    });
}

suggChips.forEach((chip) => {
    chip.addEventListener('click', () => {
        const word = chip.getAttribute('data-word');
        if (word) {
            if (customWordInput) customWordInput.value = word;
            selectTargetWord(word, true, true);
        }
    });
});

// =============================================================================
// 10. Status & Real-Time Metrics Updating
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
        if (w !== currentTargetWord) {
            currentTargetWord = w;
            targetWordEl.textContent = w;
            if (metricTargetWordEl) metricTargetWordEl.textContent = w;
            selectTargetWord(w, false);
        }
    }

    if (meta.ipa) {
        targetPhonemesEl.textContent = meta.ipa;
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
    if (detail && detail !== '—') {
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

    // 6. Update Phoneme Heatmap if phonemes and alignment provided
    if (Array.isArray(meta.expected_phonemes) && meta.expected_phonemes.length > 0) {
        renderPhonemeHeatmap(meta.expected_phonemes, meta.alignment, meta.phoneme);
    }

    // 7. Update Articulation Guide if provided
    if (meta.articulation) {
        const soundTag = meta.phoneme ? `${meta.phoneme} (/${meta.articulation.ipa || ''}/)` : 'WEAK SOUND';
        updateArticulationGuide(meta.articulation, soundTag);
    }

    // 8. Session history strip (last 5 attempts)
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
// 11. Live Microphone Frequency Visualizer
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
// 12. Transcript Logging
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
// 13. Page Initialization
// =============================================================================

window.addEventListener('DOMContentLoaded', () => {
    initWaveField();
    renderTrack('all');
    selectTargetWord('three', false, false);

    if (window.location.search.includes('coach=') || window.location.hash === '#coach') {
        setTimeout(connect, 400);
    }
});
