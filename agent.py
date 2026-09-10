"""
Say That Sound — Step 3 Voice Agent with Product Hardening & Drill UX

Pipeline:
  User Speech Audio → Deepgram Streaming STT → Command Interceptor
  → [If Command: again/slower/normal/stop] → Update State & Replay
  → [If Pronunciation Attempt] → Target Word Resolution (Safe Dict Lookup)
  → Local Phoneme CTC Analysis (Needleman-Wunsch Alignment)
  → Weak Phoneme Selection + Honest Confidence Scoring
  → GPT-OSS Structured Drill Orchestration (with Action Validation)
  → Cancellable DrillPlaybackTask (Generation ID Guard):
      Stage 1: Coaching Speech
      Stage 2: Isolated Weak Phoneme ({TH}, {SH}, etc.)
      Stage 3: Brief Pause
      Stage 4: Slowed Target Word (speed_alpha: normal=1.0, slow=0.8, slower=0.65)
      Stage 5: "Your turn" prompt
  → Rime Mist v3 WebSocket Streaming → LiveKit WebRTC Audio Output

Full Duplex & Barge-In:
  Silero VAD detects user speech during agent playback.
  AgentSession immediately halts Rime audio playback, cancels in-flight synthesis,
  and flushes buffers.
  CancellableDrillPlaybackTask generation guard ensures stale audio never leaks.
  Session state (current_target_word, weak_phoneme, speed_tier) is preserved across
  interruptions and drill commands ("again", "slower", "normal speed").
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.voice.speech_handle import SpeechHandle
from livekit.plugins import deepgram, rime, silero

from config import (
    CONFIDENCE_THRESHOLD,
    DEEPGRAM_MODEL,
    GPT_OSS_API_KEY,
    GPT_OSS_BASE_URL,
    GPT_OSS_MODEL,
    INJECT_TTS_DELAY_MS,
    PASS_THRESHOLD,
    RIME_AUDIO_FORMAT,
    RIME_LANGUAGE,
    RIME_MODEL,
    RIME_MODEL_ID,
    RIME_SPEAKER,
    RIME_SPEED_ALPHA,
    SILENCE_TIMEOUT_SECONDS,
    SLOW_REPLAY_THRESHOLD,
    SPEED_TIERS,
    SYSTEM_PROMPT,
    TARGET_VOCABULARY,
)
from drill_manager import CommandType, CancellableDrillPlaybackTask, intercept_command
from gpt_oss_orchestrator import GptOssOrchestrator
from phoneme_analyzer import PhonemeAnalyzer
from phoneme_dict import (
    get_expected_phonemes,
    get_ipa_transcription,
    get_phoneme_articulation,
    get_word_phoneme_breakdown,
    has_pronunciation,
    safe_lookup,
)
from rime_drill import format_drill_playback, get_speed_alpha
from session_logger import SessionLogger
from session_state import DrillState, PronunciationSessionState

load_dotenv()

logger = logging.getLogger("say-that-sound")
logger.setLevel(logging.INFO)


from livekit.agents import llm


class PronunciationAgent(Agent):
    """Pronunciation coaching agent powered by GPT-OSS orchestration."""

    def __init__(self, on_turn_callback=None) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)
        self.on_turn_callback = on_turn_callback

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Official LiveKit hook: called once user turn is fully endpointed and committed."""
        user_text = new_message.text_content
        if not user_text or not user_text.strip():
            return
        logger.info(f"[TURN COMMITTED] User utterance: {user_text.strip()}")
        if self.on_turn_callback:
            await self.on_turn_callback(user_text.strip())


def extract_target_word(transcript: str, fallback_vocab: list) -> Optional[str]:
    """
    Extract the target word from user speech.
    Supports Mode A (explicit: "practice the word three", "say sheep")
    and Mode B (vocabulary word detection).
    """
    text = transcript.lower().strip()

    # Pattern: "practice [the word] <word>", "say <word>", "pronounce <word>"
    match = re.search(r"(?:practice(?: the word)?|pronounce|word is|say)\s+([a-zA-Z]+)", text)
    if match:
        return match.group(1).lower()

    # Direct vocabulary match
    for w in fallback_vocab:
        # Match as discrete word boundary
        if re.search(rf"\b{re.escape(w)}\b", text):
            return w

    # Single-word utterance check against dictionary
    clean_words = re.findall(r"[a-zA-Z]+", text)
    if len(clean_words) == 1 and has_pronunciation(clean_words[0]):
        return clean_words[0].lower()

    return None


async def entrypoint(ctx: JobContext) -> None:
    """Called by the LiveKit agent framework when a new job is dispatched."""
    logger.info("Step 3 Agent entrypoint starting — waiting for participant…")
    await ctx.connect(auto_subscribe="subscribe_all")

    participant = await ctx.wait_for_participant()
    logger.info(f"Participant connected: {participant.identity}")

    # ---- Session State & Engine Initialization ----
    sess_logger = SessionLogger(session_id=participant.identity)
    sess_logger.log_event("session_start", {"participant": participant.identity, "step": 3})

    state = PronunciationSessionState(current_target_word=TARGET_VOCABULARY[0], speed_tier="slow")
    analyzer = PhonemeAnalyzer(confidence_threshold=CONFIDENCE_THRESHOLD)
    orchestrator = GptOssOrchestrator(
        model=GPT_OSS_MODEL,
        base_url=GPT_OSS_BASE_URL,
        api_key=GPT_OSS_API_KEY,
    )

    # ---- Build LiveKit Pipeline Components ----
    stt = deepgram.STT(model=DEEPGRAM_MODEL)
    tts = rime.TTS(
        model=RIME_MODEL,
        speaker=RIME_SPEAKER,
        use_websocket=True,
        speed_alpha=RIME_SPEED_ALPHA,
    )
    vad = silero.VAD.load()

    session = AgentSession(
        stt=stt,
        tts=tts,
        vad=vad,
        turn_handling={
            "interruption": {
                "min_duration": 0.5,
                "min_words": 1,
            },
        },
    )

    _current_user_text = ""
    _current_agent_text = ""
    _turn_speech_start: float | None = None
    _current_speech_handle: Optional[SpeechHandle] = None
    _is_agent_speaking = False
    _current_drill_task: Optional[CancellableDrillPlaybackTask] = None
    _audio_buffer = bytearray()

    async def _consume_user_audio(track: rtc.Track):
        """Buffer incoming raw PCM audio frames from the user's microphone."""
        nonlocal _audio_buffer
        try:
            stream = rtc.AudioStream(track)
            async for event in stream:
                frame = event.frame
                pcm = frame.data.tobytes() if hasattr(frame.data, "tobytes") else bytes(frame.data)
                _audio_buffer.extend(pcm)
                # Keep rolling buffer of last ~3 seconds of 16kHz 16-bit mono audio
                if len(_audio_buffer) > 128000:
                    _audio_buffer = _audio_buffer[-96000:]
        except Exception as e:
            logger.debug(f"Audio stream subscription ended: {e}")

    @ctx.room.on("track_subscribed")
    def _on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Subscribed to user microphone audio track from {participant.identity}")
            asyncio.create_task(_consume_user_audio(track))

    @ctx.room.on("data_received")
    def _on_data_received(data_packet: rtc.DataPacket):
        """Handle incoming client data messages (e.g. word selection from custom bar or tracks)."""
        try:
            raw_bytes = data_packet.data
            text = raw_bytes.decode("utf-8") if isinstance(raw_bytes, (bytes, bytearray)) else str(raw_bytes)
            msg = json.loads(text)
            if msg.get("type") == "select_word":
                selected = msg.get("word", "").strip().lower()
                track_words = msg.get("track_words", [])
                if selected and has_pronunciation(selected):
                    logger.info(f"[CLIENT DATA] Selected target word: '{selected}'")
                    state.current_target_word = selected
                    state.expected_phonemes = get_expected_phonemes(selected)
                    if track_words:
                        state.active_track_words = [w.lower() for w in track_words]
                    else:
                        state.active_track_words = []
                    state.weak_phoneme = None
                    state.weak_phoneme_detail = None
                    state.diagnosis_confidence = 0.0
                    state.diagnosis_status = "idle"
                    state.speed_tier = "slow"
                    state.alignment = []
                    state._bump_generation()
                    asyncio.create_task(_broadcast_state())
                    prompt = f"Target word set to {selected}. Say {selected} whenever you're ready."
                    _speak_fire_and_forget(prompt)
        except Exception as e:
            logger.debug(f"Error handling client data packet: {e}")

    _silence_timer_task: Optional[asyncio.Task] = None

    def _cancel_silence_watchdog() -> None:
        nonlocal _silence_timer_task
        if _silence_timer_task and not _silence_timer_task.done():
            _silence_timer_task.cancel()
            _silence_timer_task = None

    def _start_silence_watchdog() -> None:
        nonlocal _silence_timer_task
        _cancel_silence_watchdog()

        async def _watchdog_timer():
            try:
                await asyncio.sleep(SILENCE_TIMEOUT_SECONDS)
                if not _is_agent_speaking and ctx.room and ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                    logger.info(f"[WATCHDOG] {SILENCE_TIMEOUT_SECONDS}s silence detected — prompting user")
                    state.set_drill_state(DrillState.WAITING_FOR_RETRY)
                    await _broadcast_state()
                    _speak_fire_and_forget("I didn't catch that, try again.")
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"Watchdog exception: {e}")

        _silence_timer_task = asyncio.create_task(_watchdog_timer())

    async def _broadcast_state() -> None:
        """Broadcast current drill state to the web client via data channel."""
        try:
            state_val = state.drill_state.value.lower()
            if state_val in ("idle",):
                badge_state = "IDLE"
            elif state_val in ("listening", "waiting_for_retry"):
                badge_state = "LISTENING"
            elif state_val in ("analyzing", "scoring"):
                badge_state = "SCORING"
            elif state_val in ("coaching", "demo_phoneme", "demo_word"):
                badge_state = "COACHING"
            else:
                badge_state = state_val.upper()

            speed_alpha = SPEED_TIERS.get(state.speed_tier, 1.0)
            expected_phs = state.expected_phonemes or get_expected_phonemes(state.current_target_word)
            ipa_str = get_ipa_transcription(expected_phs)
            articulation_guide = get_phoneme_articulation(state.weak_phoneme) if state.weak_phoneme else None

            payload = json.dumps({
                "type": "drill_state",
                "word": state.current_target_word,
                "phoneme": state.weak_phoneme or "—",
                "phoneme_detail": state.weak_phoneme_detail or (state.weak_phoneme if state.weak_phoneme else "—"),
                "confidence": state.diagnosis_confidence,
                "speed_tier": state.speed_tier,
                "speed_alpha": speed_alpha,
                "drill_state": badge_state,
                "coach_text": _current_agent_text,
                "history": state.session_history,
                "expected_phonemes": expected_phs,
                "alignment": getattr(state, "alignment", []),
                "ipa": ipa_str,
                "articulation": articulation_guide,
            })
            if ctx.room and ctx.room.local_participant:
                await ctx.room.local_participant.publish_data(payload.encode("utf-8"), reliable=True)
        except Exception as e:
            logger.debug(f"Failed to publish drill state: {e}")

    async def _speak_stage(text: str) -> None:
        """
        Synthesize and play one discrete stage of speech via LiveKit session with barge-in support.
        Awaits playback completion. Releasing cleanly when finished.
        """
        nonlocal _current_speech_handle, _current_agent_text, _turn_speech_start, _is_agent_speaking

        if not text or not text.strip():
            return

        _current_agent_text = text
        start_mono = time.monotonic()
        current_alpha = getattr(tts._opts, "speed_alpha", 1.0)

        # Log exact Rime TTS invocation with parameters
        logger.info(
            f"[RIME TTS INVOCATION] text='{text}' | model='{RIME_MODEL_ID}' | speaker='{RIME_SPEAKER}' | "
            f"language='{RIME_LANGUAGE}' | format='{RIME_AUDIO_FORMAT}' | speed_alpha={current_alpha}"
        )

        handle = session.say(text, allow_interruptions=True)
        _current_speech_handle = handle

        loop = asyncio.get_event_loop()
        done_fut = loop.create_future()

        def _on_speech_done(h: SpeechHandle) -> None:
            if not done_fut.done():
                done_fut.set_result(h)

        handle.add_done_callback(_on_speech_done)

        try:
            h = await done_fut
            latency_ms = (time.monotonic() - start_mono) * 1000
            logger.info(f"[RIME TTS COMPLETED] text='{text}' | latency={latency_ms:.1f}ms")
            sess_logger.log_event("rime_tts_call", {
                "text": text,
                "model": RIME_MODEL_ID,
                "speaker": RIME_SPEAKER,
                "language": RIME_LANGUAGE,
                "audio_format": RIME_AUDIO_FORMAT,
                "speed_alpha": current_alpha,
                "latency_ms": round(latency_ms, 1),
            })

            if h.interrupted and _is_agent_speaking:
                state.on_interrupt()
                interrupt_elapsed_ms = None
                if _turn_speech_start is not None:
                    interrupt_elapsed_ms = (time.monotonic() - _turn_speech_start) * 1000

                logger.info(
                    f"[INTERRUPT] Speech interrupted mid-playback! "
                    f"Preserved state: word='{state.current_target_word}', "
                    f"phoneme='{state.weak_phoneme}', speed='{state.speed_tier}'"
                )

                sess_logger.log_interrupt(
                    word=state.current_target_word,
                    phoneme=state.weak_phoneme,
                    speed_tier=state.speed_tier,
                    drill_state=state.drill_state.value,
                    elapsed_ms=interrupt_elapsed_ms,
                    drill_id=state.drill_id,
                    interaction_generation=state.interaction_generation,
                    stale_audio_discarded=True,
                    rime_cancelled=True,
                )

                sess_logger.log_turn(
                    user_transcript=_current_user_text,
                    agent_response=_current_agent_text,
                    interrupted=True,
                    interrupt_elapsed_ms=interrupt_elapsed_ms,
                    drill_id=state.drill_id,
                    interaction_generation=state.interaction_generation,
                    drill_state=state.drill_state.value,
                )
                asyncio.create_task(_broadcast_state())
                raise asyncio.CancelledError()
            else:
                sess_logger.log_turn(
                    user_transcript=_current_user_text,
                    agent_response=_current_agent_text,
                    interrupted=False,
                    drill_id=state.drill_id,
                    interaction_generation=state.interaction_generation,
                    drill_state=state.drill_state.value,
                )
        except asyncio.CancelledError:
            if handle and not handle.done():
                handle.interrupt()
            raise

    def _speak_fire_and_forget(text: str) -> None:
        """Non-blocking fire-and-forget speech for welcome greetings or simple responses."""
        asyncio.create_task(_speak_stage(text))

    # ---- Event Hooks ----

    @session.on("user_input_transcribed")
    def _on_user_transcribed(ev) -> None:
        nonlocal _current_user_text
        _cancel_silence_watchdog()
        state.set_drill_state(DrillState.LISTENING)
        asyncio.create_task(_broadcast_state())
        if not getattr(ev, "is_final", True):
            return
        transcript = ev.transcript
        if not transcript or not transcript.strip():
            return
        _current_user_text = transcript.strip()
        sess_logger.mark_stt_done()
        logger.info(f"[USER] {_current_user_text}")

    async def _process_turn(user_text: str) -> None:
        nonlocal _current_agent_text, _current_speech_handle, _current_drill_task, _audio_buffer

        _cancel_silence_watchdog()

        # Only interrupt speech if the agent is actively speaking
        if _is_agent_speaking and _current_speech_handle and not _current_speech_handle.done():
            _current_speech_handle.interrupt()
            _current_speech_handle = None

        # Cancel any pending drill task
        if _current_drill_task and not _current_drill_task.is_cancelled:
            _current_drill_task.cancel()

        # Capture user audio frames accumulated during speech
        user_audio = bytes(_audio_buffer)
        _audio_buffer.clear()

        norm_text = user_text.lower().strip()

        # Step 3: Deterministic command interception FIRST
        cmd_type = intercept_command(norm_text)

        if cmd_type == CommandType.AGAIN:
            action = orchestrator.orchestrate_action(user_text, state)
            sess_logger.log_drill_command(
                command="again",
                word=state.current_target_word,
                phoneme=state.weak_phoneme,
                drill_id=state.drill_id,
                interaction_generation=state.interaction_generation,
            )
            await _play_drill_action(action)
            state.set_drill_state(DrillState.LISTENING)
            await _broadcast_state()
            _start_silence_watchdog()
            return

        if cmd_type == CommandType.SLOWER:
            prev_speed = state.speed_tier
            action = orchestrator.orchestrate_action(user_text, state)
            sess_logger.log_drill_command(
                command="slower",
                word=state.current_target_word,
                phoneme=state.weak_phoneme,
                previous_speed=prev_speed,
                new_speed=state.speed_tier,
                drill_id=state.drill_id,
                interaction_generation=state.interaction_generation,
            )
            await _play_drill_action(action)
            state.set_drill_state(DrillState.LISTENING)
            await _broadcast_state()
            _start_silence_watchdog()
            return

        if cmd_type == CommandType.NORMAL_SPEED:
            prev_speed = state.speed_tier
            action = orchestrator.orchestrate_action(user_text, state)
            sess_logger.log_drill_command(
                command="normal_speed",
                word=state.current_target_word,
                phoneme=state.weak_phoneme,
                previous_speed=prev_speed,
                new_speed="normal",
                drill_id=state.drill_id,
                interaction_generation=state.interaction_generation,
            )
            await _play_drill_action(action)
            state.set_drill_state(DrillState.LISTENING)
            await _broadcast_state()
            _start_silence_watchdog()
            return

        if cmd_type == CommandType.STOP:
            action = orchestrator.orchestrate_action(user_text, state)
            sess_logger.log_drill_command(
                command="stop",
                word=state.current_target_word,
                phoneme=state.weak_phoneme,
                drill_id=state.drill_id,
                interaction_generation=state.interaction_generation,
            )
            logger.info(f"[AGENT] {action.spoken_response}")
            sess_logger.mark_llm_first_token()
            state.handle_stop()
            await _broadcast_state()
            _speak_fire_and_forget(action.spoken_response)
            return

        # Friendly greeting / conversational check if not attempting a target word
        clean_no_punct = re.sub(r"[^\w\s]", "", norm_text).strip()
        if clean_no_punct in (
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "how are you",
            "test",
            "testing",
        ):
            greeting_msg = (
                "Hi there! Say a word like three, ship, sheep, rice, light, or right to start your practice."
            )
            logger.info(f"[AGENT] {greeting_msg}")
            sess_logger.mark_llm_first_token()
            state.set_drill_state(DrillState.COACHING)
            await _broadcast_state()
            await _speak_stage(greeting_msg)
            state.set_drill_state(DrillState.LISTENING)
            await _broadcast_state()
            _start_silence_watchdog()
            return

        # Identify target word (Explicit command Mode A or Initial Selection Mode B)
        norm_text = user_text.lower().strip()
        explicit_match = re.search(
            r"(?:practice(?: the word)?|pronounce|word is|say|coach|switch to|let's try|let's do)\s+([a-zA-Z]+)",
            norm_text,
        )

        target = None
        if explicit_match:
            # User explicitly requested a target word
            target = explicit_match.group(1).lower()
        elif not state.current_target_word or len(state.session_history) == 0:
            # At start of session with no attempts yet, speech can select initial target
            target = extract_target_word(user_text, state.active_track_words or TARGET_VOCABULARY)

        if target:
            # Safe dictionary check
            if not has_pronunciation(target):
                miss_text = (
                    f"I don't have a pronunciation reference for '{target}' yet. Try words like three, ship, sheep, rice, light, or right."
                )
                logger.info(f"[AGENT] Dictionary miss: {target}")
                sess_logger.log_event("dictionary_miss", {"word": target})
                sess_logger.mark_llm_first_token()
                state.set_drill_state(DrillState.COACHING)
                await _broadcast_state()
                await _speak_stage(miss_text)
                state.set_drill_state(DrillState.LISTENING)
                await _broadcast_state()
                _start_silence_watchdog()
                return
            state.current_target_word = target
            state.expected_phonemes = get_expected_phonemes(target)

        word_to_analyze = state.current_target_word or TARGET_VOCABULARY[0]

        # Analyze pronunciation using actual microphone audio and STT transcript
        state.set_drill_state(DrillState.ANALYZING)
        await _broadcast_state()

        diag = analyzer.analyze_phonemes(
            word=word_to_analyze,
            audio_frames=user_audio,
            spoken_transcript=user_text,
        )
        state.update_diagnosis(diag)

        # Log structured diagnosis
        sess_logger.log_pronunciation_diagnosis(
            word=diag.word,
            expected_phonemes=diag.expected_phonemes,
            observed_phonemes=diag.observed_phonemes,
            weak_phoneme=diag.weak_phoneme,
            confidence=diag.weak_phoneme_confidence,
            status=diag.diagnosis_status,
            alignment=[item.to_dict() for item in diag.alignment],
            model_name=diag.model_name,
            drill_id=state.drill_id,
            interaction_generation=state.interaction_generation,
        )

        score = state.diagnosis_confidence
        logger.info(
            f"[PRONUNCIATION EVAL] word='{word_to_analyze}' score={score:.2f} "
            f"weak_phoneme={diag.weak_phoneme} detail='{diag.weak_phoneme_detail}' "
            f"status={diag.diagnosis_status}"
        )

        # 1. Confidence < 60%: Rime speaks corrective model at 0.5x speed emphasizing weak sound (COACHING state)
        if score < SLOW_REPLAY_THRESHOLD:
            state.set_drill_state(DrillState.COACHING)
            state.speed_tier = "slow_replay"
            state.add_history_entry(
                word=word_to_analyze,
                passed=False,
                confidence=score,
                phoneme=diag.weak_phoneme,
                weak_detail=diag.weak_phoneme_detail,
            )
            await _broadcast_state()

            action = await orchestrator.orchestrate_action_async(
                user_transcript=user_text,
                state=state,
                diagnosis=diag.to_dict(),
            )
            action = orchestrator.validate_action(action, state, diag.to_dict())
            action.speed_tier = "slow_replay"
            if diag.weak_phoneme:
                action.action = "DEMO_PHONEME"
                action.phoneme = diag.weak_phoneme
                if not action.spoken_response or len(action.spoken_response) < 5:
                    action.spoken_response = f"Let's work on the {diag.weak_phoneme} sound in '{word_to_analyze}'. Listen to this."
            else:
                action.action = "DEMO_WORD"
                if not action.spoken_response or len(action.spoken_response) < 5:
                    action.spoken_response = f"Let's try '{word_to_analyze}' again slowly."

            await _play_drill_action(action)

        # 2. Confidence >= 75%: Rime gives positive feedback at 1.0x, marks ✓ in history, advances to next word
        elif score >= PASS_THRESHOLD:
            state.set_drill_state(DrillState.COACHING)
            state.speed_tier = "normal"
            state.add_history_entry(
                word=word_to_analyze,
                passed=True,
                confidence=score,
                phoneme=None,
                weak_detail="Clear",
            )
            vocab_to_use = state.active_track_words if state.active_track_words else TARGET_VOCABULARY
            next_word = state.advance_to_next_word(vocab_to_use)
            await _broadcast_state()

            if next_word != word_to_analyze:
                praise_msg = f"Great job! Your pronunciation of '{word_to_analyze}' was clear. Next word is {next_word}."
            else:
                praise_msg = f"Great job! Your pronunciation of '{word_to_analyze}' was clear and spot on! Say {word_to_analyze} again, or choose another word to practice."
            tts._opts.speed_alpha = SPEED_TIERS.get("normal", 1.0)
            logger.info(f"[AGENT] {praise_msg}")
            sess_logger.mark_llm_first_token()
            await _speak_stage(praise_msg)

        # 3. 60% <= score < 75%: close attempt, prompt retry
        else:
            state.set_drill_state(DrillState.COACHING)
            state.speed_tier = "normal"
            state.add_history_entry(
                word=word_to_analyze,
                passed=False,
                confidence=score,
                phoneme=diag.weak_phoneme,
                weak_detail="Close",
            )
            await _broadcast_state()

            retry_msg = f"Good attempt on '{word_to_analyze}', you're close! Try saying it once more."
            tts._opts.speed_alpha = SPEED_TIERS.get("normal", 1.0)
            logger.info(f"[AGENT] {retry_msg}")
            sess_logger.mark_llm_first_token()
            await _speak_stage(retry_msg)

        # Finished coaching turn -> back to listening, start 5s silence watchdog
        state.set_drill_state(DrillState.LISTENING)
        await _broadcast_state()
        _start_silence_watchdog()

    async def _play_drill_action(action) -> None:
        nonlocal _current_agent_text, _current_drill_task

        if _current_drill_task and not _current_drill_task.is_cancelled:
            _current_drill_task.cancel()

        playback = format_drill_playback(
            phoneme=action.phoneme,
            word=action.word,
            speed_tier=action.speed_tier,
        )

        def _update_tts_speed(speed_alpha: float):
            tts._opts.speed_alpha = speed_alpha

        # Log pronunciation demonstration
        sess_logger.log_pronunciation_demo(
            word=action.word,
            phoneme=action.phoneme,
            speed_tier=action.speed_tier,
            model=playback["model"],
            speaker=playback["speaker"],
            drill_id=state.drill_id,
            interaction_generation=state.interaction_generation,
            drill_state=state.drill_state.value,
        )

        logger.info(f"[AGENT] Action: {action.action}, word: {action.word}, phoneme: {action.phoneme}")
        sess_logger.mark_llm_first_token()

        if action.action == "DEMO_PHONEME":
            drill_task = CancellableDrillPlaybackTask(
                generation_id=state.interaction_generation,
                state=state,
                speak_fn=_speak_stage,
            )
            _current_drill_task = drill_task
            await _broadcast_state()
            await drill_task.execute(
                coaching_text=action.spoken_response,
                phoneme_repr=playback["isolated_repr"],
                word=action.word,
                speed_alpha=playback["speed_alpha"],
                tts_update_fn=_update_tts_speed,
            )
        elif action.action in ("SLOW_DOWN", "NORMAL_SPEED", "DEMO_WORD"):
            drill_task = CancellableDrillPlaybackTask(
                generation_id=state.interaction_generation,
                state=state,
                speak_fn=_speak_stage,
            )
            _current_drill_task = drill_task
            await _broadcast_state()
            await drill_task.execute(
                coaching_text=action.spoken_response,
                phoneme_repr="",
                word=action.word,
                speed_alpha=playback["speed_alpha"],
                tts_update_fn=_update_tts_speed,
            )
        elif action.action == "ASK_RETRY":
            state.set_drill_state(DrillState.WAITING_FOR_RETRY)
            await _broadcast_state()
            await _speak_stage(action.spoken_response)
        else:
            state.set_drill_state(DrillState.IDLE)
            await _broadcast_state()
            await _speak_stage(action.spoken_response)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev) -> None:
        nonlocal _turn_speech_start, _is_agent_speaking
        if ev.new_state == "speaking":
            _is_agent_speaking = True
            _cancel_silence_watchdog()
            _turn_speech_start = time.monotonic()
            sess_logger.mark_tts_first_byte()
            logger.info("[AGENT] Started speaking (TTS playback)")
        else:
            _is_agent_speaking = False

    # ---- Inject artificial delay for barge-in test ----
    if INJECT_TTS_DELAY_MS > 0:
        logger.warning(f"INJECT_TTS_DELAY_MS={INJECT_TTS_DELAY_MS} active")
        _orig_synth = tts.synthesize

        async def _delayed_synth(*args, **kwargs):
            await asyncio.sleep(INJECT_TTS_DELAY_MS / 1000.0)
            return await _orig_synth(*args, **kwargs)

        tts.synthesize = _delayed_synth  # type: ignore[assignment]

    agent_instance = PronunciationAgent(on_turn_callback=_process_turn)
    await session.start(
        room=ctx.room,
        agent=agent_instance,
    )
    logger.info("Say That Sound Step 3 Agent ready and listening.")

    # Speak welcome greeting and broadcast initial state to web client
    welcome_msg = "Welcome to Say That Sound! Say a word like three, ship, sheep, rice, light, or right to begin."
    logger.info(f"[AGENT] {welcome_msg}")
    state.set_drill_state(DrillState.COACHING)
    await _broadcast_state()
    await _speak_stage(welcome_msg)
    state.set_drill_state(DrillState.LISTENING)
    await _broadcast_state()
    _start_silence_watchdog()

    await asyncio.Event().wait()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
