"""
Repeatable acceptance test suite for Say That Sound — Step 3 Voice Agent.

Tests 3 minimal pair stress cases:
  1. 'th' contrast: 'three' (target) vs 'tree' (substitution: T for TH)
  2. 'sh' contrast: 'sheep' (target) vs 'seep' (substitution: S for SH)
  3. 'r/l' contrast: 'rice' (target) vs 'lice' (substitution: L for R)

Verifies:
  - Acoustic phoneme diagnosis & substitution detail (e.g. "TH → T substitution")
  - Confidence scoring (<60% slow-replay trigger vs >=75% passing advancement)
  - Corrective model speed tiering (0.5x slow-replay vs 1.0x normal)
  - Real Rime TTS synthesis latency benchmarking and exact call logging
  - Session history strip tracking with pass/fail (✓ / ✗) icons
  - Automatic vocabulary progression on passing attempts
"""

import asyncio
import logging
import os
import sys
import time
from typing import Dict, List

from dotenv import load_dotenv

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from config import (
    PASS_THRESHOLD,
    RIME_AUDIO_FORMAT,
    RIME_LANGUAGE,
    RIME_MODEL_ID,
    RIME_SPEAKER,
    SILENCE_TIMEOUT_SECONDS,
    SLOW_REPLAY_SPEED,
    SLOW_REPLAY_THRESHOLD,
    SPEED_TIERS,
    TARGET_VOCABULARY,
)
from phoneme_analyzer import PhonemeAnalyzer
from rime_drill import format_drill_playback, get_speed_alpha
from session_state import DrillState, PronunciationSessionState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("acceptance_test")


async def run_stress_test_case(
    case_num: int,
    target_word: str,
    mispronounced_transcript: str,
    expected_weak_phoneme: str,
    expected_substitution_contains: str,
    tts_client=None,
) -> Dict[str, any]:
    """Execute one full acceptance test case with before/after pronunciation cycle."""
    logger.info("\n" + "=" * 65)
    logger.info(f"TEST CASE {case_num}: Target='{target_word}' vs User Said='{mispronounced_transcript}'")
    logger.info("=" * 65)

    analyzer = PhonemeAnalyzer()
    state = PronunciationSessionState(current_target_word=target_word, speed_tier="normal")

    # -------------------------------------------------------------------------
    # Phase 1: Mispronounced Attempt (Confidence < 60% -> 0.5x Corrective Model)
    # -------------------------------------------------------------------------
    logger.info(f"--> Phase 1: Evaluating mispronounced attempt ('{mispronounced_transcript}')")
    diag_bad = analyzer.analyze_phonemes(
        word=target_word,
        spoken_transcript=mispronounced_transcript,
    )
    state.update_diagnosis(diag_bad)
    score_bad = state.diagnosis_confidence

    # Assertions for mispronounced attempt
    assert diag_bad.weak_phoneme == expected_weak_phoneme, (
        f"Expected weak phoneme '{expected_weak_phoneme}', got '{diag_bad.weak_phoneme}'"
    )
    assert diag_bad.weak_phoneme_detail and expected_substitution_contains in diag_bad.weak_phoneme_detail, (
        f"Expected substitution detail containing '{expected_substitution_contains}', got '{diag_bad.weak_phoneme_detail}'"
    )
    assert score_bad < SLOW_REPLAY_THRESHOLD, (
        f"Expected score < {SLOW_REPLAY_THRESHOLD}, got {score_bad}"
    )

    # Apply corrective drill logic
    state.set_drill_state(DrillState.COACHING)
    state.speed_tier = "slow_replay"
    state.add_history_entry(
        word=target_word,
        passed=False,
        confidence=score_bad,
        phoneme=diag_bad.weak_phoneme,
        weak_detail=diag_bad.weak_phoneme_detail,
    )

    # Format 0.5x corrective drill
    playback_bad = format_drill_playback(
        phoneme=diag_bad.weak_phoneme,
        word=target_word,
        speed_tier="slow_replay",
    )
    assert playback_bad["speed_alpha"] == 0.50, (
        f"Expected speed_alpha 0.50 for slow-replay, got {playback_bad['speed_alpha']}"
    )

    # Rime TTS synthesis of corrective model
    tts_bad_latency = 0.0
    if tts_client:
        tts_client.update_options(speed_alpha=0.50)
        t0 = time.monotonic()
        corrective_text = f"Let's focus on the {diag_bad.weak_phoneme} sound in {target_word}."
        logger.info(
            f"[RIME TTS LOG] text='{corrective_text}' | model='{RIME_MODEL_ID}' | speaker='{RIME_SPEAKER}' | "
            f"format='{RIME_AUDIO_FORMAT}' | speed_alpha=0.50"
        )
        stream = tts_client.synthesize(corrective_text)
        async for _ in stream:
            pass
        tts_bad_latency = (time.monotonic() - t0) * 1000
        logger.info(f"[RIME TTS BENCHMARK] Corrective model synthesized in {tts_bad_latency:.1f}ms")

    # -------------------------------------------------------------------------
    # Phase 2: Improved Pronunciation Attempt (Confidence >= 75% -> 1.0x Pass)
    # -------------------------------------------------------------------------
    logger.info(f"--> Phase 2: Evaluating improved attempt ('{target_word}')")
    diag_good = analyzer.analyze_phonemes(
        word=target_word,
        spoken_transcript=target_word,
    )
    state.update_diagnosis(diag_good)
    score_good = state.diagnosis_confidence

    assert diag_good.diagnosis_status == "perfect", (
        f"Expected status 'perfect', got '{diag_good.diagnosis_status}'"
    )
    assert score_good >= PASS_THRESHOLD, (
        f"Expected score >= {PASS_THRESHOLD}, got {score_good}"
    )

    # Apply pass & advance logic
    state.set_drill_state(DrillState.COACHING)
    state.speed_tier = "normal"
    state.add_history_entry(
        word=target_word,
        passed=True,
        confidence=score_good,
        phoneme=None,
        weak_detail="Clear",
    )
    next_word = state.advance_to_next_word(TARGET_VOCABULARY)

    assert next_word != target_word, (
        f"Target word should advance from '{target_word}' to next word"
    )

    # Rime TTS synthesis of passing praise
    tts_good_latency = 0.0
    if tts_client:
        tts_client.update_options(speed_alpha=1.0)
        t0 = time.monotonic()
        praise_text = f"Great job! Your pronunciation of '{target_word}' was clear. Let's try '{next_word}' next."
        logger.info(
            f"[RIME TTS LOG] text='{praise_text}' | model='{RIME_MODEL_ID}' | speaker='{RIME_SPEAKER}' | "
            f"format='{RIME_AUDIO_FORMAT}' | speed_alpha=1.0"
        )
        stream = tts_client.synthesize(praise_text)
        async for _ in stream:
            pass
        tts_good_latency = (time.monotonic() - t0) * 1000
        logger.info(f"[RIME TTS BENCHMARK] Passing praise synthesized in {tts_good_latency:.1f}ms")

    # Verify session history strip has 2 entries (fail then pass)
    history = state.session_history
    assert len(history) == 2
    assert history[0]["passed"] is False and history[0]["word"] == target_word
    assert history[1]["passed"] is True and history[1]["word"] == target_word

    return {
        "case": case_num,
        "target": target_word,
        "mispronounced": mispronounced_transcript,
        "weak_phoneme": diag_bad.weak_phoneme,
        "substitution_detail": diag_bad.weak_phoneme_detail,
        "bad_score": score_bad,
        "good_score": score_good,
        "corrective_speed": "0.5x",
        "next_word": next_word,
        "tts_corrective_latency_ms": round(tts_bad_latency, 1),
        "tts_praise_latency_ms": round(tts_good_latency, 1),
        "status": "PASSED",
    }


async def main():
    logger.info("=================================================================")
    logger.info("   SAY THAT SOUND — REPEATABLE ACCEPTANCE TEST SUITE            ")
    logger.info("=================================================================")

    # Initialize Rime TTS client with http_context
    from livekit.agents import utils
    from livekit.plugins import rime

    async with utils.http_context.open():
        tts = rime.TTS(
            model=RIME_MODEL_ID,
            speaker=RIME_SPEAKER,
            speed_alpha=1.0,
        )

        results = []

        # Stress Case 1: 'th' contrast (three vs tree)
        r1 = await run_stress_test_case(
            case_num=1,
            target_word="three",
            mispronounced_transcript="tree",
            expected_weak_phoneme="TH",
            expected_substitution_contains="TH → T",
            tts_client=tts,
        )
        results.append(r1)

        # Stress Case 2: 'sh' contrast (sheep vs seep)
        r2 = await run_stress_test_case(
            case_num=2,
            target_word="sheep",
            mispronounced_transcript="seep",
            expected_weak_phoneme="SH",
            expected_substitution_contains="SH → S",
            tts_client=tts,
        )
        results.append(r2)

        # Stress Case 3: 'r/l' contrast (rice vs lice)
        r3 = await run_stress_test_case(
            case_num=3,
            target_word="rice",
            mispronounced_transcript="lice",
            expected_weak_phoneme="R",
            expected_substitution_contains="R → L",
            tts_client=tts,
        )
        results.append(r3)

        # Print final summary audit table
        logger.info("\n" + "=" * 80)
        logger.info("ACCEPTANCE TEST SUITE SUMMARY")
        logger.info("=" * 80)
        print(f"{'Case':<5} | {'Target':<7} | {'Weak Sound':<10} | {'Substitution':<24} | {'Bad Score':<9} | {'Good Score':<10} | {'TTS Latency':<12} | {'Result'}")
        print("-" * 105)
        for r in results:
            sub_clean = (r['substitution_detail'] or '').replace('→', '->')
            print(
                f"{r['case']:<5} | {r['target']:<7} | {r['weak_phoneme']:<10} | {sub_clean:<24} | "
                f"{int(r['bad_score']*100)}% (<60%) | {int(r['good_score']*100)}% (>=75%) | "
                f"{r['tts_corrective_latency_ms']}ms / {r['tts_praise_latency_ms']}ms | {r['status']}"
            )
        print("-" * 105)
        logger.info("All 3 minimal pair stress cases verified successfully!")


if __name__ == "__main__":
    asyncio.run(main())
