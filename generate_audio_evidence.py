"""
Audio Evidence Generator for Say That Sound — Rime TTS Pronunciation Coach.

Generates 3 full before/after audio evidence clips demonstrating:
  1. TH contrast ('three' vs 'tree') — weak phoneme TH, 0.5x corrective drill vs 1.0x pass
  2. SH contrast ('sheep' vs 'seep') — weak phoneme SH, 0.5x corrective drill vs 1.0x pass
  3. R/L contrast ('rice' vs 'lice') — weak phoneme R, 0.5x corrective drill vs 1.0x pass

Logs exact text, model ID, speaker, language, audio format, and latency for every Rime call.
Saves WAV audio files to evidence/ directory for judge audit.
"""

import asyncio
import logging
import os
import time
import wave
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from livekit.agents import utils
from livekit.plugins import rime

from config import (
    RIME_AUDIO_FORMAT,
    RIME_LANGUAGE,
    RIME_MODEL_ID,
    RIME_SPEAKER,
    SPEED_TIERS,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audio_evidence")

EVIDENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence")


async def synthesize_rime_segment(
    tts: rime.TTS,
    text: str,
    speed_alpha: float = 1.0,
) -> Tuple[bytearray, int, int, float]:
    """
    Synthesize speech using Rime TTS, measuring latency and capturing raw PCM frames.
    """
    tts.update_options(speed_alpha=speed_alpha)
    start_time = time.monotonic()

    logger.info(
        f"[RIME TTS CALL] text='{text}' | model='{RIME_MODEL_ID}' | speaker='{RIME_SPEAKER}' | "
        f"lang='{RIME_LANGUAGE}' | format='{RIME_AUDIO_FORMAT}' | speed_alpha={speed_alpha}"
    )

    stream = tts.synthesize(text)
    raw_pcm = bytearray()
    sample_rate = 22050
    channels = 1

    async for ev in stream:
        f = ev.frame
        sample_rate = f.sample_rate
        channels = f.num_channels
        raw_pcm.extend(f.data.tobytes() if hasattr(f.data, "tobytes") else bytes(f.data))

    latency_ms = (time.monotonic() - start_time) * 1000
    logger.info(
        f"[RIME TTS SUCCESS] text='{text}' | latency={latency_ms:.1f}ms | bytes={len(raw_pcm)}"
    )

    return raw_pcm, sample_rate, channels, latency_ms


def generate_silence_pcm(duration_sec: float, sample_rate: int = 22050, channels: int = 1) -> bytes:
    """Generate quiet background silence PCM (2 bytes per sample per channel)."""
    num_samples = int(sample_rate * duration_sec) * channels
    return b"\x00" * (num_samples * 2)


async def generate_clip(
    tts: rime.TTS,
    filename: str,
    target_word: str,
    weak_sound: str,
    substitution: str,
    coaching_phrase: str,
    isolated_sound: str,
    next_word: str,
) -> Dict[str, any]:
    """
    Construct a complete before/after coaching audio sequence:
      1. Corrective Coaching intro (0.5x slow-replay speed)
      2. Isolated weak phoneme demonstration
      3. Slowed target word (0.5x speed)
      4. "Your turn." prompt
      5. Pause
      6. Pass praise feedback (1.0x normal speed) and advancement to next word
    """
    full_pcm = bytearray()
    sample_rate = 22050
    channels = 1
    total_latency_ms = 0.0
    segments_meta = []

    # Segment 1: Coaching speech
    pcm1, sample_rate, channels, lat1 = await synthesize_rime_segment(
        tts, coaching_phrase, speed_alpha=SPEED_TIERS["slow_replay"]
    )
    full_pcm.extend(pcm1)
    full_pcm.extend(generate_silence_pcm(0.4, sample_rate, channels))
    total_latency_ms += lat1
    segments_meta.append({"stage": "coaching_speech", "text": coaching_phrase, "speed": 0.5, "latency_ms": lat1})

    # Segment 2: Isolated phoneme sound
    pcm2, _, _, lat2 = await synthesize_rime_segment(
        tts, isolated_sound, speed_alpha=SPEED_TIERS["slow_replay"]
    )
    full_pcm.extend(pcm2)
    full_pcm.extend(generate_silence_pcm(0.5, sample_rate, channels))
    total_latency_ms += lat2
    segments_meta.append({"stage": "isolated_phoneme", "text": isolated_sound, "speed": 0.5, "latency_ms": lat2})

    # Segment 3: Slowed target word at 0.5x
    pcm3, _, _, lat3 = await synthesize_rime_segment(
        tts, target_word, speed_alpha=SPEED_TIERS["slow_replay"]
    )
    full_pcm.extend(pcm3)
    full_pcm.extend(generate_silence_pcm(0.4, sample_rate, channels))
    total_latency_ms += lat3
    segments_meta.append({"stage": "slow_word", "text": target_word, "speed": 0.5, "latency_ms": lat3})

    # Segment 4: "Your turn." prompt
    pcm4, _, _, lat4 = await synthesize_rime_segment(
        tts, "Your turn.", speed_alpha=SPEED_TIERS["normal"]
    )
    full_pcm.extend(pcm4)
    full_pcm.extend(generate_silence_pcm(0.8, sample_rate, channels))
    total_latency_ms += lat4
    segments_meta.append({"stage": "your_turn", "text": "Your turn.", "speed": 1.0, "latency_ms": lat4})

    # Segment 5: Passing praise (learner improved >= 75%)
    praise = f"Great job! Your pronunciation of '{target_word}' was clear. Let's try '{next_word}' next."
    pcm5, _, _, lat5 = await synthesize_rime_segment(
        tts, praise, speed_alpha=SPEED_TIERS["normal"]
    )
    full_pcm.extend(pcm5)
    total_latency_ms += lat5
    segments_meta.append({"stage": "passing_praise", "text": praise, "speed": 1.0, "latency_ms": lat5})

    # Write WAV file
    filepath = os.path.join(EVIDENCE_DIR, filename)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(full_pcm)

    duration_s = len(full_pcm) / (sample_rate * channels * 2)
    logger.info(f"==> Wrote {filepath} ({len(full_pcm)} bytes, {duration_s:.1f}s)")

    return {
        "file": filename,
        "path": filepath,
        "target_word": target_word,
        "weak_sound": weak_sound,
        "substitution": substitution,
        "duration_sec": round(duration_s, 2),
        "total_audio_bytes": len(full_pcm),
        "sample_rate": sample_rate,
        "channels": channels,
        "total_latency_ms": round(total_latency_ms, 1),
        "avg_latency_ms": round(total_latency_ms / len(segments_meta), 1),
        "segments": segments_meta,
    }


async def main():
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    logger.info(f"Generating 3 Rime TTS audio evidence clips in: {EVIDENCE_DIR}")

    async with utils.http_context.open():
        tts = rime.TTS(
            model=RIME_MODEL_ID,
            speaker=RIME_SPEAKER,
            speed_alpha=1.0,
        )

        # Clip 1: TH minimal pair (three vs tree)
        clip1 = await generate_clip(
            tts=tts,
            filename="evidence_1_th_three.wav",
            target_word="three",
            weak_sound="TH",
            substitution="TH → T substitution",
            coaching_phrase="Let's work on the TH sound in three. Listen closely.",
            isolated_sound="{TH}",
            next_word="think",
        )

        # Clip 2: SH minimal pair (sheep vs seep)
        clip2 = await generate_clip(
            tts=tts,
            filename="evidence_2_sh_sheep.wav",
            target_word="sheep",
            weak_sound="SH",
            substitution="SH → S substitution",
            coaching_phrase="Let's focus on the SH sound in sheep. Listen first.",
            isolated_sound="{SH}",
            next_word="rice",
        )

        # Clip 3: R/L minimal pair (rice vs lice)
        clip3 = await generate_clip(
            tts=tts,
            filename="evidence_3_rl_rice.wav",
            target_word="rice",
            weak_sound="R",
            substitution="R → L substitution",
            coaching_phrase="Let's practice the R sound in rice. Listen to this.",
            isolated_sound="{R}",
            next_word="light",
        )

        logger.info("\n" + "=" * 60)
        logger.info("RIME AUDIO EVIDENCE GENERATION COMPLETE")
        logger.info("=" * 60)
        for clip in [clip1, clip2, clip3]:
            logger.info(
                f"- {clip['file']}: word='{clip['target_word']}' ({clip['substitution']}) | "
                f"duration={clip['duration_sec']}s | avg_tts_latency={clip['avg_latency_ms']}ms"
            )


if __name__ == "__main__":
    asyncio.run(main())
