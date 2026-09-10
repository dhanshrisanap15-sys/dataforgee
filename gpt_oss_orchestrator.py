"""
GPT-OSS orchestration module for Say That Sound — Step 3.

Critical architectural rule:
  GPT-OSS must NOT determine which phoneme was mispronounced.
  It receives the structured diagnosis from PhonemeAnalyzer and handles:
    - Conversational coaching phrasing
    - Deciding the appropriate drill action
    - Emitting structured JSON actions

Supported actions:
  - DEMO_PHONEME: Demonstrate the isolated weak phoneme and then the slow word
  - DEMO_WORD: Demonstrate the full target word
  - REPEAT_DRILL: Repeat current drill ("again")
  - SLOW_DOWN: Step to next slower speed tier ("slower")
  - NORMAL_SPEED: Reset to normal speed tier
  - STOP: Stop the current drill
  - ASK_RETRY: Prompt user to try again when diagnosis confidence is low
  - NORMAL_CONVERSATION: Conversational fallback
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import GPT_OSS_BASE_URL, GPT_OSS_MODEL, SYSTEM_PROMPT
from session_state import PronunciationSessionState

logger = logging.getLogger("say-that-sound.gpt-oss")

VALID_ACTIONS = {
    "DEMO_PHONEME",
    "DEMO_WORD",
    "REPEAT_DRILL",
    "SLOW_DOWN",
    "NORMAL_SPEED",
    "STOP",
    "ASK_RETRY",
    "NORMAL_CONVERSATION",
}


@dataclass
class DrillAction:
    action: str
    spoken_response: str
    phoneme: Optional[str]
    word: str
    speed_tier: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "spoken_response": self.spoken_response,
            "phoneme": self.phoneme,
            "word": self.word,
            "speed_tier": self.speed_tier,
        }


class GptOssOrchestrator:
    """
    GPT-OSS orchestration engine.
    Produces deterministic, structured drill actions from pronunciation state and user intent.
    """

    def __init__(
        self,
        model: str = GPT_OSS_MODEL,
        base_url: str = GPT_OSS_BASE_URL,
        api_key: str = "",
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.system_prompt = SYSTEM_PROMPT

    def orchestrate_action(
        self,
        user_transcript: str,
        state: PronunciationSessionState,
        diagnosis: Optional[Dict[str, Any]] = None,
    ) -> DrillAction:
        """
        Produce a structured drill action based on the state and user input.
        Guarantees schema compliance and honest confidence handling.
        """
        clean_text = user_transcript.strip().lower()

        # 1. Command handling: "again" / "repeat"
        if clean_text in ("again", "repeat", "one more time", "play again", "say it again", "say that again"):
            cmd = state.handle_again()
            return DrillAction(
                action="REPEAT_DRILL",
                spoken_response=f"Here is the {cmd['phoneme'] or ''} sound in {cmd['word']} again.",
                phoneme=cmd["phoneme"],
                word=cmd["word"],
                speed_tier=cmd["speed_tier"],
            )

        # 2. Command handling: "slower" / "slow down"
        if clean_text in ("slower", "slow down", "go slower", "slower please"):
            cmd = state.handle_slower()
            return DrillAction(
                action="SLOW_DOWN",
                spoken_response=f"Slowing down. Listen carefully to {cmd['word']}.",
                phoneme=cmd["phoneme"],
                word=cmd["word"],
                speed_tier=cmd["new_speed"],
            )

        # 3. Command handling: "normal speed" / "normal"
        if clean_text in ("normal", "normal speed", "normal pace", "regular speed", "default speed"):
            cmd = state.handle_normal_speed()
            return DrillAction(
                action="NORMAL_SPEED",
                spoken_response=f"Back to normal speed. Here is {cmd['word']}.",
                phoneme=cmd["phoneme"],
                word=cmd["word"],
                speed_tier="normal",
            )

        # 4. Command handling: "stop" / "done"
        if clean_text in ("stop", "quit", "end", "done", "that's enough", "i'm done"):
            cmd = state.handle_stop()
            return DrillAction(
                action="STOP",
                spoken_response="Alright, stopping the drill. Say a new word whenever you're ready.",
                phoneme=None,
                word=cmd["word"],
                speed_tier=state.speed_tier,
            )

        # 5. Empty / unintelligible STT fallback
        if not clean_text or len(clean_text) < 2:
            return DrillAction(
                action="ASK_RETRY",
                spoken_response="I didn't catch that. Please try the word again.",
                phoneme=None,
                word=state.current_target_word or "three",
                speed_tier=state.speed_tier,
            )

        # 6. Low confidence diagnosis handling (Honest Diagnosis)
        if diagnosis and diagnosis.get("diagnosis_status") == "low_confidence":
            return DrillAction(
                action="ASK_RETRY",
                spoken_response="I'm not fully confident which sound was off. Let's try that word once more.",
                phoneme=None,
                word=state.current_target_word,
                speed_tier=state.speed_tier,
            )

        # 7. Perfect pronunciation
        if diagnosis and diagnosis.get("diagnosis_status") == "perfect":
            return DrillAction(
                action="NORMAL_CONVERSATION",
                spoken_response=f"Excellent! Your pronunciation of '{state.current_target_word}' sounded spot on.",
                phoneme=None,
                word=state.current_target_word,
                speed_tier=state.speed_tier,
            )

        # 8. Progression feedback — user improved from previous attempt
        if state.has_improved() and diagnosis and diagnosis.get("weak_phoneme"):
            weak_ph = diagnosis["weak_phoneme"]
            word = diagnosis.get("word") or state.current_target_word
            return DrillAction(
                action="DEMO_PHONEME",
                spoken_response=f"That sounded closer! Let's keep working on the {weak_ph} sound.",
                phoneme=weak_ph,
                word=word,
                speed_tier=state.speed_tier,
            )

        # 9. Accepted weak phoneme diagnosis -> Demo isolated phoneme
        if diagnosis and diagnosis.get("weak_phoneme"):
            weak_ph = diagnosis["weak_phoneme"]
            word = diagnosis.get("word") or state.current_target_word
            return DrillAction(
                action="DEMO_PHONEME",
                spoken_response=f"The {weak_ph} sound may be the part to work on. Listen first.",
                phoneme=weak_ph,
                word=word,
                speed_tier=state.speed_tier,
            )

        # 10. Default fallback
        target = state.current_target_word or "three"
        return DrillAction(
            action="DEMO_WORD",
            spoken_response=f"Let's practice pronouncing '{target}'.",
            phoneme=state.weak_phoneme,
            word=target,
            speed_tier=state.speed_tier,
        )

    def validate_action(
        self,
        action: DrillAction,
        state: PronunciationSessionState,
        diagnosis: Optional[Dict[str, Any]] = None,
    ) -> DrillAction:
        """
        Validate an action's fields against current state and diagnosis.
        Enforces that GPT-OSS must NOT determine or alter the mispronounced phoneme.
        """
        # Validate action type
        if action.action not in VALID_ACTIONS:
            logger.warning(f"Invalid action '{action.action}', falling back to DEMO_WORD")
            action.action = "DEMO_WORD"

        # Validate speed tier
        valid_tiers = {"normal", "slow", "slower", "slow_replay"}
        if action.speed_tier not in valid_tiers:
            logger.warning(f"Invalid speed_tier '{action.speed_tier}', defaulting to 'slow'")
            action.speed_tier = state.speed_tier or "slow"

        # Validate word exists
        if not action.word:
            action.word = state.current_target_word or "three"

        # Architectural Rule: Phoneme diagnoses strictly originate from the phoneme analyzer
        if diagnosis:
            diag_phoneme = diagnosis.get("weak_phoneme")
            if diag_phoneme:
                action.phoneme = diag_phoneme
            elif diagnosis.get("diagnosis_status") in ("perfect", "low_confidence", "no_input"):
                action.phoneme = None

        return action

    def parse_llm_json_response(self, response_text: str) -> Optional[DrillAction]:
        """Validate and parse a raw JSON response from a remote GPT-OSS model."""
        try:
            cleaned = response_text.strip()
            # Strip markdown code fences if present
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            try:
                data = json.loads(cleaned)
            except Exception:
                # Try json_repair if available for robust handling of minor LLM syntax glitches
                try:
                    import json_repair
                    data = json_repair.loads(cleaned)
                except Exception:
                    raise

            action = str(data.get("action", "")).upper().strip()
            if action not in VALID_ACTIONS:
                return None
            return DrillAction(
                action=action,
                spoken_response=str(data.get("spoken_response", "")).strip(),
                phoneme=data.get("phoneme"),
                word=str(data.get("word", "")).strip(),
                speed_tier=str(data.get("speed_tier", "slow")).lower().strip(),
            )
        except Exception as e:
            logger.warning(f"Failed to parse GPT-OSS response: {e}")
            return None

    async def orchestrate_action_async(
        self,
        user_transcript: str,
        state: PronunciationSessionState,
        diagnosis: Optional[Dict[str, Any]] = None,
    ) -> DrillAction:
        """
        Asynchronously invoke remote GPT-OSS model for coaching action & phrasing.
        Falls back seamlessly to deterministic orchestrate_action on network or parsing error.
        """
        clean_text = user_transcript.strip().lower()

        # Deterministic command shortcuts bypass LLM to maintain zero-latency state transitions
        if clean_text in (
            "again", "repeat", "one more time", "play again", "say it again", "say that again",
            "slower", "slow down", "go slower", "slower please",
            "normal", "normal speed", "normal pace", "regular speed", "default speed",
            "stop", "quit", "end", "done", "that's enough", "i'm done",
        ):
            return self.orchestrate_action(user_transcript, state, diagnosis)

        # If no API key configured, use deterministic rule engine
        if not self.api_key:
            return self.orchestrate_action(user_transcript, state, diagnosis)

        try:
            import aiohttp

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SayThatSound/1.0",
            }

            diag_info = {
                "target_word": state.current_target_word or diagnosis.get("word", "three") if diagnosis else "three",
                "weak_phoneme": diagnosis.get("weak_phoneme") if diagnosis else state.weak_phoneme,
                "confidence": diagnosis.get("weak_phoneme_confidence", 0.0) if diagnosis else state.diagnosis_confidence,
                "status": diagnosis.get("diagnosis_status", "idle") if diagnosis else state.diagnosis_status,
                "speed_tier": state.speed_tier,
                "user_transcript": user_transcript,
            }

            prompt = (
                f"You are a friendly spoken pronunciation practice coach.\n"
                f"Acoustic phoneme analysis results:\n"
                f"{json.dumps(diag_info, indent=2)}\n\n"
                f"CRITICAL RULES:\n"
                f"1. You MUST NOT invent or alter the weak phoneme diagnosis. Use weak_phoneme={diag_info['weak_phoneme']}.\n"
                f"2. Keep spoken_response to 1-2 concise, encouraging coaching sentences.\n"
                f"3. Valid actions: DEMO_PHONEME, DEMO_WORD, REPEAT_DRILL, SLOW_DOWN, NORMAL_SPEED, STOP, ASK_RETRY, NORMAL_CONVERSATION.\n"
                f"4. If status is 'low_confidence', set action='ASK_RETRY' and ask the user to try the word again.\n"
                f"5. If status is 'perfect', praise the user with action='NORMAL_CONVERSATION'.\n"
                f"6. If weak_phoneme is set, set action='DEMO_PHONEME'.\n\n"
                f"Output ONLY a valid JSON object matching this schema:\n"
                f'{{"action": "...", "spoken_response": "...", "phoneme": "{diag_info["weak_phoneme"] or ""}", "word": "{diag_info["target_word"]}", "speed_tier": "{diag_info["speed_tier"]}"}}'
            )

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 300,
            }

            url = f"{self.base_url.rstrip('/')}/chat/completions"
            timeout = aiohttp.ClientTimeout(total=2.5)

            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        raw_content = data["choices"][0]["message"]["content"]
                        action = self.parse_llm_json_response(raw_content)
                        if action:
                            validated = self.validate_action(action, state, diagnosis)
                            logger.info(f"[GPT-OSS] Live LLM response accepted: action={validated.action}")
                            return validated
                        else:
                            logger.warning(f"[GPT-OSS] Unparseable JSON from LLM: {raw_content[:100]}")
                    else:
                        err_text = await resp.text()
                        logger.warning(f"[GPT-OSS] API returned status {resp.status}: {err_text[:100]}")

        except Exception as e:
            logger.warning(f"[GPT-OSS] LLM invocation failed, using fallback: {e}")

        # Resilient fallback
        return self.orchestrate_action(user_transcript, state, diagnosis)
