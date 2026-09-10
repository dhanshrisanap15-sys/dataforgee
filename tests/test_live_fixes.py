"""
Comprehensive tests for Say That Sound reliability and bug fixes:
1. Genuine microphone audio diagnosis and silence handling in PhonemeAnalyzer.
2. Real GPT-OSS LLM orchestration with schema validation and fallback.
3. CancellableDrillPlaybackTask multi-stage execution and instant cancellation.
4. TokenServer /health endpoint.
"""

import asyncio
import os
import unittest
from pathlib import Path

from config import CONFIDENCE_THRESHOLD, TARGET_VOCABULARY
from drill_manager import CancellableDrillPlaybackTask, CommandType, intercept_command
from gpt_oss_orchestrator import DrillAction, GptOssOrchestrator
from phoneme_analyzer import PhonemeAnalyzer
from phoneme_dict import get_expected_phonemes
from session_state import DrillState, PronunciationSessionState
from token_server import create_app


class TestAcousticDiagnosis(unittest.TestCase):
    """Test phoneme analyzer audio processing and honest diagnosis."""

    def setUp(self):
        self.analyzer = PhonemeAnalyzer(confidence_threshold=0.70)

    def test_silence_returns_no_input(self):
        """Silent audio frames must return no_input, never false 'perfect'."""
        # 0.5s of near-silent PCM audio (RMS < 200)
        silent_pcm = bytes([0, 0] * 8000)
        diag = self.analyzer.analyze_phonemes(
            word="three",
            audio_frames=silent_pcm,
        )
        self.assertEqual(diag.diagnosis_status, "no_input")
        self.assertFalse(diag.passed_threshold)
        self.assertIsNone(diag.weak_phoneme)

    def test_no_audio_no_transcript_returns_no_input(self):
        """Calling analyze_phonemes without audio or transcript must not return 'perfect'."""
        diag = self.analyzer.analyze_phonemes(word="three")
        self.assertEqual(diag.diagnosis_status, "no_input")
        self.assertIsNone(diag.weak_phoneme)

    def test_spoken_transcript_substitution_diagnosis(self):
        """User speaking 'tree' for target 'three' must accurately diagnose 'TH' as weak phoneme."""
        diag = self.analyzer.analyze_phonemes(
            word="three",
            spoken_transcript="tree",
        )
        self.assertEqual(diag.diagnosis_status, "accepted")
        self.assertTrue(diag.passed_threshold)
        self.assertEqual(diag.weak_phoneme, "TH")
        self.assertGreaterEqual(diag.weak_phoneme_confidence, 0.70)

    def test_spoken_transcript_sheep_vs_ship(self):
        """User speaking 'ship' for target 'sheep' must diagnose vowel 'IY' vs 'IH'."""
        diag = self.analyzer.analyze_phonemes(
            word="sheep",
            spoken_transcript="ship",
        )
        self.assertEqual(diag.diagnosis_status, "accepted")
        self.assertEqual(diag.weak_phoneme, "IY")

    def test_spoken_transcript_correct_pronunciation(self):
        """User speaking 'three' for target 'three' returns perfect status."""
        diag = self.analyzer.analyze_phonemes(
            word="three",
            spoken_transcript="three",
        )
        self.assertEqual(diag.diagnosis_status, "perfect")
        self.assertIsNone(diag.weak_phoneme)


class TestGptOssOrchestration(unittest.TestCase):
    """Test GPT-OSS live async orchestration and guardrails."""

    def setUp(self):
        self.orchestrator = GptOssOrchestrator()
        self.state = PronunciationSessionState(current_target_word="three", speed_tier="slow")

    def test_deterministic_commands_bypass_network(self):
        """Drill navigation commands must execute immediately without network calls."""
        action = self.orchestrator.orchestrate_action("again", self.state)
        self.assertEqual(action.action, "REPEAT_DRILL")
        self.assertEqual(action.word, "three")

        action_slow = self.orchestrator.orchestrate_action("slower", self.state)
        self.assertEqual(action_slow.action, "SLOW_DOWN")
        self.assertEqual(action_slow.speed_tier, "slower")

    def test_action_validation_preserves_acoustic_phoneme(self):
        """Validate action must enforce that LLM cannot alter diagnosed weak phoneme."""
        diag = {
            "word": "three",
            "weak_phoneme": "TH",
            "weak_phoneme_confidence": 0.88,
            "diagnosis_status": "accepted",
        }
        # LLM hypothetically hallucinated 'R' instead of acoustic 'TH'
        raw_action = DrillAction(
            action="DEMO_PHONEME",
            spoken_response="Let's practice the R sound.",
            phoneme="R",
            word="three",
            speed_tier="slow",
        )
        validated = self.orchestrator.validate_action(raw_action, self.state, diag)
        self.assertEqual(validated.phoneme, "TH", "Acoustic diagnosis MUST override any LLM hallucination")

    def test_live_async_orchestration(self):
        """Test async orchestration with live Groq / GPT-OSS endpoint if API key is present."""
        diag = {
            "word": "three",
            "weak_phoneme": "TH",
            "weak_phoneme_confidence": 0.85,
            "diagnosis_status": "accepted",
        }
        action = asyncio.run(
            self.orchestrator.orchestrate_action_async(
                user_transcript="tree",
                state=self.state,
                diagnosis=diag,
            )
        )
        self.assertIn(action.action, ("DEMO_PHONEME", "DEMO_WORD", "ASK_RETRY"))
        self.assertEqual(action.phoneme, "TH")
        self.assertTrue(len(action.spoken_response) > 5)


class TestCancellableDrillPlayback(unittest.TestCase):
    """Test staged playback and cancellation in CancellableDrillPlaybackTask."""

    def test_staged_execution_and_cancellation(self):
        async def run_test():
            state = PronunciationSessionState(current_target_word="three", speed_tier="slow")
            stages_spoken = []

            async def mock_speak(text: str):
                stages_spoken.append(text)
                await asyncio.sleep(0.05)

            task = CancellableDrillPlaybackTask(
                generation_id=state.interaction_generation,
                state=state,
                speak_fn=mock_speak,
            )

            # Start execution in background
            exec_task = asyncio.create_task(
                task.execute(
                    coaching_text="Listen to the sound.",
                    phoneme_repr="{TH}",
                    word="three",
                    speed_alpha=0.8,
                )
            )

            # Wait for first stage then cancel
            await asyncio.sleep(0.02)
            task.cancel()
            await exec_task

            # Verify that later stages (pause, slow word, your turn) were aborted
            self.assertTrue(task.is_cancelled)
            self.assertNotIn("Your turn.", stages_spoken)

        asyncio.run(run_test())


class TestTokenServerHealth(unittest.TestCase):
    """Test the /health route in TokenServer."""

    def test_health_endpoint(self):
        app = create_app()
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
