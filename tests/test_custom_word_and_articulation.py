"""
Unit tests for Custom Word Lookup, Articulation Guidance, and Minimal Pairs Tracks.
"""

import asyncio
import unittest
from aiohttp.test_utils import TestClient, TestServer

from agent import extract_target_word
from phoneme_dict import (
    MINIMAL_PAIRS_TRACKS,
    get_expected_phonemes,
    get_ipa_transcription,
    get_phoneme_articulation,
    get_word_phoneme_breakdown,
    has_pronunciation,
)
from phoneme_analyzer import PhonemeAlignmentItem, PronunciationDiagnosis
from session_state import DrillState, PronunciationSessionState
from token_server import create_app


class TestCustomWordAndArticulation(unittest.TestCase):
    """Test arbitrary custom word lookup, IPA generation, and speech articulation guides."""

    def test_custom_word_cmudict_lookup(self):
        squirrel_data = get_word_phoneme_breakdown("squirrel")
        self.assertEqual(squirrel_data["word"], "squirrel")
        self.assertTrue(squirrel_data["in_dict"])
        self.assertEqual(squirrel_data["phonemes"], ["S", "K", "W", "ER", "AH", "L"])
        self.assertIn("s", squirrel_data["ipa"])
        self.assertEqual(len(squirrel_data["breakdown"]), 6)

        rhythm_data = get_word_phoneme_breakdown("rhythm")
        self.assertEqual(rhythm_data["word"], "rhythm")
        self.assertTrue(rhythm_data["in_dict"])
        self.assertEqual(rhythm_data["phonemes"], ["R", "IH", "DH", "AH", "M"])
        self.assertIn("ð", rhythm_data["ipa"])

    def test_articulation_guides_completeness(self):
        key_phonemes = ["TH", "DH", "R", "L", "SH", "S", "CH", "JH", "V", "F", "W", "T", "D", "ER", "IY", "IH"]
        for ph in key_phonemes:
            guide = get_phoneme_articulation(ph)
            self.assertIn("tongue", guide, f"Missing tongue guidance for {ph}")
            self.assertIn("lips", guide, f"Missing lip guidance for {ph}")
            self.assertIn("airflow", guide, f"Missing airflow guidance for {ph}")
            self.assertIn("voicing", guide, f"Missing voicing guidance for {ph}")
            self.assertIn("ipa", guide, f"Missing IPA for {ph}")
            self.assertTrue(len(guide["tongue"]) > 10, f"Tongue guidance too brief for {ph}")

    def test_minimal_pairs_tracks(self):
        track_ids = [t["id"] for t in MINIMAL_PAIRS_TRACKS]
        self.assertIn("th-vs-t", track_ids)
        self.assertIn("r-vs-l", track_ids)
        self.assertIn("sh-vs-s", track_ids)
        self.assertIn("v-vs-w", track_ids)
        for t in MINIMAL_PAIRS_TRACKS:
            self.assertTrue(len(t["words"]) >= 4)
            self.assertTrue(len(t["focus_phonemes"]) >= 2)

    def test_session_state_stores_alignment(self):
        state = PronunciationSessionState(current_target_word="three")
        self.assertEqual(state.alignment, [])

        items = [
            PhonemeAlignmentItem(expected="TH", observed="T", confidence=0.72, is_match=False),
            PhonemeAlignmentItem(expected="R", observed="R", confidence=0.95, is_match=True),
            PhonemeAlignmentItem(expected="IY", observed="IY", confidence=0.98, is_match=True),
        ]
        diag = PronunciationDiagnosis(
            word="three",
            expected_phonemes=["TH", "R", "IY"],
            observed_phonemes=["T", "R", "IY"],
            alignment=items,
            weak_phoneme="TH",
            weak_phoneme_confidence=0.72,
            diagnosis_status="accepted",
            passed_threshold=True,
            weak_phoneme_detail="Observed T instead of TH",
        )

        state.update_diagnosis(diag)
        self.assertEqual(len(state.alignment), 3)
        self.assertEqual(state.alignment[0]["expected"], "TH")
        self.assertEqual(state.alignment[0]["observed"], "T")
        self.assertFalse(state.alignment[0]["is_match"])
        self.assertTrue(state.alignment[1]["is_match"])

    def test_extract_target_word_single_custom_word(self):
        vocab = ["three", "think", "this"]
        self.assertEqual(extract_target_word("squirrel", vocab), "squirrel")
        self.assertEqual(extract_target_word("rhythm", vocab), "rhythm")
        self.assertEqual(extract_target_word("practice squirrel", vocab), "squirrel")
        self.assertEqual(extract_target_word("say three", vocab), "three")
        self.assertEqual(extract_target_word("I want to practice three today", vocab), "three")


class TestTokenServerEndpoints(unittest.TestCase):
    def test_api_phonemes_and_tracks(self):
        async def run():
            app = create_app()
            client = TestClient(TestServer(app))
            await client.start_server()

            res = await client.get("/api/phonemes?word=worcestershire")
            self.assertEqual(res.status, 200)
            data = await res.json()
            self.assertEqual(data["word"], "worcestershire")
            self.assertTrue(len(data["phonemes"]) > 0)
            self.assertTrue(len(data["breakdown"]) > 0)

            res_bad = await client.get("/api/phonemes")
            self.assertEqual(res_bad.status, 400)

            res_tracks = await client.get("/api/tracks")
            self.assertEqual(res_tracks.status, 200)
            tracks_json = await res_tracks.json()
            self.assertEqual(len(tracks_json["tracks"]), 4)

            await client.close()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
