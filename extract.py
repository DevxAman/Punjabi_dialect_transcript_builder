"""
Podcast Two-Speaker Conversation Extractor
==========================================
Extracts segments where exactly two people are talking
to each other in a podcast, and stitches them into one WAV.

Usage:
    python extract.py --audio podcast.mp3 --output result.wav
"""

import os
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from pydub import AudioSegment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  |  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Extractor")


# ─────────────────────────────────────────────────────────────
# STEP 1 — VOICE ACTIVITY DETECTION
# Finds where speech actually is (removes silence, music, noise)
# ─────────────────────────────────────────────────────────────
class VAD:
    def __init__(self, sensitivity: int = 2):
        """
        sensitivity: 0 (least aggressive) to 3 (most aggressive).
        2 works well for most podcasts.
        """
        log.info("Loading Silero VAD...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        (
            self.get_speech_timestamps,
            _,
            self.read_audio,
            _,
            _,
        ) = self.utils
        self.sensitivity = sensitivity
        log.info("VAD ready ✓")

    def get_speech_regions(self, audio_path: str):
        """Returns list of (start_sec, end_sec) tuples for all speech."""
        wav = self.read_audio(audio_path, sampling_rate=16000)
        # threshold: higher = stricter (less speech), lower = more permissive
        threshold = 0.3 + (self.sensitivity * 0.1)  # 0.3 to 0.6
        stamps = self.get_speech_timestamps(
            wav,
            self.model,
            threshold=threshold,
            sampling_rate=16000,
            min_speech_duration_ms=500,
            min_silence_duration_ms=300,
        )
        regions = [(t["start"] / 16000, t["end"] / 16000) for t in stamps]
        log.info(f"VAD: found {len(regions)} speech regions")
        return regions


# ─────────────────────────────────────────────────────────────
# STEP 2 — SPEAKER DIARIZATION
# Labels every word with who said it (SPEAKER_00, SPEAKER_01, ...)
# ─────────────────────────────────────────────────────────────
class Diarizer:
    def __init__(self, hf_token: str):
        from pyannote.audio import Pipeline
        log.info("Loading speaker diarization model (downloading ~1GB first time)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {device}")

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        ).to(device)
        log.info("Diarization ready ✓")

    def run(self, audio_path: str) -> list:
        """
        Returns list of dicts:
        [{"speaker": "SPEAKER_00", "start": 1.2, "end": 4.5}, ...]
        """
        log.info("Running diarization (this takes a few minutes)...")
        result = self.pipeline(audio_path)
        turns = []
        for segment, _, speaker in result.itertracks(yield_label=True):
            turns.append({
                "speaker": speaker,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
            })
        log.info(f"Diarization: found {len(turns)} speaker turns "
                 f"from {len(set(t['speaker'] for t in turns))} unique speakers")
        return turns


# ─────────────────────────────────────────────────────────────
# STEP 3 — CONVERSATION DETECTOR
# Finds windows where exactly 2 speakers are going back and forth
# ─────────────────────────────────────────────────────────────
class ConversationDetector:
    def __init__(
        self,
        max_gap_sec: float = 5.0,
        min_exchanges: int = 3,
        min_block_duration: float = 10.0,
        max_speakers: int = 2,
    ):
        """
        max_gap_sec       : max silence between turns to stay in same conversation block
        min_exchanges     : minimum speaker alternations (A→B→A counts as 2)
        min_block_duration: discard blocks shorter than this (seconds)
        max_speakers      : only keep blocks with exactly this many speakers (2 = dialogue)
        """
        self.max_gap_sec = max_gap_sec
        self.min_exchanges = min_exchanges
        self.min_block_duration = min_block_duration
        self.max_speakers = max_speakers

    def extract(self, turns: list) -> list:
        """
        Groups turns into conversation blocks.
        Returns list of blocks, each block is a list of turns.
        Only returns blocks with exactly 2 speakers and enough back-and-forth.
        """
        if not turns:
            return []

        blocks = []
        current = [turns[0]]

        for i in range(1, len(turns)):
            prev = turns[i - 1]
            curr = turns[i]
            gap = curr["start"] - prev["end"]

            if gap <= self.max_gap_sec:
                current.append(curr)
            else:
                block = self._evaluate(current)
                if block:
                    blocks.append(block)
                current = [curr]

        # Last block
        block = self._evaluate(current)
        if block:
            blocks.append(block)

        log.info(f"Conversation detector: {len(blocks)} valid dialogue blocks found")
        return blocks

    def _evaluate(self, turns: list):
        """Returns block dict if it passes all filters, else None."""
        if not turns:
            return None

        all_speakers = [t["speaker"] for t in turns]
        speaker_counts = {}
        for s in all_speakers:
            speaker_counts[s] = speaker_counts.get(s, 0) + 1
        top2 = sorted(speaker_counts, key=lambda x: -speaker_counts[x])[:2]
        speakers = set(top2)
        turns = [t for t in turns if t["speaker"] in speakers]
        if len(speakers) < 2:
            return None

        # Must have enough back-and-forth exchanges
        exchanges = sum(
            1 for i in range(1, len(turns))
            if turns[i]["speaker"] != turns[i - 1]["speaker"]
        )
        if exchanges < self.min_exchanges:
            return None

        start = turns[0]["start"]
        end = turns[-1]["end"]
        duration = end - start

        # Must be long enough to be meaningful
        if duration < self.min_block_duration:
            return None

        return {
            "start": start,
            "end": end,
            "duration": duration,
            "speakers": list(speakers),
            "exchanges": exchanges,
            "turns": turns,
        }


# ─────────────────────────────────────────────────────────────
# STEP 4 — AUDIO STITCHER
# Cuts the valid blocks and joins them into one WAV
# ─────────────────────────────────────────────────────────────
class Stitcher:
    def __init__(self, crossfade_ms: int = 150, padding_ms: int = 100):
        """
        crossfade_ms : smooth blend between segments (avoids clicks)
        padding_ms   : add a tiny bit of silence before/after each segment
        """
        self.crossfade_ms = crossfade_ms
        self.padding_ms = padding_ms

    def stitch(self, audio_path: str, blocks: list, output_path: str) -> str:
        """
        Cuts the conversation blocks from the source audio
        and stitches them into a single output WAV file.
        """
        log.info(f"Loading source audio: {audio_path}")
        source = AudioSegment.from_file(audio_path)
        silence = AudioSegment.silent(duration=self.padding_ms)

        segments = []
        for i, block in enumerate(blocks):
            start_ms = max(0, int(block["start"] * 1000) - self.padding_ms)
            end_ms = min(len(source), int(block["end"] * 1000) + self.padding_ms)
            chunk = source[start_ms:end_ms]
            segments.append(chunk)
            log.info(
                f"  Block {i+1:02d}: {block['start']:.1f}s → {block['end']:.1f}s  "
                f"({block['duration']:.1f}s, {block['exchanges']} exchanges, "
                f"speakers: {', '.join(block['speakers'])})"
            )

        if not segments:
            log.warning("No segments to stitch.")
            return None

        log.info(f"Stitching {len(segments)} blocks with {self.crossfade_ms}ms crossfade...")
        combined = silence
        for seg in segments:
            crossfade = min(self.crossfade_ms, len(combined), len(seg))
            combined = combined.append(seg, crossfade=crossfade)
        combined = combined.append(silence)

        # Export as WAV, preserve original sample rate
        combined.export(output_path, format="wav")
        duration_min = len(combined) / 1000 / 60
        log.info(f"Output saved → {output_path}  ({duration_min:.2f} minutes)")
        return output_path


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def run(args):
    log.info("=" * 55)
    log.info("  Podcast Two-Speaker Conversation Extractor")
    log.info("=" * 55)

    audio_path = args.audio
    output_path = args.output

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Step 1: VAD (optional but speeds up diarization)
    # Diarization handles this internally too, so VAD is mainly
    # used here to give you a quick preview of speech density.
    log.info("\n── Step 1: Voice Activity Detection ──")
    vad = VAD(sensitivity=args.vad_sensitivity)
    speech_regions = vad.get_speech_regions(audio_path)
    total_speech = sum(e - s for s, e in speech_regions)
    log.info(f"Total speech: {total_speech/60:.1f} minutes")

    # Step 2: Diarization — who speaks when
    log.info("\n── Step 2: Speaker Diarization ──")
    diarizer = Diarizer(hf_token=args.hf_token)
    turns = diarizer.run(audio_path)

    # Step 3: Find two-person conversation blocks
    log.info("\n── Step 3: Detecting Two-Speaker Conversations ──")
    detector = ConversationDetector(
        max_gap_sec=args.max_gap,
        min_exchanges=args.min_exchanges,
        min_block_duration=args.min_duration,
        max_speakers=2,
    )
    blocks = detector.extract(turns)

    if not blocks:
        log.warning(
            "No two-speaker conversation blocks found.\n"
            "Try: --max_gap 8.0 --min_exchanges 2 --min_duration 5.0"
        )
        return

    total_conv = sum(b["duration"] for b in blocks)
    log.info(f"\nFound {len(blocks)} blocks  |  {total_conv/60:.1f} minutes of dialogue")

    # Step 4: Cut and stitch
    log.info("\n── Step 4: Extracting & Stitching ──")
    stitcher = Stitcher(crossfade_ms=args.crossfade)
    stitcher.stitch(audio_path, blocks, output_path)

    log.info("\n✓ Done!")
    log.info(f"  Input  : {audio_path}")
    log.info(f"  Output : {output_path}")
    log.info(f"  Blocks : {len(blocks)}")
    log.info(f"  Length : {total_conv/60:.1f} minutes of dialogue")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract two-person conversations from a podcast"
    )

    parser.add_argument(
        "--audio", required=True,
        help="Path to input podcast file (MP3, WAV, M4A, etc.)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for output WAV file"
    )
    parser.add_argument(
        "--hf_token", required=True,
        help="Your HuggingFace access token (for pyannote diarization model)"
    )

    # Tuning options
    parser.add_argument(
        "--max_gap", type=float, default=5.0,
        help="Max silence gap (seconds) between turns to stay in same block (default: 5.0)"
    )
    parser.add_argument(
        "--min_exchanges", type=int, default=3,
        help="Minimum speaker alternations for a block to count as dialogue (default: 3)"
    )
    parser.add_argument(
        "--min_duration", type=float, default=10.0,
        help="Minimum block duration in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--crossfade", type=int, default=150,
        help="Crossfade milliseconds between stitched segments (default: 150)"
    )
    parser.add_argument(
        "--vad_sensitivity", type=int, default=2, choices=[0, 1, 2, 3],
        help="VAD aggressiveness 0-3 (default: 2)"
    )

    args = parser.parse_args()
    run(args)