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
from collections import Counter

import torch
import numpy as np
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
        max_gap_sec: float = 2.5,
        min_exchanges: int = 3,
        min_block_duration: float = 5.0,
        window_size_sec: float = 8.0,
        window_hop_sec: float = 4.0,
    ):
        self.max_gap_sec = max_gap_sec
        self.min_exchanges = min_exchanges
        self.min_block_duration = min_block_duration
        self.window_size_sec = window_size_sec
        self.window_hop_sec = window_hop_sec

    def extract(self, turns: list) -> list:
        if not turns:
            return []

        turns = sorted(turns, key=lambda x: x["start"])
        accepted = []
        audio_end = max(t["end"] for t in turns)
        win_start = turns[0]["start"]

        while win_start < audio_end:
            win_end = win_start + self.window_size_sec
            window_turns = self._slice_window(turns, win_start, win_end)
            block = self._validate_window(window_turns, win_start, win_end)
            if block:
                accepted.append(block)
            win_start += self.window_hop_sec

        merged = self._merge_adjacent_blocks(accepted)
        log.info(f"Conversation detector: {len(merged)} valid dialogue blocks found")
        return merged

    @staticmethod
    def _slice_window(turns: list, win_start: float, win_end: float) -> list:
        window_turns = []
        for t in turns:
            if t["end"] <= win_start or t["start"] >= win_end:
                continue
            clipped = {
                "speaker": t["speaker"],
                "start": max(t["start"], win_start),
                "end": min(t["end"], win_end),
            }
            if clipped["end"] > clipped["start"]:
                window_turns.append(clipped)
        return sorted(window_turns, key=lambda x: x["start"])

    def _validate_window(self, turns: list, win_start: float, win_end: float):
        if not turns:
            return None

        duration = turns[-1]["end"] - turns[0]["start"]
        if duration < self.min_block_duration:
            log.info(f"Rejected: too short [{win_start:.2f}, {win_end:.2f}]")
            return None

        speakers = {t["speaker"] for t in turns}
        if len(speakers) > 2:
            log.info(f"Rejected: >2 speakers [{win_start:.2f}, {win_end:.2f}]")
            return None
        if len(speakers) < 2:
            log.info(f"Rejected: <2 speakers [{win_start:.2f}, {win_end:.2f}]")
            return None

        exchanges = sum(
            1 for i in range(1, len(turns))
            if turns[i]["speaker"] != turns[i - 1]["speaker"]
        )
        if exchanges < self.min_exchanges:
            log.info(f"Rejected: weak alternation [{win_start:.2f}, {win_end:.2f}]")
            return None

        speaker_time = Counter()
        for t in turns:
            speaker_time[t["speaker"]] += (t["end"] - t["start"])
        times = list(speaker_time.values())
        if min(times) / max(times) < 0.3:
            log.info(f"Rejected: speaker imbalance [{win_start:.2f}, {win_end:.2f}]")
            return None

        for i in range(1, len(turns)):
            gap = turns[i]["start"] - turns[i - 1]["end"]
            if gap > self.max_gap_sec:
                log.info(f"Rejected: excessive gap [{win_start:.2f}, {win_end:.2f}]")
                return None

        return {
            "start": turns[0]["start"],
            "end": turns[-1]["end"],
            "duration": duration,
            "speakers": sorted(list(speakers)),
            "exchanges": exchanges,
            "turns": turns,
        }

    def _merge_adjacent_blocks(self, blocks: list) -> list:
        if not blocks:
            return []
        blocks = sorted(blocks, key=lambda x: x["start"])
        merged = [blocks[0]]
        for block in blocks[1:]:
            prev = merged[-1]
            if block["start"] <= prev["end"]:
                prev["end"] = max(prev["end"], block["end"])
                prev["duration"] = prev["end"] - prev["start"]
                prev["turns"].extend(block["turns"])
                prev["turns"] = sorted(prev["turns"], key=lambda x: x["start"])
                prev["speakers"] = sorted(list({t["speaker"] for t in prev["turns"]}))
                prev["exchanges"] = sum(
                    1 for i in range(1, len(prev["turns"]))
                    if prev["turns"][i]["speaker"] != prev["turns"][i - 1]["speaker"]
                )
            else:
                merged.append(block)
        return merged


class PunjabiTranscriber:
    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo"):
        from transformers import pipeline
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device=0 if torch.cuda.is_available() else -1,
        )

    def transcribe_blocks(self, audio_path: str, blocks: list, output_path: str):
        source = AudioSegment.from_file(audio_path)
        lines = []
        for i, block in enumerate(blocks, start=1):
            speaker_map = {
                spk: ("Speaker A" if idx == 0 else "Speaker B")
                for idx, spk in enumerate(block["speakers"])
            }
            lines.append(
                f"[Segment {i:02d}] {block['start']:.2f}s -> {block['end']:.2f}s"
            )
            for turn in block["turns"]:
                speaker = turn["speaker"]
                if speaker not in speaker_map:
                    continue
                start_ms = int(turn["start"] * 1000)
                end_ms = int(turn["end"] * 1000)
                chunk = source[start_ms:end_ms]
                if len(chunk) < 300:
                    continue
                arr = np.array(chunk.get_array_of_samples()).astype(np.float32)
                if chunk.channels > 1:
                    arr = arr.reshape((-1, chunk.channels)).mean(axis=1)
                arr = arr / max(np.iinfo(chunk.array_type).max, 1)
                result = self.pipe(
                    {"array": arr, "sampling_rate": chunk.frame_rate},
                    generate_kwargs={"language": "pa", "task": "transcribe"},
                )
                text = result["text"].strip()
                if text:
                    lines.append(f"{speaker_map[speaker]}: {text}")
            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).strip() + "\n")
        log.info(f"Transcript saved → {output_path}")


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
        window_size_sec=args.window_size,
        window_hop_sec=args.window_hop,
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

    # Step 5: Punjabi transcription (Gurmukhi)
    log.info("\n── Step 5: Punjabi (Puadh) Speaker-wise Transcription ──")
    transcriber = PunjabiTranscriber(model_id=args.asr_model)
    transcriber.transcribe_blocks(audio_path, blocks, args.transcript_output)

    log.info("\n✓ Done!")
    log.info(f"  Input  : {audio_path}")
    log.info(f"  Output : {output_path}")
    log.info(f"  Text   : {args.transcript_output}")
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
        "--max_gap", type=float, default=2.5,
        help="Max silence gap allowed inside a valid window (default: 2.5)"
    )
    parser.add_argument(
        "--min_exchanges", type=int, default=3,
        help="Minimum speaker alternations for a block to count as dialogue (default: 3)"
    )
    parser.add_argument(
        "--min_duration", type=float, default=5.0,
        help="Minimum accepted window duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--window_size", type=float, default=8.0,
        help="Sliding window size in seconds (default: 8.0)"
    )
    parser.add_argument(
        "--window_hop", type=float, default=4.0,
        help="Sliding window hop in seconds (default: 4.0)"
    )
    parser.add_argument(
        "--crossfade", type=int, default=150,
        help="Crossfade milliseconds between stitched segments (default: 150)"
    )
    parser.add_argument(
        "--vad_sensitivity", type=int, default=2, choices=[0, 1, 2, 3],
        help="VAD aggressiveness 0-3 (default: 2)"
    )
    parser.add_argument(
        "--transcript_output", default="conversation_punjabi.txt",
        help="Path for speaker-wise Punjabi transcript output text"
    )
    parser.add_argument(
        "--asr_model", default="openai/whisper-large-v3-turbo",
        help="ASR model id for Punjabi transcription"
    )

    args = parser.parse_args()
    run(args)
