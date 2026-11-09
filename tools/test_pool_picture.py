"""The picture case of the pool's own tools: the ids a picture request carries, the `--vision` switch, the
embeddings hand-off (GEN vs GENI), and the data: URL the bench sends a picture in.

Nothing here needs a GPU, a model or the HTTP server's dependencies: the helpers are the parts that can be wrong
quietly.  The ids one is the one that matters for a pool - a picture whose `<|image_pad|>` count does not match the
embeddings file is a request the engine refuses or, worse, positions that do not match the rows (docs/POOL_VISION.md).

    python -m unittest tools.test_pool_picture
"""
from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import pool_bench as PB        # noqa: E402
import pool_selftest as PS     # noqa: E402

PAD, START = 11, 10


class FakeTok:
    """A tokenizer that knows only the chat's markers: each keeps its own id, every other character becomes
    9000+its code.  Enough to see which markers survive and how often the picture marker is repeated."""

    MARKS = {"<|im_start|>": 1, "<|im_end|>": 2, "<|vision_start|>": START, "<|image_pad|>": PAD,
             "<|vision_end|>": 12}

    def encode(self, text, parse_special=True):
        out, i = [], 0
        while i < len(text):
            hit = next((m for m in self.MARKS if text.startswith(m, i)), None)
            if hit is None:
                out.append(9000 + ord(text[i]))
                i += 1
            else:
                out.append(self.MARKS[hit])
                i += len(hit)
        return out


class PictureIds(unittest.TestCase):
    def test_the_picture_marker_grows_to_the_image_tokens(self):
        ids = PS.picture_ids(FakeTok(), "What is this?", 7, PAD, START)
        self.assertEqual(ids.count(PAD), 7, "the one marker the template writes must become the reader's 7 tokens")
        s = ids.index(START)
        self.assertEqual(ids[s + 1:s + 8], [PAD] * 7, "the image tokens follow <|vision_start|> directly")
        self.assertEqual(ids[s + 8], 12, "<|vision_end|> still closes the picture")

    def test_a_marker_in_the_question_stays_one_token(self):
        # the same rule the server applies (#150): only a pad right after a vision_start is a picture, or text that
        # quotes these markers would take a picture's place and the counts would no longer match
        ids = PS.picture_ids(FakeTok(), "the text says <|image_pad|> here", 4, PAD, START)
        self.assertEqual(ids.count(PAD), 5, "4 image tokens plus the quoted marker, not 4 + 4")

    def test_a_prompt_without_the_markers_is_refused(self):
        with self.assertRaises(SystemExit):
            PS.picture_ids(FakeTok(), "plain text", 4, 999, START)


class VisionSwitch(unittest.TestCase):
    def test_added_once_and_left_alone_otherwise(self):
        self.assertEqual(PS.with_vision(["--model", "m"], True), ["--model", "m", "--vision"])
        self.assertEqual(PS.with_vision(["--model", "m", "--vision"], True), ["--model", "m", "--vision"])
        self.assertEqual(PS.with_vision(["--model", "m"], False), ["--model", "m"])


class EmbeddingsReachTheEngine(unittest.TestCase):
    def test_text_sends_none_and_a_picture_its_file(self):
        got = []

        class Eng:
            last = {}

            def generate(self, ids, max_new, sampling, cancel, embeddings=None, prefill_only=False):
                got.append((len(ids), embeddings))
                yield 5
                yield 6

        out = PS.run_prompts(Eng(), [([1, 2, 3], None), ([4] * 9, "/tmp/x.sve")], 32, lambda m: None)
        self.assertEqual(got, [(3, None), (9, "/tmp/x.sve")])
        self.assertEqual([r["tokens"] for r in out], [[5, 6], [5, 6]])


class BenchPicture(unittest.TestCase):
    def test_the_bench_reads_the_file_itself_and_names_its_kind(self):
        with tempfile.TemporaryDirectory() as d:
            png = Path(d) / "shot.png"
            png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 24)
            url = PB.data_url(png)
            self.assertTrue(url.startswith("data:image/png;base64,"), url[:40])
            self.assertEqual(base64.b64decode(url.split(",", 1)[1]), png.read_bytes())
            unknown = Path(d) / "shot.heic"
            unknown.write_bytes(b"fake")
            self.assertTrue(PB.data_url(unknown).startswith("data:image/png;base64,"),
                            "an unknown suffix still travels as a picture; the reader says what it makes of it")

    def test_jpeg_and_webp_get_their_own_media_type(self):
        with tempfile.TemporaryDirectory() as d:
            for name, kind in (("a.jpg", "image/jpeg"), ("b.webp", "image/webp"), ("c.gif", "image/gif")):
                p = Path(d) / name
                p.write_bytes(b"z")
                self.assertTrue(PB.data_url(p).startswith(f"data:{kind};base64,"), name)


if __name__ == "__main__":
    unittest.main()
