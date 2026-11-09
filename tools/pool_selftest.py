"""The pool's self-test on ONE PC: the model alone, then the same model split between a coordinator and a worker
on this same GPU (over 127.0.0.1), on the same prompts, greedy.  It shows that the pool runs end to end (handshake,
split, prompt chunks and verify windows over the network, the head and the drafter on the coordinator) and how its
answers and speed compare.

    .venv\\Scripts\\python tools\\pool_selftest.py strata-coder-iq1_m.json
    .venv\\Scripts\\python tools\\pool_selftest.py strata-coder-iq1_m.json --split 24 --ctx 8192 --cache 500
    .venv\\Scripts\\python tools\\pool_selftest.py strata-coder-iq1_m.json --skip-single --several-chats --overlap

--several-chats and --overlap start the coordinator as the Pool tab's switches do (batch slots, pipelined windows; with
both, STRATA_PW_WITH_BATCH=1 as serve/pool.py sets it), and --several-chats also reads two prompts at once.

--picture PATH adds a fourth request that carries an image, so the split's picture path is checked too: the file is
encoded once, before any engine starts (the reader and an engine must not share the GPU), and the same embeddings file
goes to both runs.  What it proves is the M-RoPE cell table crossing the split the same way (docs/POOL_VISION.md): a
worker without it answers something plausible from the first token, so "no token in common with the model alone" is
the failure it is looking for.  The model must be set up with the picture reader.

Both engines of the pair run on this GPU, so each gets a small expert cache (--cache, slots per engine; the model alone
gets twice that) and a short context.  Close Strata first: the GPU must be free.

The answers need not match token for token: an expert the GPU computes rounds differently from one the CPU computes,
and the pair caches other experts than the single engine (bench/results/2026-09-27-cache-parity).  What must hold:
the pool answers, its answers make sense, and most tokens agree.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

PROMPTS = (
    "What is the capital of France? Answer in one word.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "List the planets of the solar system in order from the sun, comma-separated.",
)


def chat_ids(tok, text: str) -> list[int]:
    return tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      parse_special=True)


def picture_ids(tok, question: str, n_img: int, pad: int, start: int) -> list[int]:
    """The ids of one picture request: what chat_ids() writes, with the picture's single `<|image_pad|>` grown to the
    `n_img` tokens the reader said the image became.  That is the rule serve/server.py applies to a request with an
    image (only a pad right after `<|vision_start|>` is a picture; the same text in a message stays text), kept here
    so this tool sends what the server would send.  `pad` and `start` are those markers' token ids."""
    ids = chat_ids(tok, f"<|vision_start|><|image_pad|><|vision_end|>{question}")
    out, seen = [], 0
    for j, t in enumerate(ids):
        if t == pad and j > 0 and ids[j - 1] == start:
            out += [pad] * n_img
            seen += 1
        else:
            out.append(t)
    if seen != 1:
        raise SystemExit(f"the picture markers did not survive the tokenizer ({seen} of them found)")
    return out


def with_vision(args: list[str], need: bool) -> list[str]:
    """`--vision` for the engines of a picture run - a switch, it takes no value.  A worker is not given it: a pool's
    worker takes pictures from the coordinator's CONFIG (`vision=1`), not from its own command line."""
    return args + ["--vision"] if need and "--vision" not in args else args


def with_arg(args: list[str], flag: str, value: str | None) -> list[str]:
    out = list(args)
    while flag in out:
        i = out.index(flag)
        del out[i:i + 2]
    if value is not None:
        out += [flag, value]
    return out


def base_args(cfg: dict, ctx: int, cache: int, chunk: int) -> list[str]:
    a = list(cfg["args"])
    for flag in ("--conversation-cache-mib", "--conversation-cache-slots", "--conversation-cache-min-free-mib",
                 "--kv-resident", "--layer-split", "--split-device"):
        a = with_arg(a, flag, None)
    for flag in ("--pool-peers", "--pool-split", "--pool-listen", "--pool-secret", "--pool-wire", "--pool-draft-wire",
                 "--pool-timeout-s", "--pool-wait-s"):   # the layer split's (not --pool-workers / --pool-affinity)
        a = with_arg(a, flag, None)
    a = with_arg(a, "--max-context", str(ctx))
    a = with_arg(a, "--expert-cache", str(cache))
    a = with_arg(a, "--prefill", str(chunk))
    return a


def run_prompts(eng, jobs, max_new: int, say) -> list[dict]:
    """One request per (ids, embeddings-file) job; the embeddings are None for a text prompt (`GEN`, not `GENI`)."""
    out = []
    for ids, emb in jobs:
        cancel = threading.Event()
        t0 = time.time()
        toks = [t for t in eng.generate(ids, max_new, {}, cancel, embeddings=emb) if t is not None]
        out.append({"tokens": toks, "s": time.time() - t0, **(eng.last or {})})
        say(f"    {len(toks)} tokens in {out[-1]['s']:.1f} s" + ("" if emb is None else " (with a picture)"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="the model's run config (strata-<model>.json)")
    ap.add_argument("--split", default="24", help="the worker's first layer (default 24)")
    ap.add_argument("--ctx", type=int, default=8192, help="context for this test (default 8192)")
    ap.add_argument("--cache", type=int, default=400, help="expert cache slots per engine of the pair (default 400)")
    ap.add_argument("--chunk", type=int, default=1024, help="prompt chunk (default 1024)")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--port", type=int, default=7799)
    ap.add_argument("--wire", default="f32", choices=["f32", "f16", "bf16"])
    ap.add_argument("--skip-single", action="store_true", help="only the pool run")
    ap.add_argument("--several-chats", action="store_true", help="the coordinator with batch slots (--batch 2), and two prompts at once")
    ap.add_argument("--overlap", action="store_true", help="the coordinator with pipelined windows (--pipeline-windows 2)")
    ap.add_argument("--worker-reserve-mib", type=int, default=None,
                    help="VRAM the worker keeps free (--vram-reserve-mib): a pool PC fills its free VRAM with experts, and "
                         "the worker starts first, so on one GPU this leaves the coordinator room")
    ap.add_argument("--picture", default=None, metavar="PATH",
                    help="add a fourth request that carries this image (encoded once, sent to both runs): checks the "
                         "picture's positions across the split.  Needs a model set up with the picture reader")
    ap.add_argument("--picture-question", default="Describe this picture in one short sentence.",
                    help="what the picture request asks (default: a one-sentence description)")
    a = ap.parse_args()
    from serve.server import StrataEngine, child_env, vision_env, Vision, IMAGE_PAD, VISION_START
    import strata_tokenizer as ST

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tpath / "token_type.json").read_text()))
    ids_list = [chat_ids(tok, p) for p in PROMPTS]
    env = child_env(cfg)
    log = str(Path(cfg.get("log") or "strata-selftest.log").with_suffix("")) + "-selftest.log"
    say = lambda m: print(m, flush=True)   # noqa: E731

    # a job is (ids, embeddings file): the file is None for a text prompt
    jobs: list[tuple[list[int], str | None]] = [(i, None) for i in ids_list]
    pic = False
    if a.picture:
        vcfg = cfg.get("vision") or {}
        if not vcfg:
            say('[selftest] --picture needs a model set up with the picture reader: this run config has no "vision" '
                "section (run setup again with images on)")
            return 2
        if not Path(a.picture).is_file():
            say(f"[selftest] --picture: there is no file {a.picture}")
            return 2
        # The reader and an engine must not be on the GPU at the same time (the server encodes inside the same FIFO as
        # its requests for this reason), so the picture is encoded here - before either engine starts, and once, so
        # both runs are given the very same embeddings.  Its environment is the server's too: the config may name a
        # card of its own for the reader (vision_env, #408).
        say(f"[selftest] encoding {a.picture} with the picture reader ...")
        if a.worker_reserve_mib is None:
            say("    note: on a split the expert cache cannot make room for the reader (the server says the same) - "
                "if the first picture fails for VRAM, add --worker-reserve-mib 1600")
        vlog = open(str(Path(log).with_suffix("")) + "-vision.log", "a", encoding="utf-8")
        vis = Vision(vcfg, log=vlog, env=vision_env(cfg, env))
        try:
            emb, n_img = vis.encode(a.picture)
        finally:
            vis.unload()
        pic = True
        say(f"    the image became {n_img} tokens; embeddings in {emb}")
        jobs = jobs + [(picture_ids(tok, a.picture_question, n_img,
                                    tok.encode(IMAGE_PAD, parse_special=True)[0],
                                    tok.encode(VISION_START, parse_special=True)[0]), str(emb))]

    single = None
    if not a.skip_single:
        say(f"[selftest] 1/2: the model alone ({2 * a.cache} cache slots, context {a.ctx}) ...")
        eng = StrataEngine(cfg["exe"], with_vision(base_args(cfg, a.ctx, 2 * a.cache, a.chunk), pic),
                           cwd=cfg.get("cwd"), log=log, env=env)
        try:
            single = run_prompts(eng, jobs, a.max_new, say)
        finally:
            eng.close()

    say(f"[selftest] 2/2: the pool on this PC: coordinator layers 0-{int(a.split) - 1}, worker {a.split}-47 "
        f"over 127.0.0.1:{a.port} ({a.wire}) ...")
    secret = secrets.token_hex(8)
    penv = dict(env, STRATA_POOL_SECRET=secret, STRATA_POOL_NAME="selftest")
    wargs = base_args(cfg, a.ctx, a.cache, a.chunk) + ["--pool-listen", f"127.0.0.1:{a.port}"]
    if a.worker_reserve_mib is not None:
        wargs = with_arg(wargs, "--vram-reserve-mib", str(a.worker_reserve_mib))
    wlog = open(str(Path(log).with_suffix("")) + "-worker.log", "a", encoding="utf-8")
    worker = subprocess.Popen([cfg["exe"], "--serve", *wargs], cwd=cfg.get("cwd"), stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=wlog, text=True, env=penv)
    lines = []

    def pump():
        for line in worker.stdout:
            lines.append(line.strip())
            if line.startswith("POOL "):
                say("    worker: " + line.strip())
    threading.Thread(target=pump, daemon=True).start()
    for _ in range(600):                                   # its weights first, then it listens
        if any(x.startswith("POOL LISTEN") for x in lines) or worker.poll() is not None:
            break
        time.sleep(0.5)
    if worker.poll() is not None:
        say(f"[selftest] the worker engine exited (code {worker.returncode}); see {wlog.name}")
        return 1
    cargs = with_vision(base_args(cfg, a.ctx, a.cache, a.chunk), pic) + \
        ["--pool-peers", f"127.0.0.1:{a.port}", "--pool-split", a.split, "--pool-wire", a.wire]
    cenv = dict(penv)
    if a.several_chats:
        cargs += ["--batch", "2"]
    if a.overlap:
        cargs += ["--pipeline-windows", "2"]
        if a.several_chats:
            cenv["STRATA_PW_WITH_BATCH"] = "1"              # as serve/pool.py's pool_env sets it with both switches
    try:
        eng = StrataEngine(cfg["exe"], cargs, cwd=cfg.get("cwd"), log=log, env=cenv)
    except RuntimeError as e:
        say(f"[selftest] the coordinator did not start: {e} (see {log})")
        worker.kill()
        return 1
    try:
        say(f"    pool: {eng.info.get('pool_workers')}, {eng.info.get('expert_slots')} experts on the GPU in total")
        pooled = run_prompts(eng, jobs, a.max_new, say)
        if a.several_chats:                                # two at once: the lone chat's path hands over to the slots
            say("    two prompts at once ...")
            both: list = [None, None]

            def one(k: int):
                both[k] = run_prompts(eng, [jobs[k + 1]], a.max_new, say)[0]   # text prompts: the picture stays alone
            th = [threading.Thread(target=one, args=(k,)) for k in (0, 1)]
            for t in th:
                t.start()
            for t in th:
                t.join()
            for k in (0, 1):
                got = both[k]["tokens"] if both[k] else []
                say(f"    at once [{k + 2}]: {len(got)} tokens, "
                    f"{'the same as alone' if got == pooled[k + 1]['tokens'] else 'not the same as alone'}")
                if not got:
                    pooled[k + 1]["tokens"] = []            # counted as a failure below
    finally:
        eng.close()
        worker.terminate()
        try:
            worker.wait(30)
        except subprocess.TimeoutExpired:
            worker.kill()

    say("")
    ok = True
    names = list(PROMPTS) + ([f"a picture ({a.picture}): {a.picture_question}"] if pic else [])
    for i, p in enumerate(names):
        pt = pooled[i]["tokens"]
        text = tok.decode(pt)
        say(f"[{i + 1}] {p}")
        say(f"    pool : {text.strip()[:300]!r}")
        if not pt:
            ok = False
        if single:
            st = single[i]["tokens"]
            same = 0
            while same < min(len(st), len(pt)) and st[same] == pt[same]:
                same += 1
            say(f"    alone: {tok.decode(st).strip()[:300]!r}")
            say(f"    the first {same} of {max(len(st), len(pt))} tokens are the same; "
                f"decode {single[i].get('generated', 0) / max(single[i].get('decode_ms', 1) / 1000, 1e-3):.1f} tok/s "
                f"alone, {pooled[i].get('generated', 0) / max(pooled[i].get('decode_ms', 1) / 1000, 1e-3):.1f} pooled")
            if pic and i == len(names) - 1 and same == 0:
                # the picture is where a wrong position table shows: a worker that rotates its own layers by the
                # identity still answers in good English, from the first token (docs/POOL_VISION.md).  Answers may
                # part later - this cache is small and each side rounds experts its own way - so only no agreement at
                # all is called a failure here
                ok = False
                say("    [selftest] the pool's picture answer shares no token with the model alone: the image "
                    "positions did not cross the split the same way (this version's engine on both sides?)")
    if "paris" not in tok.decode(pooled[0]["tokens"]).lower():
        ok = False
        say("[selftest] the pool's first answer does not say Paris")
    say(f"[selftest] {'PASS' if ok else 'FAIL'}: the pool " + ("answers end to end" if ok else "did not answer correctly"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
