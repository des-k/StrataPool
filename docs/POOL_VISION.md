# Pictures on a pool (the split across PCs)

A pool reads pictures now: the model's layers are split over the PCs and a chat sends an image.
For what a pool is and how to set one up, see [POOL.md](POOL.md). This page is how the pictures cross, and what is
and is not proven.

## What one PC does with a picture

1. **`strata-vision` reads the picture.** It is its own process, started by the server (`serve/server.py`, the
   `Vision` class), and it writes the picture's rows to a file: `SVE1` records of `n x n_embd` floats, one per image,
   in prompt order.
2. **The engine gets `GENI <max_new> <file> <ids>`.** Each image's rows go to its run of `<|image_pad|>` tokens as an
   embedding override (`sp.embd_rows`, `src/program/generate.cpp`). The engine was started with `--vision` to accept
   these; the rows are read once, per request.
3. **The picture rotates by three positions.** qwen4exp uses interleaved M-RoPE: every token has a `(t, h, w)` and an
   image's tokens share `t` and differ in `h` and `w` (`include/strata/kernels/mrope.hpp`). The engine keeps that in a
   device table `int32[cells][3]` with `cells = max_context + 64`. **Every rope kernel reads it**, in the prompt path,
   in the verify window and in the draft layer - and it must be *allocated* before any CUDA graph is captured, though
   its contents may change between requests. A text request puts the identity back.

Only the third one is a problem across PCs: it lives inside every layer, and the layers live on different PCs.
The embeddings are not: they enter at the embedding stage, which is on the coordinator, before the rows cross.

## What the pool does now

| Step | Who | Where |
| --- | --- | --- |
| `CONFIG vision=1` | coordinator -> every worker | `PoolLink::config_for`; the worker allocates its own table while it loads, before its graphs are captured |
| `READY mrope=1` | worker -> coordinator | says this PC's engine keeps such a table |
| `MROPE i64 cell0, i32 n, int32 cells[n*3]` | coordinator -> every worker | a picture request ships its cells ahead of its first window |
| lazy ACK | worker -> coordinator | drained with the window's other ACKs, so a picture costs no round trip of its own |

The table is one per PC, which is what the layer split *inside* one PC already does for its extra cards: a second
device gets its own table before its graphs are captured (`src/program/generate.cpp`, the `--layer-split` stage
setup) and the picture request uploads to each. The pool follows the same rule one level up: the coordinator sends
its cells over the link, and each PC's rope kernels read their own copy.

Nothing else crosses that did not cross before. `VERIFY` and `PREFILL` already carry cell indices (every caller in
the engine passes a cell index, `mrope.hpp`), and `VERIFY`'s rows already hold the picture's embeddings.

### Why a worker without the table means no pictures, not a warning

A worker whose engine has no table would rotate its layers by the identity while the coordinator rotates its own by
the image's `(t, h, w)`. The token after that is not an error - it is a plausible wrong word, and every layer above
it keeps going. So the pool refuses instead of half-shipping:

- the coordinator asks, and only sends cells when **every** worker answered `mrope=1` (`PoolLink::vision_ok`);
- otherwise it turns pictures off at start and says why, once, in the log;
- `GENI` then answers with the reason rather than a silent wrong answer;
- the Pool tab says it before the start (`config_problems` in `serve/pool.py`, which reads the marker out of the
  engine binary the way it reads `--pool-listen`).

## What it costs

- **VRAM**: `(max_context + 64) x 12` bytes per PC - 480 KiB at a 40K context, 1,5 MiB at 128K. On a worker it is
  there only when the pool carries pictures.
- **Network**: today a picture request sends the whole table once (1,5 MiB at 128K), and the request that follows a
  picture sends the identity back once. A text-only request sends nothing. Sending only the cells that changed is
  open work below - the frame already carries a start cell and a count for it.
- **The picture's own rows**: `n x n_embd x 4` bytes from `strata-vision` to the engine on the same PC, as on one PC.

No timing is claimed here. These are sizes, from the widths in the code.

## When pictures stay off

- a worker runs an engine built before this: `images are off for this pool` in the start log, and the Pool tab says
  it first;
- **DeepSeek-V4.1-Flash**: its own engine has no layer split, so a pool of it uses Share requests
  ([POOL.md](POOL.md#what-you-need));
- `--layer-split` with a pool: a pool node uses one GPU, as before.

## Open work

- **Sending less per picture request.** Read from the code rather than guessed: everything from the first picture's
  cell to the **end of the table** changes, because the text after a picture is shifted and the cells past the prompt
  are filled with that same shift (the tail fill in `src/program/generate.cpp`). So "send only what changed" saves
  nothing when the picture sits near the start, which is the usual shape of an agent conversation. What would pay, in
  order: skip the send when the table equals the one already shipped; and, only if a measurement on a real pool asks
  for it, send the *rule* instead of the data ("identity up to cell X, this grid at these cells, everything after
  shifted by S" - about 30 bytes) and let each PC build the table itself. The rule must produce a bit-identical
  table, so it needs a comparison test before anyone trusts it.
- The tower needs its own CUDA context on the coordinator's card, and **on a split the expert cache cannot make room
  for it** - the engine turns `--vram-elastic` off there, so the server already warns to keep the room up front
  (`--vram-reserve-mib` on the encoder's card; `serve/server.py`, the `vision` startup note). A config can also name a
  card of its own for the reader (`vision.cuda_device`, #408), which leaves the engine's card whole. When the card has
  nothing to give, setup can put the reader on the CPU (`--vision cpu`), slowly.

## What was checked, and what was not

Checked on one machine (macOS, Apple Silicon, clang 21, no CUDA, no model, no second PC), 2026-11-08:

| Check | Result |
| --- | --- |
| `strata-pool-test` (frames, codecs, split search - no GPU) | **passed**: `all checks passed` |
| `src/program/generate.cpp`, `src/pool/link.cpp`, `src/pool/pool_link_test.cpp` | syntax-check **clean** through `metal/`'s compat headers; both new blocks sit outside every `#if`, so they were really compiled |
| `config_problems` + the MROPE marker in `serve/pool.py` | **10 cases as expected** (CPython 3.12) |
| `tools/test_pool_picture.py` (new) | **passed**: 7 tests - the ids a picture request carries (the marker grows to the reader's token count, a marker quoted in text stays one token), the `--vision` switch, the embeddings reaching the engine (`GEN` vs `GENI`), and the bench's `data:` URL |
| `tools/pool_selftest.py`, `tools/pool_bench.py` | byte-compiled; `--help` and the missing-file path run. Their picture run needs a GPU and a model: **not run** |

Not run here, and needed before this is called working:

| Check | Why it matters |
| --- | --- |
| `strata-pool-link-test` | it links `strata_engine`, so it needs a CUDA or HIP machine. Its new `test_vision_positions` covers CONFIG/READY, the cells reaching both workers, the lazy ACKs, and a worker without a table |
| a CUDA and a HIP build of every backend a file touches | CONTRIBUTING: this repo's CI does not build them |
| **a picture on two PCs, and the same picture on one PC** | the only check that proves the positions agree: same request, greedy, and the answers should match closely (see below) |
| `serve/test_pool.py` whole | its import needs `jinja2`, which this machine does not have (the changed test was run directly instead) |

## Checking it on your two PCs

1. Rebuild the engine **on both PCs** (`START-HERE.bat`, or `--build`), then `POOL-INTEGRITY.bat` on each. A worker
   that was not rebuilt turns pictures off, and the start log says so.
2. Build and run the pool tests on either PC:
   `ctest --test-dir build -R strata_pool_test --output-on-failure` and `build/strata-pool-link-test`
   (the link test is built but not registered in CTest).
3. Turn the picture reader on in the Pool tab's model (setup asks about images; `--vision` in its args), pick
   **Split layers**, Apply, and check the log has no `images are off` line.
4. The check that matters, same request both ways, greedy:

   ```sh
   # one PC alone (pool off), then on the coordinator with the pool running
   curl -s http://127.0.0.1:8095/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model": "strata", "temperature": 0, "max_tokens": 64,
     "messages": [{"role": "user", "content": [
       {"type": "image_url", "image_url": {"url": "file:///path/to/shot.png"}},
       {"type": "text", "text": "What is in this picture?"}]}]}'
   ```

   Answers need not match token for token - a pool caches other experts than one PC
   (`bench/results/2026-09-27-cache-parity`) - but a picture the pool reads wrong says something unrelated, or says
   the picture is missing.
5. The check that runs by itself, on **one** PC (both engines share the GPU, like `POOL-SELFTEST.bat` always did):

   ```sh
   .venv\Scripts\python tools\pool_selftest.py strata-<model>.json --picture shot.png
   ```

   It encodes the picture once, before either engine starts, and sends the same embeddings to the model alone and to
   the split. It prints how many leading tokens the two answers share and fails when the picture answer shares
   **none** - which is what a position table that did not cross the split looks like. Answers may still part later:
   this test runs a small expert cache, and each side rounds its experts its own way.
6. Across a real two-PC pool, the same comparison through the server, by number rather than by reading:

   ```sh
   .venv\Scripts\python tools\pool_bench.py --label alone-with-picture --skip pair,agent --picture shot.png
   .venv\Scripts\python tools\pool_bench.py --label split-with-picture --skip pair,agent --picture shot.png   # on the coordinator
   ```

   Compare the `text_sha1` of the two `picture` rows (each run carries its own id in the prompt, so nothing is reused
   from a cache). A different hash is not yet a failure - the same is true of `code` and `prose` on two machines with
   different expert caches - but the picture row is the one to read together with step 4.
