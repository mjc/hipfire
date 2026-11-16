# gfx1010 Lloyd MQ3 batched prefill

- **Date:** 2026-11-06
- **Lifecycle:** historical; fixture-bound measurement, not a performance floor or admission.
- **GPU:** RX 5700 XT, gfx1010, HIP 7.2, 8,176 MiB reported VRAM. The system hipfire service was stopped and ffmpeg was paused. A single clock observation during the earlier paired run was 1,995 MHz with a junction temperature of 88 °C; clocks and power were not sampled for every run.
- **Model:** Qwen3.5 9B Lloyd MQ3, 4,569,785,344 bytes, md5 `a25c2361b41343ac17348b1a1c3631cd`.
- **Prompt:** [`ttft_511.txt`](../../benchmarks/prompts/ttft_511.txt), md5 `7423e8940920082c6fa11576d23bc9a2`; 511 tokens through the bench request.
- **Revisions:** baseline `efa8fbb1d666b3b9c9732c924802373382bd10c4`; candidate `26223450ceb575e8816d6f92c0d158017f2b4643`. Both binaries were built in the shared hipfire devenv. Candidate was built from the working tree whose code became that commit.
- **Binary md5:** baseline CLI `9fe04f75a89a1b2aace588a65743003a`, daemon `4ab088eb9013344292e3b54f843d2620`; candidate CLI `017daa0b7b5f42f913ad133a32677ff5`, daemon `585a0a7a476eadc7577436f87d2976d8`.

## Method and results

Three interleaved fresh-process pairs, each with two warmups and three measured
runs. Each arm used `hipfire bench` with `--backend noslots --workload stateless
--spec off --kv-mode q8 --max-seq 1024 --runs 3 --warmups 2 --max-tokens 96
--json`, `HIPFIRE_GRAPH=0`, `HIPFIRE_DEVICES=gfx1010`, and a 10-second DPM warmup.
The effective KV backend was legacy. The full JSON results are the `lloyd-final-*` files in
[`data-2026-11-06-gfx1010-lloyd-mq3/`](data-2026-11-06-gfx1010-lloyd-mq3/);
`lloyd-baseline-*` and `lloyd-candidate-*` preserve the earlier run before the
mixed-format guard.

| Fresh process | Baseline TTFT ms, three samples | Candidate TTFT ms, three samples |
|---|---|---|
| 1 | 12007.8, 12030.2, 12027.5 | 4670.7, 4674.4, 4670.7 |
| 2 | 11997.6, 12005.7, 11985.0 | 4665.1, 4663.4, 4677.7 |
| 3 | 12035.7, 12044.4, 12039.6 | 4664.7, 4673.4, 4665.6 |

The pooled TTFT medians were 12,027.5 ms and 4,670.7 ms, a 61.17% reduction.
Pooled decode medians were 40.6 and 40.7 tok/s. These numbers describe this
fixture only. The samples are unusually tight; the serve outputs below were
inspected for degeneracy, but that does not establish performance on other
prompts or systems.

## Correctness evidence and limits

The gfx1010 kernel oracle checked set and residual accumulation on 15 combinations
of group counts 1, 2, and 5 and batch sizes 1, 7, 8, 9, and 17 against a CPU
reference. Focused admission tests and `scripts/pre-commit-ci.sh` passed. The
legacy-Q8 greedy serve battery and related-turn chain each completed five turns
without empty, attractor, runaway, or retrieval-miss flags at 256 output tokens;
the decoded responses were read. A graph-on coding prompt remained coherent but
hit the 256-token limit, so it is not a completed response.

Before the mixed-format guard, a production-path prefill oracle was built for
both revisions with its KV backend set to legacy Q8. A post-guard model smoke
confirmed this fixture still takes the batched route. Two raw tokenized prompts (232 coding tokens and 499 benchmark
text tokens, without the daemon chat wrapper) had identical 32-token greedy
continuations. Final-logit relative L2 differences were 0.00974 and 0.02508;
the maximum per-position KL divergences over the 32 identical-prefix logits
were 0.000155 and 0.001751. These are comparisons against the baseline, not a
teacher quality evaluation or bit-exact proof. The per-position values are in
[`oracle-legacy-summary.json`](data-2026-11-06-gfx1010-lloyd-mq3/oracle-legacy-summary.json).

The Redline daemon harness reported stable decode capture, valid AQL contracts,
and three-position HIP/AQL shadow parity. Its prefill capture recorded zero
launches, so it does not prove capture or replay of the new prefill kernels.
The local receipt is `runs/candidate-redline.json` under
`.codeinsight+research/gfx1010-qwen35-9b-2026-11-06/`; that path is a discovery
pointer, not a durable artifact. The gfx1010 batched path has not been measured
on adjacent architectures and is admitted only on gfx1010.
