# Internal runtime fork

This repository retains the upstream llama.cpp history and MIT license. The
baseline for this update is `b76b13379d3b7d67ab0bc424516bb6fd2522bd8e`, which
includes v0.7.0 and the pooled-embedding optimization from upstream PR #30445.
The initial GitLab project README is preserved in
[gitlab-project-initial-readme.md](gitlab-project-initial-readme.md).

## Local changes to the server

`POST /completion` accepts the optional boolean `use_slot_prompt` (default
`false`). With an explicit `id_slot`, it prepends that slot's cached tokens to
the submitted prompt. A request must select a populated slot and contain a
single completion. This extension is independent of model names, clients,
databases and deployment environments; it is not a standard OpenAI parameter.

Typical sequence:

1. Prefill a document using `/completion` with `id_slot: 0`, `n_predict: 0`
   and `cache_prompt: true`.
2. Save it with `POST /slots/0?action=save` and a `filename`.
3. Restore it into a reserved slot using `/slots/0?action=restore`.
4. Send the continuation to `/completion` with `id_slot: 0`,
   `use_slot_prompt: true`, `cache_prompt: true` and the new prompt only.

The client must reserve the slot for the entire restore/continue sequence.
An explicit slot number is not an authorization or concurrency lock. Restoring
state into a slot shared by unrelated requests can change their context.
Token arrays avoid retokenizing the boundary between prefix and continuation.

A second small patch rejects recurrent checkpoints extending beyond the
validated common token prefix. The core context implementation uses upstream's
native rollback settings. The historical unconditional `n_rs_seq = 8` override,
manual checkpoint insertion after restore, and custom position arithmetic are
not part of this version. Upstream now persists and restores checkpoint data.

## Compatibility evidence

Isolated CPU tests passed with a small transformer and Qwen3.5-0.8B Q4_K_M:
save/restore, deterministic continuation, process restart, document isolation,
invalid slot rejection, parallel streaming and corrupt-state rejection. The
hybrid model also passed OpenAI chat and streaming. A long hybrid prefix of
8,801 tokens required only 6 new prompt tokens for continuation, with output
identical to the full 8,807-token request, using native rollback settings.

These tests do not establish GPU throughput or compatibility with every model.
An old hybrid cache from build 9426 was rejected by the updated server. Keep
old runtime/cache pairs and regenerate caches for the new version in a separate
directory. Model weights are unchanged by this process.

Run tests against any supplied server and model:

```sh
python tests/prompt-cache/regression.py --binary /path/to/llama-server \
  --model /path/to/model.gguf --output /private/results

# Optional hybrid checkpoint test; allow time for CPU prefill.
python tests/prompt-cache/regression.py --binary /path/to/llama-server \
  --model /path/to/hybrid.gguf --output /private/long-results \
  --context-size 32768 --prefix-repetitions 800
```

## GitLab CI

The pipeline uses shell runners tagged `linux-build` and `windows-build`.
They must be enabled for this project. It automatically builds Linux CPU and
Windows CUDA with CPU fallback. A separate Windows CPU build is manual.

Prerequisites for the runner service account:

- Both: Git, Python 3.10+, CMake 3.28+, network access to upstream build
  dependencies and the pinned 1.2 MB synthetic test model.
- Linux: GCC/Clang C++17 toolchain and make or Ninja.
- Windows: Python available as `python.exe`, `python3.exe` or the `py.exe` launcher, Visual Studio Build Tools with
  MSVC x64 and a Windows SDK, and a CUDA Toolkit compatible with that compiler.
  `CUDA_PATH` must identify the selected toolkit. The current CUDA recipe
  follows the upstream CUDA 12.4 configuration. Ninja is preferred; the Visual
  Studio generator is a fallback.

No toolchain installer runs in these jobs. Windows uses PowerShell 5.1 and
initializes the installed MSVC environment. BoringSSL is compiled for TLS.

Pipeline variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BUILD_JOBS` | `4` | Limit compiler concurrency. |
| `CUDA_ARCHITECTURES` | `86` | CMake CUDA architectures; change for other GPUs. |
| `CUDA_CCCL_VERSION` | `v3.4.3` | Upstream CCCL compatibility dependency; empty uses the toolkit's bundled version. |
| `MSVC_TOOLSET_VERSION` | unset | Optional installed MSVC toolset selection, such as `14.39`, for CUDA/compiler compatibility. |
| `PUBLISH_PACKAGES` | `1` | Publish only from the default branch or a protected ref. Set `0` to retain local output only. |

Packages include all generated server DLLs/shared libraries, CPU backends,
licenses, source/build provenance, per-file SHA256 values and test results.
Windows CUDA packages also carry the required CUDA runtime/cuBLAS DLLs and
toolkit EULA. Target Windows machines need the Microsoft Visual C++ x64
Redistributable and a compatible NVIDIA driver. Linux packages depend on the
runner's libc/libstdc++ and OpenMP; build on the oldest supported target distro.

Tests launch the staged package on loopback with temporary slot files and CPU
inference. Failure prevents publication. Models and prompts are excluded from
release archives. CUDA compilation plus CPU testing is not a GPU validation;
test a new package on a separate worker port before adopting it.

After tests pass, the default branch publishes archives, checksums and manifests
to **Deploy > Package registry** using `CI_JOB_TOKEN`. Versions contain the
source SHA, pipeline ID and job ID. Job artifacts retain only small reports,
avoiding the job-artifact size limit for large CUDA packages. GitLab CA files
are honored without disabling TLS verification. Build outputs also remain in
the job's `dist/` directory until runner cleanup.

There is no deployment job in this repository. Consumers choose an immutable
package version. Upstream merges should be reviewed against the three modified
server files and rerun the transformer/hybrid tests before release.

For a local CPU build and package, run `python3 ci/gitlab/build_runtime.py
--backend cpu`. To verify an existing build, add `--skip-build --build-dir
/path/to/build --model /path/to/test.gguf`.

This fork's port and CI tooling were prepared with OpenAI Codex assistance.
