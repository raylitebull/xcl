# Testing and Merge-Checking Pull Requests

This guide is for contributors and AI agents asked to test or review a pull request (e.g., `"test #1234"` or `"test <PR URL>"`). Its goal is to turn every test run into a clear, `#3041`-compatible verdict so maintainers do not have to guess whether a PR is ready from raw test counts alone.

- **Current focus and blockers:** [#1833](https://github.com/xcl-dev/xcl/issues/1833)
- **Duplicate PR tracker and batches:** [#1607](https://github.com/xcl-dev/xcl/issues/1607)
- **Official merge queue:** [#3041](https://github.com/xcl-dev/xcl/issues/3041)

## 1. Pick a PR (if none was given)

1. Open [#3041](https://github.com/xcl-dev/xcl/issues/3041) and pick the highest-ranked PR marked `free` (or with 1 check if you have a different GPU vendor, e.g. NVIDIA vs AMD).
2. Or check the live machine-readable testing queue (gate-passing PRs still needing testers):
   `https://raw.githubusercontent.com/xcl-dev/xcl/testing-feed/queue.json`
3. Before spending time testing, verify the PR is not a draft, has no merge conflicts, has green `Build and test (Windows, no GPU)` + `PR description` checks, and is not already superseded in [#1607](https://github.com/xcl-dev/xcl/issues/1607).

## 2. Claim the PR (when doing a merge-check)

If you intend to post a full `merge-check`, check that the PR's current head does not already have two active claims/checks (a third check is useful only when the first two disagree or for a GPU vendor neither tested). Post on **the PR** (not on #3041):

```text
merge-check claim
head: <full PR head commit SHA>
```

Claims expire after 6 hours. When finished, edit the claim comment into your `merge-check` report (or post a new comment on the PR).

## 3. Choose the right test tier

Do **not** run a long shader precompile or full world-load session for every PR. Match the test depth to what the PR touches:

| Tier | What the PR changes | Required verification |
|---|---|---|
| **Tier 0** | Docs, scripts, CI workflows, or test-only diffs | Diff review + relevant script tests (`python -m unittest ...`) or `cargo test` for touched test targets. No retail game run needed. |
| **Tier 1** | Recompiler/ISA/DXIL logic, HLE (`kernelx`, `XGameRuntime`, `VFS`, `PE loader`, `DirectStorage`), CLI/tooling | Clean merge onto current base + `cargo build --workspace` + full `cargo test --workspace --no-fail-fast -- --test-threads=4` (report exact counts and `SKIPPED` GPU tests) + full diff review. |
| **Tier 2** | `D3D12.X` runtime, memory/VRAM/sparse heaps, command lists, sync/barriers | Everything in **Tier 1** + a 60 s `scripts/smoke.ps1 -Game <folder>` comparison on the base commit vs. the merged PR tree (if you own a retail title). If you cannot run a retail title, say so explicitly under `review:` / `performance:` and note what remains untested. |
| **Tier 3** | Shader prep/caching/PSO libraries, mesh/graphics rendering, world-load blockers (`#1833`) | Everything in **Tier 2** + targeted verification of the changed path (e.g., world-load / FPS / peak commit / visual check on the target GPU: wave32 NVIDIA or wave64 AMD). |

## 4. Run the merge-check locally

Never modify `main` or test against a stale base:

```powershell
git fetch origin
git worktree add ..\wt-check-<PR> --detach origin/<base>
cd ..\wt-check-<PR>
git fetch origin pull/<PR>/head:pr-<PR>
git merge --no-edit pr-<PR>          # verify clean merge against live base
git diff --check origin/<base>..HEAD # whitespace / conflict markers
cargo build --workspace
cargo test --workspace --no-fail-fast -- --test-threads=4
```

- Count and report every `SKIPPED <test>: <reason>` notice — skipped GPU tests are **not** coverage.
- Read the **entire diff** (`git diff origin/<base>..pr-<PR>`) before calling a PR safe: check scope (under ~400 lines unless justified), hidden behavior changes, UB/lifetime/sync bugs, license/provenance rules, and whether tests actually cover the claimed fix.
- Clean up your temporary worktree when done (`git worktree remove ..\wt-check-<PR>`).

## 5. Post the exact `#3041` `merge-check` format

The `#3041` queue parser reads the first line (`merge-check`), the `head:` commit SHA, and the `verdict:` line (`safe to merge` or `not yet`). **Always include a clear human-readable verdict and why** — never post raw test counts alone.

```text
merge-check
base: <full base commit SHA you merged into>
head: <full PR head commit SHA>
merge: clean | conflicts
build: cargo build --workspace -> ok | failed
tests: cargo test --workspace --no-fail-fast -- --test-threads=4 -> N passed, N failed, N ignored; SKIPPED: <count and reasons>; failing: <none, or names and whether they fail on base too>
review: <concrete findings in the merge result, most serious first; write "none" only after reading the whole diff; mention any untested GPU/game paths>
performance: <can it cost frame time or memory by default? measured (smoke/world delta) or reasoned>
verdict: safe to merge | not yet: <specific blocker or missing evidence>
hardware / tool and model: <GPU + driver, CPU, Windows version> / <manual, or tool and exact model name>
```

### Verdict discipline
- Use `verdict: safe to merge` **only** when the merge is clean, build and suite pass (no new failures), you read the full diff with no unresolved critical findings, and the evidence matches the PR's tier.
- Use `verdict: not yet: <reason>` whenever anything blocks merging (failing test, bug/UB in diff, missing GPU coverage for a vendor-specific change, unmeasured perf/memory risk, scope/duplicate issue, or stale/missing smoke run).
- One open critical finding or measured regression blocks a merge regardless of how many other checks said `safe to merge`.
- **Never** invent numbers, test counts, or `xcl-smoke` blocks. Report only what you actually ran on your machine.
- **Agents:** do not post comments on GitHub unless the user explicitly asked you to post them in this session; otherwise print the completed `merge-check` block for the user to review and post.
