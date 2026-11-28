"""Deterministic PR triage, label sync, and live testing-queue feed generator.

Used by:
- .github/workflows/auto-triage.yml (single-PR label sync on PR / merge-check events)
- .github/workflows/testing-feed.yml (scheduled queue.json publisher on `testing-feed` branch)

Standard library only; no LLM calls; never checks out or executes untrusted PR code.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

CI_CHECK_NAME = "Build and test (Windows, no GPU)"
PR_CHECK_NAME = "PR description"

MANAGED_LABELS = frozenset({
    "gpu:nvidia",
    "gpu:amd",
    "gpu:both",
    "needs:nvidia-test",
    "needs:amd-test",
    "area:d3d12",
    "area:recompiler",
    "area:shaders",
    "area:memory",
    "area:directstorage",
    "area:hle",
    "area:tests",
    "area:docs-ci",
    "needs-rebase",
    "needs-pr-check",
    "needs-1st-check",
    "needs-2nd-check",
    "ready-to-merge",
    "checks-disagree",
})

NVIDIA_RE = re.compile(
    r"\b(nvidia|rtx\s*\d*|gtx\s*\d*|geforce|wave32|split[-_ ]?wave64|nvdev|upper_lanes)\b",
    re.I,
)
AMD_RE = re.compile(
    r"\b(amd(?!\s+(?:ryzen|epyc|threadripper|athlon))|radeon|rdna\s*\d*|rx\s*\d{3,4}\w*|wave64)\b",
    re.I,
)
BODY_TEST_SECTION_RE = re.compile(r"\*\*(?:Tested|Regression run|Not tested|AI use)\*\*", re.I)
# Matches a SHA in `head: abcdef1234` or ``head: [`abcdef1`](url)`` or `head: \`abcdef1\``
HEAD_SHA_RE = re.compile(r"^\s*head\s*:\s*(?:\[?`?([0-9a-fA-F]{7,40})`?\]?)", re.I)
VERDICT_RE = re.compile(r"^\s*verdict\s*:\s*(.+?)\s*$", re.I)

MEMORY_PATHS = (
    "crates/xcl/src/gpu/uma.rs",
    "crates/xcl/src/hle/d3d12x/vram.rs",
    "crates/xcl/src/hle/d3d12x/console_heap.rs",
    "crates/xcl/src/hle/d3d12x/sparse.rs",
    "crates/xcl/src/hle/d3d12x/alloc_pool.rs",
    "crates/xcl/src/hle/d3d12x/memory_report.rs",
    "crates/xcl/src/hle/d3d12x/uma_driver.rs",
    "crates/xcl/src/hle/kernelx/lazy.rs",
    "crates/xcl/src/hle/kernelx/placeholders.rs",
)

SHADER_PATHS = (
    "crates/xcl/src/shaders.rs",
    "crates/xcl/src/shaders/",
    "crates/xcl/src/gpu/shader.rs",
    "crates/xcl/src/gpu/pack.rs",
    "crates/xcl/src/hle/d3d12x/pso_",
    "crates/xcl/src/hle/d3d12x/shader_",
    "crates/xcl/src/hle/d3d12x/used_journal.rs",
    "crates/xcl/src/hle/d3d12x/present/preparation.rs",
    "scripts/pso_corpus/",
    "scripts/precompile-shaders.ps1",
)

RECOMP_PATHS = (
    "crates/xcl/src/gpu/dxil/",
    "crates/xcl/src/gpu/recomp/",
    "crates/xcl/src/gpu/rdna2",
    "crates/xcl/src/disasm.rs",
)


def is_test_path(path: str) -> bool:
    p = path.replace("\\", "/")
    return (
        p.startswith("tests/")
        or p.startswith("crates/xcl/tests/")
        or p.endswith("_tests.rs")
        or p.endswith("/tests.rs")
        or p.endswith(".tests.ps1")
        or (p.startswith("scripts/") and os.path.basename(p).startswith("test_"))
    )


def is_runtime_path(path: str) -> bool:
    p = path.replace("\\", "/")
    if is_test_path(p):
        return False
    return (
        p.startswith("crates/xcl/src/")
        or p.startswith("crates/xcl/data/")
        or p in ("crates/xcl/build.rs", "crates/xcl/Cargo.toml", "Cargo.toml", "Cargo.lock")
    )


def classify_areas(files: list[str]) -> list[str]:
    areas: set[str] = set()
    for raw in files:
        p = raw.replace("\\", "/")
        if is_test_path(p):
            areas.add("area:tests")
        if any(p == m or p.startswith(m) for m in MEMORY_PATHS):
            areas.add("area:memory")
        if any(p == s or p.startswith(s) for s in SHADER_PATHS):
            areas.add("area:shaders")
        if any(p == r or p.startswith(r) for r in RECOMP_PATHS):
            areas.add("area:recompiler")
        if (
            p.startswith("crates/xcl/src/hle/d3d12x/")
            or p == "crates/xcl/src/hle/d3d12_layout.rs"
            or (
                p.startswith("crates/xcl/src/gpu/")
                and not any(p.startswith(r) for r in RECOMP_PATHS)
                and p not in ("crates/xcl/src/gpu/uma.rs", "crates/xcl/src/gpu/shader.rs", "crates/xcl/src/gpu/pack.rs")
            )
        ):
            areas.add("area:d3d12")
        if p.startswith("crates/xcl/src/hle/dstorage") or p.startswith("crates/xcl/src/hle/bcpack/"):
            areas.add("area:directstorage")
        if (
            (p.startswith("crates/xcl/src/hle/") and not p.startswith("crates/xcl/src/hle/d3d12") and not p.startswith("crates/xcl/src/hle/dstorage") and not p.startswith("crates/xcl/src/hle/bcpack/"))
            or p in (
                "crates/xcl/src/vfs.rs",
                "crates/xcl/src/vfs_hooks.rs",
                "crates/xcl/src/loader.rs",
                "crates/xcl/src/pe.rs",
                "crates/xcl/src/tls.rs",
                "crates/xcl/src/eh.rs",
                "crates/xcl/src/resolver.rs",
                "crates/xcl/src/hooks.rs",
                "crates/xcl/src/stubs.rs",
            )
        ):
            areas.add("area:hle")
        if (
            p.startswith(".github/")
            or p.startswith("docs/")
            or p.endswith(".md")
            or (p.startswith("scripts/") and not is_test_path(p))
        ):
            areas.add("area:docs-ci")
    return sorted(areas)


def classify_gpu_targets(files: list[str], title: str, body: str = "") -> list[str]:
    areas = set(classify_areas(files))
    has_runtime = any(is_runtime_path(f) for f in files)
    gpu_areas = {"area:d3d12", "area:recompiler", "area:shaders", "area:memory"}
    # Ignore Tested / Not tested / Regression run / AI use boilerplate (which always mentions GPU/CPU hardware).
    intro = BODY_TEST_SECTION_RE.split(body, maxsplit=1)[0][:1500]
    text = f"{title}\n{intro}" if (areas & gpu_areas) else title
    path_text = " ".join(files)

    has_nv = bool(NVIDIA_RE.search(text) or NVIDIA_RE.search(path_text))
    # "split-wave64" and "wave64-on-wave32" are NVIDIA wave32 emulation mechanisms, not AMD-only.
    cleaned_for_amd = re.sub(r"split[-_ ]?wave64|wave64[- ]on[- ]wave32", "", text, flags=re.I)
    has_amd = bool(AMD_RE.search(cleaned_for_amd) or AMD_RE.search(path_text))

    if has_nv and has_amd:
        return ["nvidia", "amd"]
    if has_nv and not has_amd:
        return ["nvidia"]
    if has_amd and not has_nv:
        return ["amd"]

    if has_runtime and (areas & gpu_areas):
        return ["nvidia", "amd"]
    return []


def classify_test_tier(files: list[str], title: str, labels: set[str] | None = None) -> int:
    labels = labels or set()
    if not any(is_runtime_path(f) for f in files):
        return 0
    areas = set(classify_areas(files))
    low_title = title.lower()
    if (
        "area:shaders" in areas
        or "goal:world-load" in labels
        or any(k in low_title for k in ("world", "pso", "precompil", "shader", "mesh", "present"))
        or any(
            f.replace("\\", "/").startswith((
                "crates/xcl/src/hle/d3d12x/present",
                "crates/xcl/src/gpu/recomp/mesh_",
                "crates/xcl/src/gpu/recomp/pso",
            ))
            for f in files
        )
    ):
        return 3
    if "area:d3d12" in areas or "area:memory" in areas:
        return 2
    return 1


def build_testing_hint(areas: list[str], gpu_targets: list[str], tier: int) -> str:
    tier_desc = {
        0: "Tier 0: diff review + unit/script tests (no retail game run needed)",
        1: "Tier 1: clean merge + full cargo test suite + diff review",
        2: "Tier 2: full cargo test suite + 60s scripts/smoke.ps1 main-vs-PR",
        3: "Tier 3: full suite + smoke.ps1 + world-load/shader verification",
    }[tier]
    short_areas = [a.removeprefix("area:") for a in areas[:3]]
    area_str = ", ".join(short_areas) if short_areas else "general"
    gpu_str = "/".join(g.upper() for g in gpu_targets) if gpu_targets else "any GPU"
    return f"{tier_desc} [{area_str}; {gpu_str}]"


def parse_merge_checks(
    comments: list[dict[str, Any]],
    pr_author: str,
    head_sha: str,
) -> dict[str, Any]:
    """Parse #3041 merge-check comments for the PR's current head commit.

    Matches throny's #3041 parser rules:
    - First non-empty line is exactly `merge-check`.
    - `head:` line carries a >=7-hex prefix of `head_sha` (plain or markdown-linked).
    - `verdict:` starts with `safe to merge` or `not yet`.
    - PR author's own comments do not count; latest valid check per account wins.
    """
    head_low = (head_sha or "").lower()
    author_low = (pr_author or "").lower()
    by_user: dict[str, dict[str, Any]] = {}

    for c in comments:
        user = (c.get("author") or "").strip()
        if not user or user.lower() == author_low:
            continue
        body = c.get("body") or ""
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if not lines or lines[0].lower() != "merge-check":
            continue

        sha_match = None
        verdict = None
        for ln in lines[1:]:
            if sha_match is None:
                m = HEAD_SHA_RE.match(ln)
                if m:
                    sha_match = m.group(1).lower()
            if verdict is None:
                vm = VERDICT_RE.match(ln)
                if vm:
                    vtxt = vm.group(1).strip().lower()
                    if vtxt.startswith("safe to merge"):
                        verdict = "safe"
                    elif vtxt.startswith("not yet"):
                        verdict = "not_yet"

        if not sha_match or not verdict or not head_low.startswith(sha_match):
            continue

        # Detect which GPU vendor(s) the tester ran on (prefer hardware line if present)
        hw_lines = [ln for ln in lines[1:] if ln.lower().startswith("hardware")]
        hw_text = "\n".join(hw_lines) if hw_lines else body
        hw_nv = bool(re.search(r"\b(nvidia|rtx|gtx|geforce)\b", hw_text, re.I))
        hw_amd = bool(AMD_RE.search(hw_text))

        by_user[user.lower()] = {
            "user": user,
            "verdict": verdict,
            "nvidia": hw_nv,
            "amd": hw_amd,
        }

    safe_users = sorted(v["user"] for v in by_user.values() if v["verdict"] == "safe")
    not_yet_users = sorted(v["user"] for v in by_user.values() if v["verdict"] == "not_yet")
    tested_nvidia = any(v["nvidia"] for v in by_user.values() if v["verdict"] == "safe")
    tested_amd = any(v["amd"] for v in by_user.values() if v["verdict"] == "safe")

    return {
        "safe_users": safe_users,
        "not_yet_users": not_yet_users,
        "safe_count": len(safe_users),
        "not_yet_count": len(not_yet_users),
        "tested_nvidia": tested_nvidia,
        "tested_amd": tested_amd,
    }


def extract_gate_statuses(check_runs: list[dict[str, Any]]) -> tuple[str, str]:
    """Return (ci_state, pr_check_state) using the latest check run per check name."""
    latest: dict[str, dict[str, Any]] = {}
    for run in check_runs:
        name = run.get("name")
        if name not in (CI_CHECK_NAME, PR_CHECK_NAME):
            continue
        ts = run.get("completed_at") or run.get("started_at") or ""
        prev = latest.get(name)
        if prev is None or ts >= (prev.get("completed_at") or prev.get("started_at") or ""):
            latest[name] = run

    def state_of(name: str) -> str:
        r = latest.get(name)
        if not r:
            return "MISSING"
        return (r.get("conclusion") or r.get("status") or "UNKNOWN").upper()

    return state_of(CI_CHECK_NAME), state_of(PR_CHECK_NAME)


def compute_managed_labels(
    *,
    files: list[str],
    title: str,
    body: str,
    existing_labels: set[str],
    is_draft: bool,
    is_conflicting: bool,
    ci_state: str,
    pr_check_state: str,
    merge_checks: dict[str, Any],
) -> set[str]:
    desired: set[str] = set()
    areas = classify_areas(files)
    gpu_targets = classify_gpu_targets(files, title, body)
    desired.update(areas)

    if gpu_targets == ["nvidia"]:
        desired.add("gpu:nvidia")
    elif gpu_targets == ["amd"]:
        desired.add("gpu:amd")
    elif gpu_targets == ["nvidia", "amd"]:
        desired.add("gpu:both")

    if is_conflicting:
        desired.add("needs-rebase")
    if pr_check_state in ("FAILURE", "TIMED_OUT", "ACTION_REQUIRED"):
        desired.add("needs-pr-check")

    gates_green = (
        not is_draft
        and not is_conflicting
        and ci_state == "SUCCESS"
        and pr_check_state == "SUCCESS"
        and "duplicate" not in existing_labels
    )
    if gates_green:
        safe = merge_checks["safe_count"]
        not_yet = merge_checks["not_yet_count"]
        if safe >= 1 and not_yet >= 1:
            desired.add("checks-disagree")
        elif safe >= 2 and not_yet == 0:
            desired.add("ready-to-merge")
        elif safe == 1 and not_yet == 0:
            desired.add("needs-2nd-check")
        elif not_yet == 0:
            desired.add("needs-1st-check")

        still_needs_testing = (safe < 2 and not_yet == 0) or (safe >= 1 and not_yet >= 1)
        if still_needs_testing:
            if "nvidia" in gpu_targets and not merge_checks["tested_nvidia"]:
                desired.add("needs:nvidia-test")
            if "amd" in gpu_targets and not merge_checks["tested_amd"]:
                desired.add("needs:amd-test")

    return desired


class GitHubClient:
    def __init__(self, token: str):
        self.token = token

    def _request(self, method: str, url: str, data: Any = None) -> Any:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "xcl-triage-automation",
        }
        body = None
        if data is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {method} {url} failed ({e.code}): {err_body[:400]}") from e

    def rest(self, method: str, path: str, data: Any = None) -> Any:
        return self._request(method, f"https://api.github.com/{path.lstrip('/')}", data=data)

    def rest_paginated(self, path: str, max_pages: int = 10) -> list[Any]:
        items: list[Any] = []
        sep = "&" if "?" in path else "?"
        for page in range(1, max_pages + 1):
            batch = self.rest("GET", f"{path}{sep}per_page=100&page={page}")
            if not batch:
                break
            items.extend(batch)
            if len(batch) < 100:
                break
        return items

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        res = self._request(
            "POST",
            "https://api.github.com/graphql",
            data={"query": query, "variables": variables or {}},
        )
        if res.get("errors"):
            raise RuntimeError(f"GraphQL error: {res['errors'][:2]}")
        return res["data"]


def sync_pr_labels(gh: GitHubClient, repo: str, number: int, current: set[str], desired: set[str]) -> tuple[list[str], list[str]]:
    to_add = sorted(desired - current)
    to_remove = sorted((current & MANAGED_LABELS) - desired)
    if to_add:
        gh.rest("POST", f"repos/{repo}/issues/{number}/labels", data={"labels": to_add})
    for lbl in to_remove:
        enc = urllib.parse.quote(lbl, safe="")
        try:
            gh.rest("DELETE", f"repos/{repo}/issues/{number}/labels/{enc}")
        except RuntimeError:
            pass
    return to_add, to_remove


def cmd_triage_one(gh: GitHubClient, repo: str, pr_number: int) -> None:
    pr = gh.rest("GET", f"repos/{repo}/pulls/{pr_number}")
    if pr.get("state") != "open":
        print(f"PR #{pr_number} is {pr.get('state')}; skipping.")
        return

    files_raw = gh.rest_paginated(f"repos/{repo}/pulls/{pr_number}/files", max_pages=5)
    files = [f["filename"] for f in files_raw]

    head_sha = pr["head"]["sha"]
    checks_raw = gh.rest("GET", f"repos/{repo}/commits/{head_sha}/check-runs?per_page=100")
    ci_state, pr_check_state = extract_gate_statuses(checks_raw.get("check_runs") or [])

    comments_raw = gh.rest_paginated(f"repos/{repo}/issues/{pr_number}/comments", max_pages=5)
    comments = [
        {"author": (c.get("user") or {}).get("login", ""), "body": c.get("body") or ""}
        for c in comments_raw
    ]
    mc = parse_merge_checks(comments, (pr.get("user") or {}).get("login", ""), head_sha)

    existing = {l["name"] for l in (pr.get("labels") or [])}
    is_conflicting = pr.get("mergeable_state") == "dirty" or pr.get("mergeable") is False

    desired = compute_managed_labels(
        files=files,
        title=pr.get("title") or "",
        body=pr.get("body") or "",
        existing_labels=existing,
        is_draft=bool(pr.get("draft")),
        is_conflicting=is_conflicting,
        ci_state=ci_state,
        pr_check_state=pr_check_state,
        merge_checks=mc,
    )
    added, removed = sync_pr_labels(gh, repo, pr_number, existing, desired)
    print(f"PR #{pr_number}: added={added} removed={removed} ci={ci_state} pr_check={pr_check_state} safe={mc['safe_count']}")


STAGE_A_QUERY = """
query($q: String!, $cursor: String) {
  search(query: $q, type: ISSUE, first: 50, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        number
        title
        url
        isDraft
        mergeable
        baseRefName
        headRefOid
        updatedAt
        author { login }
        labels(first: 25) { nodes { name } }
        commits(last: 1) {
          nodes {
            commit {
              statusCheckRollup {
                contexts(first: 15) {
                  nodes {
                    ... on CheckRun { name conclusion status completedAt startedAt }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""


def fetch_details_batch(gh: GitHubClient, owner: str, name: str, numbers: list[int]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    chunk_size = 20
    for i in range(0, len(numbers), chunk_size):
        chunk = numbers[i : i + chunk_size]
        parts = []
        for n in chunk:
            parts.append(
                f"pr_{n}: pullRequest(number: {n}) {{ "
                f"body files(first: 60) {{ nodes {{ path }} }} "
                f"comments(last: 30) {{ nodes {{ author {{ login }} body }} }} }}"
            )
        query = f'query {{ repository(owner: "{owner}", name: "{name}") {{ ' + " ".join(parts) + " } }"
        data = gh.graphql(query)["repository"]
        for n in chunk:
            node = data.get(f"pr_{n}")
            if node:
                out[n] = node
    return out


def cmd_build_feed(gh: GitHubClient, repo: str, out_path: str, max_pages: int, sync_labels: bool) -> None:
    owner, name = repo.split("/", 1)
    search_q = (
        f"repo:{repo} is:pr is:open -is:draft base:main "
        f"status:success -label:duplicate -label:off-goal sort:updated-desc"
    )
    cursor = None
    all_open: list[dict[str, Any]] = []

    for _ in range(max_pages):
        data = gh.graphql(STAGE_A_QUERY, {"q": search_q, "cursor": cursor})
        pr_conn = data["search"]
        all_open.extend(n for n in pr_conn["nodes"] if n and "number" in n)
        if not pr_conn["pageInfo"]["hasNextPage"]:
            break
        cursor = pr_conn["pageInfo"]["endCursor"]

    candidates: list[dict[str, Any]] = []
    for pr in all_open:
        if pr["isDraft"] or pr["baseRefName"] != "main":
            continue
        labels = {l["name"] for l in pr["labels"]["nodes"]}
        if "duplicate" in labels or "off-goal" in labels:
            continue
        if pr["mergeable"] != "MERGEABLE":
            continue
        commits = pr["commits"]["nodes"]
        contexts = []
        if commits and commits[0]["commit"].get("statusCheckRollup"):
            for ctx in commits[0]["commit"]["statusCheckRollup"]["contexts"]["nodes"]:
                if ctx and "name" in ctx:
                    contexts.append({
                        "name": ctx["name"],
                        "conclusion": ctx.get("conclusion"),
                        "status": ctx.get("status"),
                        "completed_at": ctx.get("completedAt"),
                        "started_at": ctx.get("startedAt"),
                    })
        ci_state, pr_check_state = extract_gate_statuses(contexts)
        if ci_state != "SUCCESS" or pr_check_state != "SUCCESS":
            continue
        pr["_labels"] = labels
        pr["_ci_state"] = ci_state
        pr["_pr_check_state"] = pr_check_state
        candidates.append(pr)

    details = fetch_details_batch(gh, owner, name, [p["number"] for p in candidates])

    items: list[dict[str, Any]] = []
    ready_to_merge: list[dict[str, Any]] = []

    for pr in candidates:
        num = pr["number"]
        det = details.get(num)
        if not det:
            continue
        files = [f["path"] for f in (det.get("files") or {}).get("nodes", [])]
        comments = [
            {"author": (c.get("author") or {}).get("login", ""), "body": c.get("body") or ""}
            for c in (det.get("comments") or {}).get("nodes", [])
        ]
        author = (pr.get("author") or {}).get("login", "")
        head_sha = pr["headRefOid"]
        mc = parse_merge_checks(comments, author, head_sha)
        labels = pr["_labels"]
        body = det.get("body") or ""

        areas = classify_areas(files)
        gpu_targets = classify_gpu_targets(files, pr["title"], body)
        tier = classify_test_tier(files, pr["title"], labels)

        if sync_labels:
            desired = compute_managed_labels(
                files=files,
                title=pr["title"],
                body=body,
                existing_labels=labels,
                is_draft=False,
                is_conflicting=False,
                ci_state=pr["_ci_state"],
                pr_check_state=pr["_pr_check_state"],
                merge_checks=mc,
            )
            sync_pr_labels(gh, repo, num, labels, desired)

        if mc["safe_count"] >= 2 and mc["not_yet_count"] == 0:
            ready_to_merge.append({
                "number": num,
                "title": pr["title"],
                "url": pr["url"],
                "head_sha": head_sha,
                "checked_by": mc["safe_users"],
            })
            continue

        # Exclude PRs blocked solely by `not yet` unless checks disagree (needs 3rd opinion)
        if mc["not_yet_count"] > 0 and mc["safe_count"] == 0:
            continue

        needs_nvidia = "nvidia" in gpu_targets and not mc["tested_nvidia"]
        needs_amd = "amd" in gpu_targets and not mc["tested_amd"]

        items.append({
            "number": num,
            "title": pr["title"],
            "url": pr["url"],
            "author": author,
            "base": pr["baseRefName"],
            "head_sha": head_sha,
            "updated_at": pr["updatedAt"],
            "goal": "goal:world-load" in labels,
            "gpu_targets": gpu_targets,
            "needs_nvidia": needs_nvidia,
            "needs_amd": needs_amd,
            "test_tier": tier,
            "areas": areas,
            "safe_checks": mc["safe_count"],
            "not_yet_checks": mc["not_yet_count"],
            "checked_by": mc["safe_users"],
            "disagrees": mc["safe_count"] >= 1 and mc["not_yet_count"] >= 1,
            "testing_hint": build_testing_hint(areas, gpu_targets, tier),
        })

    # Order: goal:world-load first, then disagrees/1-check-away, then higher tier, then oldest updated
    items.sort(
        key=lambda x: (
            0 if x["goal"] else 1,
            0 if x["disagrees"] else (1 if x["safe_checks"] == 1 else 2),
            -x["test_tier"],
            x["updated_at"],
            x["number"],
        )
    )
    ready_to_merge.sort(key=lambda x: x["number"])

    feed = {
        "schema_version": 1,
        "repo": repo,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counts": {
            "total_open_scanned": len(all_open),
            "gate_passing_candidates": len(candidates),
            "needing_tests": len(items),
            "ready_to_merge": len(ready_to_merge),
        },
        "items": items,
        "ready_to_merge": ready_to_merge,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feed, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}: {feed['counts']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministic XCL PR triage and testing feed builder")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("triage-one", help="Sync managed labels on one open PR")
    p1.add_argument("--repo", required=True)
    p1.add_argument("--pr", type=int, required=True)

    p2 = sub.add_parser("build-feed", help="Generate queue.json for gate-passing PRs needing tests")
    p2.add_argument("--repo", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--max-pages", type=int, default=20)
    p2.add_argument("--sync-labels", action="store_true")

    args = p.parse_args()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        sys.exit(2)

    gh = GitHubClient(token)
    if args.cmd == "triage-one":
        cmd_triage_one(gh, args.repo, args.pr)
    elif args.cmd == "build-feed":
        cmd_build_feed(gh, args.repo, args.out, args.max_pages, args.sync_labels)


if __name__ == "__main__":
    main()
