#!/usr/bin/env python3
"""
Octopus Deploy API key validator for octopus_rbac_report.py
===========================================================

Checks whether an API key can actually see everything octopus_rbac_report.py
needs, and explains *why* when it can't. It exercises the same endpoints the
report uses, but unlike the report it never hides a failure: every 401, 403,
404, empty result, or non-JSON response is shown with the server's own error
message.

Why this exists: the report script treats several failures as "no data"
instead of errors, so a bad key or bad URL can produce an empty report with
no warning at all:

  * Any 404 returns None -> get_all() returns [] (e.g. a wrong --url)
  * /api/users/{id}/teams errors are swallowed -> users end up with no teams
  * Per-space project/environment/tenant lookups are swallowed on any error
  * Octopus filters many collections by permission and returns 200 with zero
    items rather than 403

Read-only: GET requests only. Standard library only, no pip install.

Usage:
  export OCTOPUS_API_KEY="API-XXXX"
  python3 validate_octopus_api_key.py --url https://myinstance.octopus.app
  python3 validate_octopus_api_key.py --url ... --spaces "Retail,Finance" --full
  python3 validate_octopus_api_key.py --url ... --json-out key-check.json

Exit code: 0 = no FAIL results, 1 = at least one FAIL, 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

VERSION = "1.0.0"

# What the report needs, and which of its API calls each permission feeds.
REQUIRED_SYSTEM_PERMISSIONS: Dict[str, str] = {
    "UserView": "/api/users and /api/users/{id}/teams",
    "TeamView": "/api/teams (system teams) and /api/teams/{id}/scopeduserroles",
    "UserRoleView": "/api/userroles",
    "SpaceView": "/api/spaces (without it, only spaces you belong to are listed)",
}
REQUIRED_SPACE_PERMISSIONS: Dict[str, str] = {
    "TeamView": "space teams in /api/teams and their /scopeduserroles",
    "ProjectView": "/api/{space}/projects (scope-name resolution)",
    "ProjectGroupView": "/api/{space}/projectgroups (scope-name resolution)",
    "EnvironmentView": "/api/{space}/environments (scope-name resolution)",
    "TenantView": "/api/{space}/tenants (scope-name resolution)",
}

PASS, WARN, FAIL, INFO, SKIP = "PASS", "WARN", "FAIL", "INFO", "SKIP"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

@dataclass
class Response:
    url: str
    status: Optional[int]          # None = connection failure
    body: Any = None               # parsed JSON, or None
    raw_text: str = ""             # first part of a non-JSON body
    error: str = ""                # Octopus ErrorMessage or transport error

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300 and self.body is not None

    @property
    def is_json(self) -> bool:
        return self.body is not None


class Client:
    def __init__(self, base_url: str, api_key: str, timeout: int, verbose: bool):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.verbose = verbose
        self.calls = 0

    def get(self, path: str) -> Response:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="GET", headers={
            "X-Octopus-ApiKey": self.api_key,
            "Accept": "application/json",
            "User-Agent": f"octopus-api-key-validator/{VERSION}",
        })
        self.calls += 1
        if self.verbose:
            print(f"  GET {url}", file=sys.stderr)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return self._parse(url, resp.status, resp.read())
        except urllib.error.HTTPError as exc:
            return self._parse(url, exc.code, exc.read() or b"")
        except urllib.error.URLError as exc:
            return Response(url, None, error=f"Connection error: {exc.reason}")
        except Exception as exc:  # timeouts, SSL, etc.
            return Response(url, None, error=f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _parse(url: str, status: int, raw: bytes) -> Response:
        text = raw.decode("utf-8", errors="replace")
        try:
            body = json.loads(text) if text.strip() else None
        except json.JSONDecodeError:
            return Response(url, status, None, raw_text=text[:200].strip(),
                            error="Response was not JSON (wrong URL? got an HTML page or proxy response)")
        error = ""
        if isinstance(body, dict) and status >= 400:
            error = body.get("ErrorMessage") or ""
            extra = body.get("Errors") or []
            if extra:
                error = f"{error} | {'; '.join(str(e) for e in extra)}"
        return Response(url, status, body if status < 400 else None, error=error)


def count_items(body: Any) -> Tuple[int, Optional[int]]:
    """Returns (items_on_this_page, TotalResults_if_reported)."""
    if isinstance(body, list):
        return len(body), len(body)
    if isinstance(body, dict):
        items = body.get("Items") or []
        return len(items), body.get("TotalResults")
    return 0, None


def get_all(client: Client, path: str, page: int = 1000) -> Tuple[Response, List[dict]]:
    """Paged fetch; returns the first error Response if any page fails."""
    items: List[dict] = []
    skip = 0
    sep = "&" if "?" in path else "?"
    while True:
        resp = client.get(f"{path}{sep}skip={skip}&take={page}")
        if not resp.ok:
            return resp, items
        if isinstance(resp.body, list):
            items.extend(resp.body)
            return resp, items
        page_items = resp.body.get("Items") or []
        items.extend(page_items)
        total = resp.body.get("TotalResults")
        if total is None or len(page_items) < page or len(items) >= total:
            return resp, items
        skip += page


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class Check:
    section: str
    name: str
    result: str
    detail: str = ""
    endpoint: str = ""
    http_status: Optional[int] = None
    hint: str = ""


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)

    def add(self, *args, **kwargs) -> Check:
        c = Check(*args, **kwargs)
        self.checks.append(c)
        return c

    def counts(self) -> Dict[str, int]:
        out = {k: 0 for k in (PASS, WARN, FAIL, INFO, SKIP)}
        for c in self.checks:
            out[c.result] += 1
        return out


def describe_failure(resp: Response) -> Tuple[str, str]:
    """Turns a failed Response into (detail, hint)."""
    if resp.status is None:
        return resp.error, "Check --url, DNS, VPN/proxy, and TLS."
    if resp.status == 401:
        return f"401 {resp.error}".strip(), "API key is wrong, expired, or revoked."
    if resp.status == 403:
        return f"403 {resp.error}".strip(), "The key's user lacks a permission this call needs (see message)."
    if resp.status == 404:
        return ("404 Not Found",
                "The report script silently turns 404 into an EMPTY result. "
                "Usually a wrong --url (it must be the server root, e.g. https://x.octopus.app, "
                "with no /app or /api suffix) or an endpoint this server version doesn't have.")
    if not resp.is_json and resp.raw_text:
        return f"{resp.status}: {resp.error}. Body starts: {resp.raw_text[:80]!r}", "Check --url."
    return f"HTTP {resp.status} {resp.error}".strip(), ""


def probe_collection(report: Report, client: Client, section: str, name: str, path: str,
                     expect_nonempty: bool, empty_hint: str = "") -> Optional[Response]:
    resp = client.get(path)
    if not resp.ok:
        detail, hint = describe_failure(resp)
        report.add(section, name, FAIL, detail, path, resp.status, hint)
        return None
    shown, total = count_items(resp.body)
    n = total if total is not None else shown
    if n == 0 and expect_nonempty:
        report.add(section, name, FAIL, "200 OK but 0 items returned", path, resp.status,
                   empty_hint or "Octopus filters this collection by permission; the key's user can't see any.")
    else:
        report.add(section, name, PASS, f"{n} item(s) visible", path, resp.status)
    return resp


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_url_shape(report: Report, url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        report.add("Connectivity", "URL format", FAIL, f"Unexpected scheme in {url!r}",
                   hint="Use https://your-instance.octopus.app")
    elif parsed.path not in ("", "/"):
        report.add("Connectivity", "URL format", WARN, f"URL has a path component: {parsed.path!r}",
                   hint="Only valid if Octopus is hosted under a virtual directory. "
                        "Copying a browser URL like .../app#/Spaces-1 here breaks every API call.")
    else:
        report.add("Connectivity", "URL format", PASS, url)


def check_server(report: Report, client: Client) -> bool:
    resp = client.get("/api")
    if not resp.ok:
        detail, hint = describe_failure(resp)
        report.add("Connectivity", "Octopus API root", FAIL, detail, "/api", resp.status, hint)
        return False
    body = resp.body if isinstance(resp.body, dict) else {}
    app = body.get("Application", "?")
    ver = body.get("Version", "?")
    result = PASS if "octopus" in str(app).lower() else WARN
    report.add("Connectivity", "Octopus API root", result, f"{app} {ver}", "/api", resp.status,
               "" if result == PASS else "Server responded but doesn't identify as Octopus Deploy.")
    return True


def check_identity(report: Report, client: Client) -> Optional[dict]:
    resp = client.get("/api/users/me")
    if not resp.ok:
        detail, hint = describe_failure(resp)
        report.add("Identity", "Key owner (/api/users/me)", FAIL, detail, "/api/users/me", resp.status, hint)
        return None
    me = resp.body
    flags = []
    if me.get("IsService"):
        flags.append("service account")
    if not me.get("IsActive", True):
        flags.append("INACTIVE")
    desc = f"{me.get('DisplayName') or ''} ({me.get('Username')}, {me.get('Id')})"
    if flags:
        desc += " [" + ", ".join(flags) + "]"
    result = FAIL if not me.get("IsActive", True) else PASS
    report.add("Identity", "Key owner (/api/users/me)", result, desc, "/api/users/me", resp.status,
               "Inactive users can't use the API." if result == FAIL else "")
    return me


def check_permissions(report: Report, client: Client, me: dict,
                      spaces: List[dict]) -> Optional[dict]:
    """Uses /api/users/{id}/permissions to compare effective permissions with
    what the report needs. Returns the raw permission set (or None)."""
    path = f"/api/users/{me['Id']}/permissions"
    resp = client.get(path)
    if not resp.ok:
        detail, hint = describe_failure(resp)
        report.add("Effective permissions", "Read own permission set", WARN, detail, path, resp.status,
                   hint + " Falling back to endpoint probes only.")
        return None

    perms = resp.body
    system_perms = set(perms.get("SystemPermissions") or [])
    space_perms: Dict[str, List[dict]] = perms.get("SpacePermissions") or {}
    is_admin = "AdministerSystem" in system_perms

    teams = perms.get("Teams") or []
    team_names = ", ".join(sorted(t.get("Name", "?") for t in teams)) or "none"
    report.add("Effective permissions", "Teams the key's user is in", INFO, team_names, path)

    if is_admin:
        report.add("Effective permissions", "AdministerSystem", PASS,
                   "Key has AdministerSystem (implies every system permission)", path)

    for perm, used_for in REQUIRED_SYSTEM_PERMISSIONS.items():
        if is_admin or perm in system_perms:
            report.add("Effective permissions", f"System: {perm}", PASS, f"needed for {used_for}", path)
        else:
            report.add("Effective permissions", f"System: {perm}", FAIL, f"MISSING - needed for {used_for}",
                       path, hint=f"Grant a system role containing {perm} (e.g. System Manager) to a "
                                  f"system team the key's user belongs to.")

    # Space-level: Octopus reports each space permission as a list of grants,
    # each with a SpaceId and optional RestrictedTo* lists.
    for space in spaces:
        sid, sname = space["Id"], space["Name"]
        missing, restricted = [], []
        for perm in REQUIRED_SPACE_PERMISSIONS:
            grants = [g for g in space_perms.get(perm, []) if g.get("SpaceId") in (sid, None)]
            if not grants:
                missing.append(perm)
                continue
            unrestricted = any(
                not any(v for k, v in g.items() if k.startswith("RestrictedTo")) for g in grants
            )
            if not unrestricted:
                restricted.append(perm)
        if missing:
            report.add("Effective permissions", f"Space: {sname}", WARN if is_admin else FAIL,
                       f"missing {', '.join(missing)}", path,
                       hint="System Manager has NO space-level permissions. Add the key's user to a team "
                            "with Space Manager (or a custom read-only role) in this space, or the report "
                            "will miss this space's teams/role grants and show raw IDs for scopes.")
        elif restricted:
            report.add("Effective permissions", f"Space: {sname}", WARN,
                       f"{', '.join(restricted)} granted only with project/environment/tenant restrictions",
                       path, hint="Scope-name resolution will be partial; raw IDs will appear for the rest.")
        else:
            report.add("Effective permissions", f"Space: {sname}", PASS,
                       "all required space permissions, unrestricted", path)

    if perms.get("IsPermissionsComplete") is False:
        report.add("Effective permissions", "Permission set completeness", WARN,
                   "Server reports IsPermissionsComplete=false (list may be truncated)", path)
    return perms


def check_endpoints(report: Report, client: Client, spaces: List[dict], users: List[dict],
                    full: bool, sample: int) -> None:
    sec = "Report API calls"

    # 1. The account-wide collections the report fetches first.
    probe_collection(report, client, sec, "Users", "/api/users?skip=0&take=1", True,
                     "Needs system UserView.")
    probe_collection(report, client, sec, "User roles", "/api/userroles?skip=0&take=1", True,
                     "Built-in roles always exist, so 0 means the key can't see them (UserRoleView).")
    probe_collection(report, client, sec, "Spaces", "/api/spaces?skip=0&take=1", True,
                     "Key's user can see no spaces.")

    # 2. Teams exactly as the report calls it, vs. explicit space + system query.
    resp_plain, teams_plain = get_all(client, "/api/teams")
    if not resp_plain.ok:
        detail, hint = describe_failure(resp_plain)
        report.add(sec, "Teams (as the report calls it)", FAIL, detail, "/api/teams", resp_plain.status, hint)
    else:
        result = PASS if teams_plain else FAIL
        report.add(sec, "Teams (as the report calls it)", result, f"{len(teams_plain)} team(s) visible",
                   "/api/teams", resp_plain.status,
                   "" if teams_plain else "Every instance has 'Everyone'; 0 means no TeamView.")

    teams_explicit: List[dict] = []
    if spaces:
        ids = ",".join(s["Id"] for s in spaces)
        path = f"/api/teams?spaces={urllib.parse.quote(ids)}&includeSystem=true"
        resp_exp, teams_explicit = get_all(client, path)
        if resp_exp.ok and resp_plain.ok:
            plain_ids = {t["Id"] for t in teams_plain}
            extra = [t for t in teams_explicit if t["Id"] not in plain_ids]
            if extra:
                report.add(sec, "Teams: plain vs. spaces+includeSystem", WARN,
                           f"/api/teams returned {len(teams_plain)}, explicit query returned "
                           f"{len(teams_explicit)}; {len(extra)} team(s) the report would miss "
                           f"(e.g. {', '.join(t.get('Name', '?') for t in extra[:3])})",
                           path, resp_exp.status,
                           hint="Script issue, not a key issue: change fetch_reference_data() to call "
                                "/api/teams?spaces=<all ids>&includeSystem=true.")
            else:
                report.add(sec, "Teams: plain vs. spaces+includeSystem", PASS,
                           f"both return {len(teams_explicit)} team(s)", path, resp_exp.status)

    # 3. Scoped role grants per team (the report calls this for every team).
    teams = teams_explicit or teams_plain
    targets = teams if full else teams[:sample]
    failures, empty = [], 0
    for t in targets:
        path = f"/api/teams/{t['Id']}/scopeduserroles?skip=0&take=1"
        r = client.get(path)
        if not r.ok:
            failures.append((t, r))
        elif count_items(r.body)[1] in (0, None) and count_items(r.body)[0] == 0:
            empty += 1
    label = f"Scoped role grants ({len(targets)} of {len(teams)} teams checked)"
    if not targets:
        report.add(sec, label, SKIP, "no teams visible to test")
    elif failures:
        t, r = failures[0]
        detail, hint = describe_failure(r)
        report.add(sec, label, FAIL, f"{len(failures)} team(s) failed; first: {t.get('Name')} -> {detail}",
                   r.url.replace(client.base_url, ""), r.status,
                   hint + (" A 404 here is silently treated as 'no roles' by the report." if r.status == 404 else ""))
    elif empty == len(targets):
        report.add(sec, label, WARN, "every checked team has 0 role grants",
                   hint="Possible but unusual; can indicate grants are filtered by permission.")
    else:
        report.add(sec, label, PASS, f"{len(targets) - empty} with grants, {empty} without")

    # 4. Per-space name lookups (the report swallows ALL errors here).
    for space in spaces:
        sid, sname = space["Id"], space["Name"]
        problems, counts = [], []
        for coll in ("projects", "projectgroups", "environments", "tenants"):
            path = f"/api/{sid}/{coll}?skip=0&take=1"
            r = client.get(path)
            if not r.ok:
                problems.append(f"{coll}: {describe_failure(r)[0]}")
            else:
                shown, total = count_items(r.body)
                counts.append(f"{coll}={total if total is not None else shown}")
        if problems:
            report.add(sec, f"Scope lookups: {sname}", WARN, "; ".join(problems),
                       f"/api/{sid}/...", hint="Report hides these errors and falls back to raw IDs.")
        else:
            report.add(sec, f"Scope lookups: {sname}", PASS, ", ".join(counts), f"/api/{sid}/...")

    # 5. User -> team resolution (the report swallows ALL errors here, which
    #    is what makes users show up with no teams/roles).
    humans = [u for u in users if not u.get("IsService")][:sample if not full else None]
    if not humans or not spaces:
        report.add(sec, "User team resolution", SKIP, "no users or spaces visible to test")
        return
    pairs = [(u, s) for u in humans for s in (spaces if full else spaces[:3])]
    bad, zero = [], 0
    for u, s in pairs:
        path = f"/api/users/{u['Id']}/teams?spaces={s['Id']}&includeSystem=True"
        r = client.get(path)
        if not r.ok:
            bad.append((u, s, r))
        elif count_items(r.body)[0] == 0:
            zero += 1
    label = f"User team resolution ({len(pairs)} user/space pairs checked)"
    if bad:
        u, s, r = bad[0]
        detail, hint = describe_failure(r)
        report.add(sec, label, FAIL,
                   f"{len(bad)} pair(s) failed; first: {u.get('Username')} in {s['Name']} -> {detail}",
                   r.url.replace(client.base_url, ""), r.status,
                   hint + " The report swallows this error, so users appear with no teams or roles.")
    elif zero == len(pairs):
        report.add(sec, label, WARN, "every pair resolved to 0 teams",
                   hint="Every user should at least be in 'Everyone' when includeSystem=True; "
                        "this suggests the key can't see team membership for other users.")
    else:
        report.add(sec, label, PASS, f"{len(pairs) - zero} pair(s) with teams, {zero} with none")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_report(report: Report, client: Client) -> None:
    colors = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    palette = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", INFO: "\033[36m", SKIP: "\033[90m"}
    reset = "\033[0m" if colors else ""

    section = None
    for c in report.checks:
        if c.section != section:
            section = c.section
            print(f"\n== {section} ==")
        tag = f"{palette[c.result] if colors else ''}[{c.result}]{reset}"
        print(f"  {tag} {c.name}: {c.detail}")
        if c.endpoint and c.result in (FAIL, WARN):
            status = f" (HTTP {c.http_status})" if c.http_status else ""
            print(f"         endpoint: {c.endpoint}{status}")
        if c.hint and c.result in (FAIL, WARN):
            print(f"         fix: {c.hint}")

    n = report.counts()
    print(f"\nSummary: {n[PASS]} pass, {n[WARN]} warn, {n[FAIL]} fail, {n[SKIP]} skipped "
          f"({client.calls} API calls)")
    if n[FAIL]:
        print("Verdict: this key will NOT produce a complete report. Fix the FAIL items above.")
    elif n[WARN]:
        print("Verdict: the report will run, but some data will be missing or show raw IDs (see WARN).")
    else:
        print("Verdict: this key has everything octopus_rbac_report.py needs.")


def mask(key: str) -> str:
    return key[:4] + "*" * max(0, len(key) - 8) + key[-4:] if len(key) > 8 else "****"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Validate an Octopus API key for octopus_rbac_report.py")
    p.add_argument("--url", required=True, help="Octopus Server root URL, e.g. https://myinstance.octopus.app")
    p.add_argument("--api-key", help="API key (falls back to OCTOPUS_API_KEY)")
    p.add_argument("--spaces", help="Same as the report's --spaces: comma-separated names or IDs")
    p.add_argument("--sample", type=int, default=5,
                   help="How many teams/users to probe for per-item calls (default 5)")
    p.add_argument("--full", action="store_true", help="Probe every team, user and space (slower)")
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--json-out", help="Also write results to this JSON file")
    p.add_argument("--verbose", action="store_true", help="Print each request URL to stderr")
    args = p.parse_args(argv)

    api_key = args.api_key or os.environ.get("OCTOPUS_API_KEY")
    if not api_key:
        print("error: provide --api-key or set OCTOPUS_API_KEY", file=sys.stderr)
        return 2

    print(f"Validating key {mask(api_key)} against {args.url}")
    client = Client(args.url, api_key, args.timeout, args.verbose)
    report = Report()

    check_url_shape(report, args.url)
    if not check_server(report, client):
        print_report(report, client)
        return 1

    me = check_identity(report, client)
    if me is None:
        print_report(report, client)
        return 1

    # Spaces, filtered the same way the report filters them.
    resp, all_spaces = get_all(client, "/api/spaces")
    spaces = all_spaces
    if args.spaces:
        wanted = {s.strip().lower() for s in args.spaces.split(",") if s.strip()}
        spaces = [s for s in all_spaces if s["Id"].lower() in wanted or s["Name"].lower() in wanted]
        unmatched = wanted - {s["Id"].lower() for s in spaces} - {s["Name"].lower() for s in spaces}
        if not spaces:
            report.add("Spaces", "--spaces filter", FAIL,
                       f"matched none of {len(all_spaces)} visible space(s): {args.spaces}",
                       hint="The report would run with zero spaces, so every user gets no teams/roles. "
                            "Names must match exactly (case-insensitive).")
        elif unmatched:
            report.add("Spaces", "--spaces filter", WARN, f"no visible space named: {', '.join(sorted(unmatched))}")
        else:
            report.add("Spaces", "--spaces filter", PASS, f"{len(spaces)} space(s) selected")
    if resp.ok:
        report.add("Spaces", "Visible spaces", INFO,
                   ", ".join(f"{s['Name']} ({s['Id']})" for s in all_spaces) or "none")

    check_permissions(report, client, me, spaces)

    _, users = get_all(client, "/api/users")
    check_endpoints(report, client, spaces, users, args.full, args.sample)

    print_report(report, client)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"url": args.url, "key_owner": me.get("Username"),
                       "summary": report.counts(), "checks": [asdict(c) for c in report.checks]}, f, indent=2)
        print(f"Wrote {args.json_out}")

    return 1 if report.counts()[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
