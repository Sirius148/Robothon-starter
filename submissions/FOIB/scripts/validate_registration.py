"""
Validate registration.json against the required schema.

Usage:
    python scripts/validate_registration.py                    # checks registration.json in project root
    python scripts/validate_registration.py path/to/reg.json  # explicit path

Exit codes:
    0 — valid
    1 — validation error (printed to stderr)
    2 — file not found or JSON parse error
"""
import json
import re
import sys
import os

# ---------------------------------------------------------------------------
# Required schema
# ---------------------------------------------------------------------------
_REQUIRED_STR = ["team", "track", "contact_email"]
_OPTIONAL_STR = ["repository_url", "description", "mujoco_version"]
_VALID_TRACKS = {"pick-and-place", "fragile-object", "foib-egg"}  # extend as needed
_EMAIL_RE     = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# ---------------------------------------------------------------------------

def validate(reg: dict) -> list[str]:
    """Return list of error strings; empty list = valid."""
    errors = []

    for field in _REQUIRED_STR:
        if field not in reg:
            errors.append(f"missing required field: '{field}'")
        elif not isinstance(reg[field], str) or not reg[field].strip():
            errors.append(f"'{field}' must be a non-empty string")

    if "contact_email" in reg and isinstance(reg["contact_email"], str):
        if not _EMAIL_RE.match(reg["contact_email"]):
            errors.append(f"'contact_email' does not look like an email address: {reg['contact_email']!r}")

    if "track" in reg and isinstance(reg["track"], str):
        if _VALID_TRACKS and reg["track"].lower() not in _VALID_TRACKS:
            errors.append(
                f"'track' value {reg['track']!r} not recognised; expected one of: {sorted(_VALID_TRACKS)}"
            )

    for field in _OPTIONAL_STR:
        if field in reg and not isinstance(reg[field], str):
            errors.append(f"optional field '{field}' must be a string if present")

    unknown = set(reg) - set(_REQUIRED_STR) - set(_OPTIONAL_STR)
    if unknown:
        errors.append(f"unknown fields (will be ignored by scorer): {sorted(unknown)}")

    return errors


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.path.join(os.path.dirname(__file__), "..", "registration.json")
        path = os.path.normpath(path)

    try:
        with open(path) as fh:
            reg = json.load(fh)
    except FileNotFoundError:
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        print("Create registration.json in the project root.", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        print(f"ERROR: JSON parse error in {path}: {exc}", file=sys.stderr)
        sys.exit(2)

    if not isinstance(reg, dict):
        print("ERROR: registration.json must be a JSON object ({...})", file=sys.stderr)
        sys.exit(1)

    errors = validate(reg)

    if errors:
        print(f"INVALID  {path}")
        for e in errors:
            print(f"  ✗  {e}")
        sys.exit(1)
    else:
        print(f"OK  {path}")
        print(f"    team    : {reg.get('team')}")
        print(f"    track   : {reg.get('track')}")
        print(f"    email   : {reg.get('contact_email')}")
        if reg.get("description"):
            print(f"    desc    : {reg['description'][:80]}")
        sys.exit(0)


if __name__ == "__main__":
    main()
