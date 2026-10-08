#!/usr/bin/env bash
# Shared by project launchers; importing this file never starts a process.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

is_true() {
  case "${1,,}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac
}

run_python() {
  if is_true "${DRY_RUN:-false}"; then
    printf '%q ' "${PYTHON_BIN}" "$@"
    printf '\n'
  else
    "${PYTHON_BIN}" "$@"
  fi
}

require_path() {
  local name="$1" value="${!1:-}"
  if [[ -z "${value}" ]]; then
    echo "Set ${name} explicitly (see README.md)." >&2
    return 1
  fi
  if ! is_true "${DRY_RUN:-false}" && [[ ! -e "${value}" ]]; then
    echo "${name} does not exist: ${value}" >&2
    return 1
  fi
}
