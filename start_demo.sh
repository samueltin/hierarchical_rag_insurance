#!/usr/bin/env bash
#
# Bring up all four services for a demo, wait until each is actually serving,
# then print the URLs to paste into a browser.
#
#   ./run_demo.sh            start everything, Ctrl-C to stop
#   ./run_demo.sh --force    kill whatever is holding the ports first
#
# Logs go to .logs/<service>.log. Every service is stopped on exit.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv/bin"
LOGS="$ROOT/.logs"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# name | port | health path | command
SERVICES=(
  "Chat API|8000|/health|$VENV/uvicorn api.main:app --port 8000 --log-level warning"
  "Admin API|8100|/health|$VENV/uvicorn admin.api:app --port 8100 --log-level warning"
  "Chat UI|8501|/_stcore/health|$VENV/streamlit run ui/app.py --server.port 8501 --server.headless true"
  "Admin console|8503|/_stcore/health|$VENV/streamlit run admin/ui.py --server.port 8503 --server.headless true"
)

PIDS=""

# ---------------------------------------------------------------------------

fail() { printf '\n  %s\n\n' "$1" >&2; exit 1; }

shutdown() {
  trap - INT TERM EXIT
  [ -n "$PIDS" ] && { printf '\nStopping services...\n'; kill $PIDS 2>/dev/null; wait $PIDS 2>/dev/null; }
  printf 'Stopped.\n'
  exit 0
}

port_pid() { lsof -ti:"$1" -sTCP:LISTEN 2>/dev/null | head -1; }

wait_until_serving() {   # name, port, path, pid
  local name=$1 port=$2 path=$3 pid=$4 i
  for i in $(seq 1 120); do
    if curl -fsS -o /dev/null --max-time 3 "http://localhost:$port$path" 2>/dev/null; then
      printf '  ready   %s (:%s)\n' "$name" "$port"
      return 0
    fi
    # A service that has already exited will never become ready.
    if ! kill -0 "$pid" 2>/dev/null; then
      printf '  FAILED  %s (:%s) exited during startup\n' "$name" "$port"
      printf '          last lines of %s:\n' "$LOGS/$(slug "$name").log"
      sed 's/^/          /' "$LOGS/$(slug "$name").log" | tail -12
      return 1
    fi
    sleep 1
  done
  printf '  TIMEOUT %s (:%s) did not respond within 120s\n' "$name" "$port"
  return 1
}

slug() { printf '%s' "$1" | tr '[:upper:] ' '[:lower:]-'; }

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

[ -x "$VENV/uvicorn" ] || fail "No virtualenv at .venv — run: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
[ -f "$ROOT/.env" ]    || fail "No .env file — copy env.example to .env and fill in your Azure settings."

for required in AZURE_OPENAI_ENDPOINT AZURE_SEARCH_ENDPOINT BLOB_CONNECTION_STRING CHAT_DEPLOYMENT EMBEDDING_DEPLOYMENT; do
  grep -qE "^${required}=.+" "$ROOT/.env" || printf '  warning: %s is not set in .env\n' "$required"
done
grep -qE "^AZURE_CONTENT_SAFETY_ENDPOINT=.+" "$ROOT/.env" \
  || printf '  note: AZURE_CONTENT_SAFETY_ENDPOINT not set — groundedness checks will be skipped\n'

for service in "${SERVICES[@]}"; do
  IFS='|' read -r name port path command <<< "$service"
  existing=$(port_pid "$port")
  if [ -n "$existing" ]; then
    if [ "$FORCE" = "1" ]; then
      printf '  freeing port %s (pid %s)\n' "$port" "$existing"
      kill "$existing" 2>/dev/null
      sleep 1
    else
      fail "Port $port is in use by pid $existing ($(ps -o comm= -p "$existing")).
  Re-run with --force to stop it, or: kill $existing"
    fi
  fi
done

mkdir -p "$LOGS"
trap shutdown INT TERM EXIT

# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------

printf '\nStarting services...\n'
for service in "${SERVICES[@]}"; do
  IFS='|' read -r name port path command <<< "$service"
  log="$LOGS/$(slug "$name").log"
  : > "$log"
  $command >> "$log" 2>&1 &
  pid=$!
  PIDS="$PIDS $pid"
  printf '  started %s (:%s, pid %s)\n' "$name" "$port" "$pid"
done

printf '\nWaiting for services to respond...\n'
index=0
for service in "${SERVICES[@]}"; do
  IFS='|' read -r name port path command <<< "$service"
  index=$((index + 1))
  pid=$(printf '%s' "$PIDS" | awk -v n="$index" '{print $n}')
  wait_until_serving "$name" "$port" "$path" "$pid" || fail "Startup failed — see $LOGS/"
done

# ---------------------------------------------------------------------------
# Ready
# ---------------------------------------------------------------------------

cat <<BANNER

  ─────────────────────────────────────────────
   All 4 services are up.

     Chat UI          http://localhost:8501
     Admin console    http://localhost:8503

     Chat API docs    http://localhost:8000/docs
     Admin API docs   http://localhost:8100/docs
  ─────────────────────────────────────────────

  Logs: .logs/          Press Ctrl-C to stop everything.

BANNER

wait
