#!/usr/bin/env bash
#
# Stop the four demo services.
#
#   ./stop_demo.sh           stop whatever is listening on the demo ports
#   ./stop_demo.sh --force   escalate to SIGKILL for anything that will not exit
#
# Useful when run_demo.sh was killed without running its exit trap, leaving
# services orphaned and holding their ports.

set -uo pipefail

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# name | port
SERVICES=(
  "Chat API|8000"
  "Admin API|8100"
  "Chat UI|8501"
  "Admin console|8503"
)

GRACE=10      # seconds to wait for a clean shutdown before reporting/killing

port_pids() { lsof -ti:"$1" -sTCP:LISTEN 2>/dev/null; }

printf '\nStopping services...\n'

stopped=0
skipped=0
for service in "${SERVICES[@]}"; do
  IFS='|' read -r name port <<< "$service"
  pids=$(port_pids "$port")

  if [ -z "$pids" ]; then
    printf '  not running   %s (:%s)\n' "$name" "$port"
    continue
  fi

  # Only stop processes from this project — another app may legitimately be on
  # one of these ports, and killing it would be a nasty surprise.
  for pid in $pids; do
    command=$(ps -o command= -p "$pid" 2>/dev/null)
    case "$command" in
      *hierarchical_rag_insurance*|*api.main*|*admin.api*|*ui/app.py*|*admin/ui.py*)
        kill "$pid" 2>/dev/null
        ;;
      *)
        printf '  SKIPPED       :%s held by pid %s (%s) — not one of ours\n' \
               "$port" "$pid" "$(ps -o comm= -p "$pid" 2>/dev/null)"
        skipped=$((skipped + 1))
        continue
        ;;
    esac

    # Wait for it to go, then escalate if asked.
    gone=0
    for _ in $(seq 1 $GRACE); do
      kill -0 "$pid" 2>/dev/null || { gone=1; break; }
      sleep 1
    done

    if [ "$gone" = "0" ] && [ "$FORCE" = "1" ]; then
      kill -9 "$pid" 2>/dev/null
      sleep 1
      kill -0 "$pid" 2>/dev/null || gone=1
    fi

    if [ "$gone" = "1" ]; then
      printf '  stopped       %s (:%s, pid %s)\n' "$name" "$port" "$pid"
      stopped=$((stopped + 1))
    else
      printf '  STILL RUNNING %s (:%s, pid %s) — re-run with --force\n' "$name" "$port" "$pid"
    fi
  done
done

printf '\n  %d stopped' "$stopped"
[ "$skipped" -gt 0 ] && printf ', %d skipped (not this project)' "$skipped"
printf '\n\n'

# Report anything still holding a demo port.
remaining=""
for service in "${SERVICES[@]}"; do
  IFS='|' read -r name port <<< "$service"
  [ -n "$(port_pids "$port")" ] && remaining="$remaining $port"
done
if [ -n "$remaining" ]; then
  printf '  Ports still in use:%s\n\n' "$remaining"
  exit 1
fi
