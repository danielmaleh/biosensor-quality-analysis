#!/usr/bin/env bash
set -euo pipefail

VERSION=${1:-}
if [[ -z "$VERSION" ]]; then
  read -rp "Enter version identifier (e.g. 42): " VERSION
fi

if [[ -z "$VERSION" ]]; then
  echo "Version identifier is required" >&2
  exit 1
fi

docker build --push -t "registry.gitlab.com/linobiotechag/playground/the-daniel-test:${VERSION}" .

COMPOSE_FILE="compose.yml"
REMOTE_HOST="txa"
REMOTE_PATH="/home/nvidia/projects/daniels-test/compose.yml"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Missing $COMPOSE_FILE in current directory" >&2
  exit 1
fi

# Replace placeholder with the provided version in a temporary compose file
tmp_compose=$(mktemp)
cleanup() { rm -f "$tmp_compose"; }
trap cleanup EXIT
sed "s|VERSION_PLACEHOLDER|${VERSION}|g" "$COMPOSE_FILE" > "$tmp_compose"

# Copy compose.yml to the remote server
scp "$tmp_compose" "${REMOTE_HOST}:${REMOTE_PATH}"

# Restart the stack remotely (ignore failure if it was not running)
ssh "$REMOTE_HOST" "cd /home/nvidia/projects/daniels-test && docker compose down || true"
ssh "$REMOTE_HOST" "cd /home/nvidia/projects/daniels-test && docker compose up -d"
