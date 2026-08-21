#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" != "0" ]; then
  echo "Run as root: sudo bash $0" >&2
  exit 2
fi

PROJECT_DIR="/home/openclaw/projects/text-watermark-roundtrip"
GUARD_ETC="/etc/agent-api-guard"
GUARD_BIN="/usr/local/bin/guard"
CONFIG_PATH="$GUARD_ETC/config.yaml"
SECRETS_PATH="$GUARD_ETC/secrets.env"
PROFILE_PATH="$PROJECT_DIR/configs/runpod-v2-guard-profile.yaml"
MERGE_HELPER="$PROJECT_DIR/scripts/merge_runpod_v2_guard.py"
BACKUP_DIR="/root/agent-api-guard-runpod-v2-setup"
TS="$(date +%Y%m%d-%H%M%S)"
BACKUP_PATH="$BACKUP_DIR/config.yaml.before-$TS"

for path in "$CONFIG_PATH" "$SECRETS_PATH" "$PROFILE_PATH" "$MERGE_HELPER"; do
  if [ ! -f "$path" ]; then
    echo "Required file is missing: $path" >&2
    exit 2
  fi
done

if ! awk -F= '
  /^[[:space:]]*(export[[:space:]]+)?RUNPOD_API_KEY[[:space:]]*=/ { found=1 }
  END { exit(found ? 0 : 1) }
' "$SECRETS_PATH"; then
  echo "RUNPOD_API_KEY is not present in protected Guard secrets." >&2
  exit 2
fi

install -d -m 0700 "$BACKUP_DIR"
cp -p "$CONFIG_PATH" "$BACKUP_PATH"

rollback_on_error() {
  status=$?
  trap - ERR
  cp -p "$BACKUP_PATH" "$CONFIG_PATH" || true
  chown root:agent-api-guard "$CONFIG_PATH" || true
  chmod 0640 "$CONFIG_PATH" || true
  systemctl restart agent-api-guard.service >/dev/null 2>&1 || true
  exit "$status"
}
trap rollback_on_error ERR

python3 "$MERGE_HELPER" --config "$CONFIG_PATH" --profile "$PROFILE_PATH"
chown root:agent-api-guard "$CONFIG_PATH" "$SECRETS_PATH"
chmod 0640 "$CONFIG_PATH" "$SECRETS_PATH"
"$GUARD_BIN" validate --config "$CONFIG_PATH" --secrets "$SECRETS_PATH"
systemctl restart agent-api-guard.service
curl -fsS --retry 10 --retry-delay 1 --retry-connrefused \
  http://127.0.0.1:8787/healthz >/dev/null

trap - ERR
echo "RunPod v2 Pod-only Guard route is active. Backup: $BACKUP_PATH"
