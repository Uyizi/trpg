#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
node_version="22.14.0"
node_dir="$project_dir/.tooling/node-v${node_version}-linux-x64"
# This installer stays inside this project and makes direct connections only.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy
unset npm_config_proxy npm_config_https_proxy NPM_CONFIG_PROXY NPM_CONFIG_HTTPS_PROXY
mkdir -p "$project_dir/.tooling"
if [[ -x "$node_dir/bin/node" ]]; then
  export PATH="$node_dir/bin:$PATH"
elif command -v node >/dev/null 2>&1 && node -e 'const [a,b]=process.versions.node.split(".").map(Number);process.exit(a>22||(a===22&&b>=12)?0:1)'; then
  :
else
  if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    printf '%s\n' 'Install Node.js >=22.12 for this platform, then rerun this script.' >&2
    exit 1
  fi
  archive="node-v${node_version}-linux-x64.tar.xz"
  base="https://nodejs.org/dist/v${node_version}"
  curl -q --noproxy '*' --fail --location --connect-timeout 15 --max-time 180 "$base/$archive" --output "$project_dir/.tooling/$archive"
  curl -q --noproxy '*' --fail --location --connect-timeout 15 --max-time 60 "$base/SHASUMS256.txt" --output "$project_dir/.tooling/node-SHASUMS256.txt"
  (cd "$project_dir/.tooling" && awk -v name="$archive" '$2==name' node-SHASUMS256.txt | sha256sum --check --strict -)
  tar -xJf "$project_dir/.tooling/$archive" -C "$project_dir/.tooling"
  export PATH="$node_dir/bin:$PATH"
fi
export npm_config_registry="https://registry.npmjs.org/"
export npm_config_cache="$project_dir/.tooling/npm-cache"
export npm_config_userconfig="$project_dir/.tooling/npm-userconfig"
: > "$npm_config_userconfig"
cd "$project_dir/web"
node --version
npm --version
npm ci
npm run build
printf 'Frontend built: %s/web/dist\n' "$project_dir"
