#!/usr/bin/env bash
# Measure TTFT for stealth/space-bunny-alpha. Run in a shell where your
# OpenRouter key is valid (pi clearly has one that works).
#
#   bash ~/Developer/dg-prose-eval/ttft.sh
#
# TTFT = time_starttransfer on a streaming request, i.e. time to first byte.
# Run it 3x on the same day and compare across days to separate a real
# regression from ordinary variance.
set -u
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
MODEL="${1:-stealth/space-bunny-alpha}"
N="${2:-5}"

echo "model: $MODEL"
curl -s "https://openrouter.ai/api/v1/models/$MODEL" \
  | python3 -c "import sys,json;d=json.load(sys.stdin).get('data',{});print('  listed:',d.get('id'),'| ctx:',d.get('context_length'))" \
  2>/dev/null || echo "  (model lookup failed)"

echo
for i in $(seq 1 "$N"); do
  curl -s -N -o /dev/null -X POST "https://openrouter.ai/api/v1/chat/completions" \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: ok$i\"}],\"max_tokens\":40,\"stream\":true}" \
    -w "  probe $i: TTFB %{time_starttransfer}s   total %{time_total}s\n"
  sleep 1
done

echo
echo "also check status page for provider incidents:"
echo "  https://status.openrouter.ai"
