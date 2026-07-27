#!/usr/bin/env bash
# Builds docs/final_report.md into a polished .docx for submission.
#
# Strips the markdown's own leading "# Title / **Author:** / **Date:**"
# block before handing off to pandoc: that block exists so the report
# reads correctly as plain markdown on GitHub, but pandoc's -M title/
# -M author/-M date flags below already generate a proper Word title
# page from the same information, so including both produced a literal
# duplicate title block on the rendered page 1 (caught by actually
# rendering to PDF and looking at it, not assumed away).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-reports/final_report.docx}"
mkdir -p "$(dirname "$OUT")"

tail -n +6 docs/final_report.md > /tmp/_report_body.md

pandoc /tmp/_report_body.md \
  -o "$OUT" \
  --toc --toc-depth=2 \
  --resource-path=docs \
  -M title="Behavior-Aware EV Battery Health Monitoring" \
  -M subtitle="Final Report" \
  -M author="Naveen Vaidyanathan" \
  -M date="2026-07-22"

rm -f /tmp/_report_body.md
echo "Wrote $OUT"
echo "Note: the Table of Contents is a live Word field pandoc inserted un-populated"
echo "(standard pandoc behavior) -- open in Word and press F9, or right-click it and"
echo "choose 'Update Field', to populate page numbers. Headings already use proper"
echo "Word Heading styles, so Word's Navigation Pane (View > Navigation Pane) works"
echo "immediately without that step."
