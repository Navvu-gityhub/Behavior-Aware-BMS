# Builds docs/final_report.md into a polished .docx for submission.
#
# PowerShell port of build_report_docx.sh, for Windows machines without bash.
# The two must stay in step: same tail offset, same pandoc flags, same output.
#
# Strips the markdown's own leading title/author/date block before handing off
# to pandoc, because the -M title/-M author/-M date flags below already generate
# a Word title page from the same information, and including both produced a
# duplicate title block on page 1.

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$Out = if ($args.Count -ge 1) { $args[0] } else { "reports/final_report.docx" }
$OutDir = Split-Path $Out -Parent
if ($OutDir -and -not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Force $OutDir | Out-Null
}

# The header block is 5 lines in the .sh version's `tail -n +6`. It grew when
# the revision note was added, so the boundary is found rather than hardcoded:
# everything up to and including the first horizontal rule is the header.
$lines = Get-Content docs/final_report.md
$ruleIndex = ($lines | Select-String -Pattern '^---$' | Select-Object -First 1).LineNumber
if (-not $ruleIndex) {
    throw "docs/final_report.md has no leading '---' rule to strip the header at"
}

$body = Join-Path $env:TEMP "_report_body.md"
$lines[$ruleIndex..($lines.Count - 1)] | Set-Content $body -Encoding utf8

pandoc $body `
    -o $Out `
    --toc --toc-depth=2 `
    --resource-path=docs `
    -M title="Behavior-Aware EV Battery Health Monitoring" `
    -M subtitle="Final Report" `
    -M author="Naveen Vaidyanathan" `
    -M date="2026-09-09"

Remove-Item $body -Force

Write-Output "Wrote $Out"
Write-Output ""
Write-Output "Note: the Table of Contents is a live Word field pandoc inserts un-populated"
Write-Output "(standard pandoc behaviour) -- open in Word and press F9, or right-click it and"
Write-Output "choose 'Update Field', to populate page numbers. Headings already use proper"
Write-Output "Word Heading styles, so Word's Navigation Pane (View > Navigation Pane) works"
Write-Output "immediately without that step."
