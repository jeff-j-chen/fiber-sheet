param([string]$Base, [string]$Out, [string]$Anchors, [int]$Z0, [int]$Z1, [int]$Step = 10, [string]$Family = "h")
# Brute force: pinned-chain fibres at many levels through the given anchor fibres; prints level, name, mean p, weak %.
$py = "C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
$m = "C:\Users\ChenJeff\.VC3D\afv_model\fiber_anchor_trace.py"
New-Item -ItemType Directory -Force $Out | Out-Null
for ($z = $Z0; $z -le $Z1; $z += $Step) {
    $r = & $py $m $Base "$Out\L$z" --family $Family --at $z --chain --lateral-from $Base --depth 5 --anchors $Anchors 2>&1 | Select-String "chain of"
    if ($r) { Write-Output "$z  $r" } else { Write-Output "$z  (no chain)" }
}
