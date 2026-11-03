param([string]$Base, [string]$Out, [int]$Z0, [int]$Z1, [int]$Step = 20, [string]$Ref = "fiber_only_20261102a_flatboi", [string]$Anchors = "v_")
# Anchor-chain horizontals at many z levels through the verticals (ref sheet normals); prints level, chain summary.
$py = "C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
$m = "C:\Users\ChenJeff\.VC3D\afv_model\fiber_anchor_trace.py"
New-Item -ItemType Directory -Force $Out | Out-Null
for ($z = $Z0; $z -le $Z1; $z += $Step) {
    $r = & $py $m $Base "$Out\L$z" --family h --at $z --chain --lateral-from $Base --depth 5 --ref $Ref --anchors $Anchors 2>&1 | Select-String "chain of"
    if ($r) { Write-Output "$z  $r" } else { Write-Output "$z  (no chain)" }
}
