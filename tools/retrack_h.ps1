param([string]$Dir, [string]$Out, [string]$Names, [string]$Ref = "fiber_only_20261102a_flatboi")
# Re-trace each named horizontal pinned at the crossings of the verticals in $Dir (anchor chain, track = its own path).
$py = "C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
$m = "C:\Users\ChenJeff\.VC3D\afv_model\fiber_anchor_trace.py"
New-Item -ItemType Directory -Force $Out | Out-Null
foreach ($n in $Names.Split(",")) {
    $f = Get-ChildItem "$Dir\$n*.path.npy" | Select-Object -First 1
    if (-not $f) { Write-Output "$n : not found"; continue }
    $name = $f.Name -replace '\.path\.npy$', ''
    $r = & $py $m $Dir $Out --family h --track $f.FullName --chain --lateral-from $Dir --depth 5 --ref $Ref --anchors v_ --name $name 2>&1 | Select-String "chain of|no strand" 
    Write-Output ("{0}: {1}" -f $name, (($r | ForEach-Object { $_.Line.Trim() }) -join " | "))
}
