param([string]$Src, [string]$Out, [int]$Rounds = 6, [string]$Extra = "", [double]$Spacing = 14)
# Iterate crossings (fibre-normal offsets) + crossing-based trimming until nothing changes.
$py = "C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
$m = "C:\Users\ChenJeff\.VC3D\afv_model"
if (Test-Path $Out) { Remove-Item $Out -Recurse -Force }
New-Item -ItemType Directory $Out | Out-Null
Copy-Item "$Src\*.path.npy", "$Src\layers.json" $Out
for ($i = 1; $i -le $Rounds; $i++) {
    & $py "$m\fiber_crossings_fn.py" $Out | Out-Null
    $tmp = "$Out`_tmp"
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    $log = & $py "$m\fiber_trim_crossings.py" $Out $tmp --ply 9 --spacing $Spacing @($Extra.Split(" ") | Where-Object { $_ }) 2>&1
    $before = (Get-ChildItem "$Out\*.path.npy" | Measure-Object Length -Sum).Sum
    $after = (Get-ChildItem "$tmp\*.path.npy" | Measure-Object Length -Sum).Sum
    Write-Output "round $i : $before -> $after bytes"
    if ($i -eq 1 -or $before -ne $after) { $log | Where-Object { $_ -notmatch "uncut|wrote" } | ForEach-Object { Write-Output "   $_" } }
    Remove-Item "$Out\*.path.npy", "$Out\*.json" -Exclude layers.json
    Copy-Item "$tmp\*.path.npy" $Out
    Remove-Item $tmp -Recurse -Force
    if ($before -eq $after) { break }
}
& $py "$m\fiber_crossings_fn.py" $Out | Out-Null
