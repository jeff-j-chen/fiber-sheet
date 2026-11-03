param([string]$Base, [string]$New, [double]$Spacing = 14)
# Copy base set + candidate fibres into a temp dir, compute crossings, print the candidates' crossing lines.
$py = "C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
$m = "C:\Users\ChenJeff\.VC3D\afv_model"
$t = Join-Path $env:TEMP "evalset"
if (Test-Path $t) { Remove-Item $t -Recurse -Force }
New-Item -ItemType Directory $t | Out-Null
Copy-Item "$Base\*.path.npy", "$Base\layers.json" $t
Copy-Item "$New\*.path.npy" $t -Force
& $py "$m\fiber_crossings_fn.py" $t | Out-Null
$new = Get-ChildItem "$New\*.path.npy" | ForEach-Object { $_.Name -replace '\.path\.npy$', '' }
foreach ($fam in "h", "v") {
    & $py "$m\fiber_xline.py" "$t\crossings.csv" 9 $Spacing $fam | Where-Object { $l = $_; $new | Where-Object { $l.StartsWith($_.Substring(0, [Math]::Min(18, $_.Length))) } }
}
