param([string]$Dir, [string]$Out, [string]$User = "fiber_only_20261102a_user", [int]$HStep = 8)
# TPS surface from a fibre set (same flags as the grid1_final18b surface) and its dy table against the user's sheet.
$py = "C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
$P = "C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths"
$m = "C:\Users\ChenJeff\.VC3D\afv_model"
if (Test-Path "$P\$Out") { Remove-Item -Recurse -Force "$P\$Out" }
& $py "$m\fiber_surface_tps.py" $Dir "$P\$Out" --plane h --hweight 10 --hstep $HStep --no-hull --reach 1000 2>&1 | Select-Object -Last 1
& $py "$m\dy_map.py" $User $Out "$env:TEMP\dy_$Out.png" 2>&1 | Select-String -Pattern '^ *(1[0-3][0-9]{3}|dy|z\\x)' | ForEach-Object { $_.Line -replace '^( *\d+)\s+(?:\S+\s+){4}', '$1 ' }

