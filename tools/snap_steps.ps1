param([string]$Dir, [string]$Tag, [string]$Box, [string]$Names, [string]$Sheet = "fiber_only_20261102a_flatboi", [double]$OffH = 0, [double]$OffV = -9, [int]$Depth = 6)
# Snap named fibres to a reference sheet inside a box (uvguide --whole) and write them back into the fibre dir.
$py = "C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
$F = Split-Path $Dir -Parent
$t = Join-Path $F "_w$Tag"
if (Test-Path $t) { Remove-Item -Recurse -Force $t }
& $py "C:\Users\ChenJeff\.VC3D\afv_model\fiber_uvguide.py" $Dir $Sheet $t --box $Box --off-h $OffH --off-v $OffV --depth $Depth --whole --names $Names 2>&1 | Select-Object -Last 30
Copy-Item "$t\*.path.npy" $Dir -Force
