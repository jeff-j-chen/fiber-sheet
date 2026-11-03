param([string]$Src, [string]$Out, [string]$Names, [string]$Extra = "", [double]$Lo = 1900, [double]$Hi = 6600)
# Re-trace + extend each named fibre with fiber_march.py; one log line block per fibre.
$py = "C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
$m = "C:\Users\ChenJeff\.VC3D\afv_model\fiber_march.py"
New-Item -ItemType Directory -Force $Out | Out-Null
foreach ($n in $Names.Split(",")) {
    $p = Get-ChildItem $Src -Filter "$n*.path.npy" | Select-Object -First 1
    if (-not $p) { Write-Output "missing $n"; continue }
    $a = @($p.FullName, $Out, "--lo", $Lo, "--hi", $Hi) + ($Extra.Split(" ") | Where-Object { $_ })
    & $py $m @a 2>&1 | Where-Object { $_ -notmatch "Warning|warn" }
}
