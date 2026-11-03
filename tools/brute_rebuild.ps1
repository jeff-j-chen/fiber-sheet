param([string]$Base, [string]$Surface, [string]$Out, [string]$Bottom, [string]$Top, [string]$X, [string]$Levels, [int]$Depth = 8, [string]$Extra = "")
# Brute-force new horizontals: slice-trace + strand lock at each z level, then report its crossing line.
$py = "C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
$m = "C:\Users\ChenJeff\.VC3D\afv_model"
foreach ($z in $Levels.Split(",")) {
    $o = "$Out\z$z"
    if (Test-Path $o) { Remove-Item $o -Recurse -Force }
    $r = & $py "$m\fiber_rebuild_region.py" $Base $Surface $o --bottom $Bottom --top $Top --x $X --targets "z$z" --depth $Depth @($Extra.Split(" ") | Where-Object { $_ }) 2>&1 | Select-String "strand p"
    $t = Join-Path $env:TEMP "bl_eval"
    if (Test-Path $t) { Remove-Item $t -Recurse -Force }
    New-Item -ItemType Directory $t | Out-Null
    Copy-Item "$Base\*.path.npy" $t
    Copy-Item "$o\*.path.npy" $t
    & $py "$m\fiber_crossings_fn.py" $t | Out-Null
    $name = (Get-ChildItem "$o\*.path.npy" | Select-Object -First 1).Name -replace '\.path\.npy$', ''
    $xl = & $py "$m\fiber_xline.py" "$t\crossings.csv" 9 14 h | Where-Object { $_.StartsWith($name.Substring(0, [Math]::Min(18, $name.Length))) }
    $sp = if ($r) { ($r.Line -split 'strand p ')[1].Split(',')[0] } else { "?" }
    Write-Output ("z{0} p{1} :: {2}" -f $z, $sp, (($xl -join ' ') + '                   ').Substring(19))
}
