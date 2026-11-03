param([string]$OutDir, [string]$Seeds, [string]$Extra = "--xmono --reacquire-ahead 60 --seed-radius 2")
# Traces horizontal candidates "x,y,z;x,y,z;..." into $OutDir and copies the verticals of grid1_final.
$py = "C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
$src = "C:\Users\ChenJeff\.VC3D\afv_predictions\fibers\grid1_final"
Set-Location "C:\Users\ChenJeff\.VC3D\afv_model"
New-Item -ItemType Directory -Force $OutDir | Out-Null
Copy-Item "$src\v_*.json", "$src\v_*.path.npy" $OutDir
foreach ($s in ($Seeds -split ';' | Where-Object { $_ })) {
  $c = $s -split ','
  $n = "h_{0}_{1}_{2}" -f $c
  $args_ = @("trace_fiber_3d.py", $c[0], $c[1], $c[2], "h", "$OutDir\$n.json") + ($Extra -split ' ' | Where-Object { $_ })
  $r = (& $py @args_ 2>&1 | ForEach-Object { "$_" }) -join "`n"
  $len = if ($r -match 'traced (\d+) vx \(([\d.]+) mm\)') { "$($Matches[2]) mm" } else { "FAILED" }
  $plus = if ($r -match '\+ end at \[([\d.]+)') { $Matches[1] } else { "-" }
  $minus = if ($r -match '- end at \[([\d.]+)') { $Matches[1] } else { "-" }
  $mp = if ($r -match 'mean prob ([\d.]+)') { $Matches[1] } else { "-" }
  "{0,-20} {1,9}  x {2,7} .. {3,7}  mean prob {4}" -f $n, $len, $minus, $plus, $mp
}
