param([string]$OutDir, [string]$Seeds, [string]$Family = "v", [string]$Src = "C:\Users\ChenJeff\.VC3D\afv_predictions\fibers\grid1_final2",
      [string]$Extra = "--zmono --reacquire-ahead 60 --seed-radius 2")
# Traces candidates "x,y,z;x,y,z;..." of one family into $OutDir and copies every fibre of $Src for crossing checks.
$py = "C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
Set-Location "C:\Users\ChenJeff\.VC3D\afv_model"
New-Item -ItemType Directory -Force $OutDir | Out-Null
Copy-Item "$Src\*.json", "$Src\*.path.npy" $OutDir -Exclude groups.json, layers.json
foreach ($s in ($Seeds -split ';' | Where-Object { $_ })) {
  $c = $s -split ','
  $n = "{0}_{1}_{2}_{3}" -f $Family, $c[0], $c[1], $c[2]
  $args_ = @("trace_fiber_3d.py", $c[0], $c[1], $c[2], $Family, "$OutDir\$n.json") + ($Extra -split ' ' | Where-Object { $_ })
  $r = (& $py @args_ 2>&1 | ForEach-Object { "$_" }) -join "`n"
  $len = if ($r -match 'traced (\d+) vx \(([\d.]+) mm\)') { "$($Matches[2]) mm" } else { "FAILED" }
  $mp = if ($r -match 'mean prob ([\d.]+)') { $Matches[1] } else { "-" }
  "{0,-22} {1,9}  mean prob {2}" -f $n, $len, $mp
}
