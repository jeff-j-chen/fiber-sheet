param([string]$OutDir, [string]$Extra = "--xmono --reacquire-ahead 60")
# Traces all horizontals (grid1 seeds + the two replacement seeds) into $OutDir and copies the verticals.
$py = "C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
$src = "C:\Users\ChenJeff\.VC3D\afv_predictions\fibers\grid1_conservative"
Set-Location "C:\Users\ChenJeff\.VC3D\afv_model"
New-Item -ItemType Directory -Force $OutDir | Out-Null
Copy-Item "$src\v_*.json", "$src\v_*.path.npy" $OutDir
$seeds = @(@(3417,3513,13514,12),@(3394,3491,13385,12),@(3377,3480,13277,12),@(3369,3475,13176,12),
           @(3289,3435,12847,12),@(3326,3423,12750,12),@(3313,3375,12532,12),@(3307,3343,12374,12),
           @(3297,3319,12216,12),@(3254,3353,12044,12),@(3224,3382,11934,12),@(3458,3391,11931,3),
           @(3481,3393,12050,3))
foreach ($s in $seeds) {
  $n = "h_{0}_{1}_{2}" -f $s
  $args_ = @("trace_fiber_3d.py", $s[0], $s[1], $s[2], "h", "$OutDir\$n.json", "--seed-radius", $s[3]) + ($Extra -split ' ' | Where-Object { $_ })
  $r = (& $py @args_ 2>&1 | ForEach-Object { "$_" }) -join "`n"
  $len = if ($r -match 'traced (\d+) vx \(([\d.]+) mm\)') { "$($Matches[2]) mm" } else { "FAILED" }
  $plus = if ($r -match '\+ end at \[([\d.]+)') { $Matches[1] } else { "-" }
  $minus = if ($r -match '- end at \[([\d.]+)') { $Matches[1] } else { "-" }
  $mp = if ($r -match 'mean prob ([\d.]+)') { $Matches[1] } else { "-" }
  "{0,-20} {1,9}  x {2,7} .. {3,7}  mean prob {4}" -f $n, $len, $minus, $plus, $mp
}
