param([string]$OutDir, [string]$Extra = "")
# Traces the grid1 seeds into $OutDir; one summary line per fibre.
$py = "C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
Set-Location "C:\Users\ChenJeff\.VC3D\afv_model"
New-Item -ItemType Directory -Force $OutDir | Out-Null
Get-ChildItem $OutDir -File | Remove-Item -Force
$seeds = @(@("v",3253,3439,12976),@("v",3631,3498,13153),@("v",3735,3456,12690),@("v",4393,3359,12670),
           @("v",5002,2937,12640),@("h",3417,3513,13514),@("h",3394,3491,13385),@("h",3377,3480,13277),
           @("h",3369,3475,13176),@("h",3289,3435,12847),@("h",3326,3423,12750),@("h",3313,3375,12532),
           @("h",3307,3343,12374),@("h",3297,3319,12216),@("h",3254,3353,12044),@("h",3224,3382,11934))
foreach ($s in $seeds) {
  $n = "{0}_{1}_{2}_{3}" -f $s
  $args_ = @("trace_fiber_3d.py", $s[1], $s[2], $s[3], $s[0], "$OutDir\$n.json", "--seed-radius", "12") + ($Extra -split ' ' | Where-Object { $_ })
  $r = (& $py @args_ 2>&1 | ForEach-Object { "$_" }) -join "`n"
  $seedDist = if ($r -match 'seed \[[^\]]*\] \(([\d.]+) vx') { $Matches[1] } else { "-" }
  $len = if ($r -match 'traced (\d+) vx \(([\d.]+) mm\)') { "$($Matches[2]) mm" } else { "FAILED" }
  "{0,-20} seed {1,5} vx from point   {2}" -f $n, $seedDist, $len
  if ($len -eq "FAILED") { ($r -split "`n" | Where-Object { $_ -match 'no . fibre|Error' }) -join ' ' }
}
