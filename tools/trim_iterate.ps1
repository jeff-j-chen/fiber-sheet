param([string]$Dir, [string]$Segment, [int]$Rounds = 3)
# Alternately trims horizontals and verticals of $Dir (in place) to their crossing-confirmed stretches.
$py = "C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
Set-Location "C:\Users\ChenJeff\.VC3D\afv_model"
for ($i = 1; $i -le $Rounds; $i++) {
  foreach ($fam in "h", "v") {
    & $py fiber_crossings.py $Dir $Segment --xmax 6700 | Out-Null
    $out = & $py fiber_trim_crossings.py $Dir $Dir --family $fam 2>&1 | ForEach-Object { "$_" }
    if ($i -eq $Rounds) { "--- round $i, $fam"; $out | Where-Object { $_ -notmatch '^wrote' } }
  }
}
& $py fiber_crossings.py $Dir $Segment --xmax 6700 | Out-Null
