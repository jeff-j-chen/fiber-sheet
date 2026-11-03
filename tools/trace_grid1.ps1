param([string]$Segment = "C:\Users\ChenJeff\.VC3D\remote_cache\open_data\projects\paths\auto_grown_20261012065803387_straightened2_abf")
# Traces the requested vertical/horizontal seeds with the extended (re-acquiring) settings.
$py  = "C:\Users\ChenJeff\.VC3D\afv_venv\Scripts\python.exe"
$out = "C:\Users\ChenJeff\.VC3D\afv_predictions\fibers\grid1"
New-Item -ItemType Directory -Force $out | Out-Null
Set-Location "C:\Users\ChenJeff\.VC3D\afv_model"
$seeds = @(
  @("v", 3253, 3439, 12976), @("v", 3631, 3498, 13153), @("v", 3735, 3456, 12690),
  @("v", 4393, 3359, 12670), @("v", 5002, 2937, 12640),
  @("h", 3417, 3513, 13514), @("h", 3394, 3491, 13385), @("h", 3377, 3480, 13277),
  @("h", 3369, 3475, 13176), @("h", 3289, 3435, 12847), @("h", 3326, 3423, 12750),
  @("h", 3313, 3375, 12532), @("h", 3307, 3343, 12374), @("h", 3297, 3319, 12216),
  @("h", 3254, 3353, 12044), @("h", 3224, 3382, 11934)
)
foreach ($s in $seeds) {
  $name = "{0}_{1}_{2}_{3}" -f $s[0], $s[1], $s[2], $s[3]
  "===== $name =====" | Add-Content "$out\trace.log"
  & $py trace_fiber_3d.py $s[1] $s[2] $s[3] $s[0] "$out\$name.json" --segment $Segment --reacquire-ahead 200 *>> "$out\trace.log"
}
"ALL DONE" | Add-Content "$out\trace.log"
