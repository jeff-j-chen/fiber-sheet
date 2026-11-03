param(
    [Parameter(Mandatory = $true)][string]$Segment,
    [string]$Output = "",
    [int]$Iterations = 50,
    [ValidateSet("symmetric_dirichlet", "conformal")][string]$Energy = "symmetric_dirichlet"
)
# Windows replacement for VC3D's SLIM-flatten: vc_tifxyz2obj -> flatboi.py (libigl SLIM) -> vc_obj2tifxyz.
$ErrorActionPreference = "Stop"
$bin    = "C:\Users\ChenJeff\.VC3D\vc3d\bin"
$python = "C:\Users\ChenJeff\.VC3D\slim_venv\Scripts\python.exe"
$slim   = "C:\Users\ChenJeff\.VC3D\slim\flatboi.py"

$Segment = (Resolve-Path $Segment).Path.TrimEnd('\')
if (-not (Test-Path "$Segment\meta.json")) { throw "Not a tifxyz folder (no meta.json): $Segment" }
if (-not $Output) { $Output = "${Segment}_flatboi" }
if (Test-Path $Output) { throw "Output already exists: $Output" }

$work = Join-Path $env:TEMP ("slim_" + (Split-Path $Segment -Leaf))
New-Item -ItemType Directory -Force $work | Out-Null
$obj = Join-Path $work "seg.obj"

Write-Host "[1/3] tifxyz -> OBJ"
& "$bin\vc_tifxyz2obj.exe" $Segment $obj
if ($LASTEXITCODE -ne 0) { throw "vc_tifxyz2obj failed ($LASTEXITCODE)" }

Write-Host "[2/3] SLIM ($Energy, $Iterations iterations)"
& $python $slim $obj $Iterations $Energy | Where-Object { $_ -notmatch '^PROGRESS' }
if ($LASTEXITCODE -ne 0) { throw "SLIM failed ($LASTEXITCODE)" }

Write-Host "[3/3] OBJ -> tifxyz"
& "$bin\vc_obj2tifxyz.exe" (Join-Path $work "seg_flatboi.obj") $Output "--tifxyz-source=$Segment"
if ($LASTEXITCODE -ne 0) { throw "vc_obj2tifxyz failed ($LASTEXITCODE)" }

# vc_obj2tifxyz writes a GUID uuid and the source's seed_surface_id; both must equal the folder name.
$metaPath = Join-Path $Output "meta.json"
$name = Split-Path $Output -Leaf
$meta = [System.IO.File]::ReadAllText($metaPath)
$meta = [regex]::Replace($meta, '("uuid"\s*:\s*)"[^"]*"', ('${1}"' + $name + '"'))
$meta = [regex]::Replace($meta, '("seed_surface_id"\s*:\s*)"[^"]*"', ('${1}"' + $name + '"'))
[System.IO.File]::WriteAllText($metaPath, $meta, (New-Object System.Text.UTF8Encoding($false)))

Remove-Item $work -Recurse -Force
Write-Host "Done: $Output"
