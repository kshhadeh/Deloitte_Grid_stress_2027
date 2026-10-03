# Rebuild the large data files stored in this repo as split zip parts (<name>.zip.001, .002, ...).
# Run once after cloning:  powershell -ExecutionPolicy Bypass -File .\restore_large_files.ps1
# Add -DeleteParts to remove the parts afterwards.
param([switch]$DeleteParts)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$groups = Get-ChildItem $root -Recurse -File -Filter "*.zip.*" |
    Where-Object { $_.Name -match '\.zip\.\d{3}$' } |
    Group-Object { $_.FullName.Substring(0, $_.FullName.Length - 4) }
foreach ($g in $groups) {
    $zip = $g.Name
    $out = [System.IO.File]::Create($zip)
    foreach ($p in ($g.Group | Sort-Object Name)) {
        $bytes = [System.IO.File]::ReadAllBytes($p.FullName); $out.Write($bytes, 0, $bytes.Length)
    }
    $out.Close()
    Expand-Archive -Path $zip -DestinationPath (Split-Path $zip) -Force
    Remove-Item $zip
    if ($DeleteParts) { $g.Group | Remove-Item }
    Write-Host "restored $($zip.Substring($root.Length + 1, $zip.Length - $root.Length - 5))"
}
Write-Host "Done: $($groups.Count) file(s) restored."
