# Build CoSirca desktop (release). Usage:  .\build.ps1        (from the desktop folder)
# Needs Rust (rustup.rs). With the GNU toolchain, MinGW binutils (dlltool) must be on PATH;
# WinLibs from winget provides them:  winget install BrechtSanders.WinLibs.POSIX.UCRT
$ErrorActionPreference = 'Stop'
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
$winlibs = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Directory -Filter 'BrechtSanders.WinLibs*' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($winlibs) { $env:Path = "$($winlibs.FullName)\mingw64\bin;$env:Path" }
Push-Location $PSScriptRoot
try {
    cargo build --release
    cargo test --release
    Write-Host "`nBuilt:" -ForegroundColor Green
    Get-ChildItem target\release\cosirca.exe, target\release\cosirca-cli.exe | ForEach-Object { "  $($_.FullName)  ($([math]::Round($_.Length/1MB,1)) MB)" }
} finally {
    Pop-Location
}
