# 将 CMake 生成的 .sln 从 out/build/ 复制到根目录并重写路径
param(
    [string]$SourceSln,
    [string]$DestSln
)

if (-not (Test-Path $SourceSln)) {
    Write-Host "Source .sln not found: $SourceSln" -ForegroundColor Yellow
    exit 0
}

$sln = Get-Content -Raw $SourceSln

# 给所有 .vcxproj 路径加 out\build\ 前缀（尚未有此前缀的）
$re = [regex]'"((?!(?:out\\build|\.\.\\))[^"]+)\.vcxproj"'
$sln = $re.Replace($sln, { param($m) '"out\build\' + $m.Groups[1].Value + '.vcxproj"' })

Set-Content -Path $DestSln -Value $sln -NoNewline

Write-Host "Copied .sln: $SourceSln -> $DestSln"
