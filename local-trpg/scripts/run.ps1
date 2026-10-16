param([int]$Port = 8768)
$ErrorActionPreference = 'Stop'
$Project = '/home/example/ai_project/local-trpg'
Write-Host "打开 http://127.0.0.1:$Port ，保留此窗口。首次模型加载需要数分钟。" -ForegroundColor Cyan
& wsl.exe -d Ubuntu -- bash "$Project/scripts/run.sh" --port $Port
