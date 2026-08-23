$python = "C:\Users\Noodl\Projects\FuckAround\MJ\OCRs\H-1\.venv\Scripts\python.exe"
$script = "C:\Users\Noodl\Projects\FuckAround\MJ\OCRs\H-1\DataDump\pipeline.py"
$stdout = "C:\Users\Noodl\Projects\FuckAround\MJ\OCRs\H-1\DataDump\pipeline_stdout.log"
$stderr = "C:\Users\Noodl\Projects\FuckAround\MJ\OCRs\H-1\DataDump\pipeline_stderr.log"

$attempt = 1
while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] Attempt $attempt - starting pipeline..."
    
    $proc = Start-Process -FilePath $python -ArgumentList $script -RedirectStandardOutput $stdout -RedirectStandardError $stderr -NoNewWindow -PassThru
    $proc.WaitForExit()
    $code = $proc.ExitCode
    
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] Pipeline exited with code $code"
    
    if ($code -eq 0) {
        Write-Host "[$ts] Pipeline completed successfully"
        break
    }
    
    $lastLine = Get-Content $stdout -Tail 2 -ErrorAction SilentlyContinue | Select-Object -Last 1
    Write-Host "[$ts] Last output: $lastLine"
    
    if ($lastLine -match "Starting HTTP server") {
        Write-Host "[$ts] Server started - pipeline finished spotting, exiting"
        break
    }
    
    $attempt++
    Write-Host "[$ts] Waiting 5s before restart..."
    Start-Sleep -Seconds 5
}