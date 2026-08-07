Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like '*sight-worker*' -or $_.Name -eq 'sens-broker.exe' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
