param(
  [Parameter(Mandatory = $true, Position = 0)]
  [string]$Script,

  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$python = $env:VIRALOPS_PYTHON

if (-not $python) {
  $defaultVenvPython = "D:\vops-venv\Scripts\python.exe"
  if (Test-Path $defaultVenvPython) {
    $python = $defaultVenvPython
  } else {
    $python = "python"
  }
}

& $python $Script @Args
exit $LASTEXITCODE
