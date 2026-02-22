param(
  [string]$Repo = "rozy0311/viralops-engine",
  [string]$EnvPath = (Join-Path $PSScriptRoot "..\.env"),
  [switch]$Apply
)

$ErrorActionPreference = 'Stop'

$DryRun = -not $Apply

function Read-DotEnv([string]$path) {
  if (-not (Test-Path $path)) {
    throw "Env file not found: $path"
  }

  $map = @{}
  foreach ($line in Get-Content -LiteralPath $path) {
    $trim = $line.Trim()
    if (-not $trim) { continue }
    if ($trim.StartsWith('#')) { continue }
    if (-not ($trim -match '^[A-Za-z_][A-Za-z0-9_]*\s*=')) { continue }

    $parts = $trim.Split('=', 2)
    $key = $parts[0].Trim()
    $value = $parts[1]

    # Keep everything after '=' (including spaces), but strip surrounding quotes.
    $value = $value.Trim()
    if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
      $value = $value.Substring(1, $value.Length - 2)
    }

    $map[$key] = $value
  }

  return $map
}

function Require-Command([string]$name) {
  $cmd = Get-Command $name -ErrorAction SilentlyContinue
  if (-not $cmd) {
    throw "Missing required command: $name (install GitHub CLI)"
  }
}

Require-Command gh

$envMap = Read-DotEnv $EnvPath

# Map local .env keys -> GitHub Actions secret names expected by .github/workflows/ui-automation.yml
$secretPlan = @{
  # LLM
  'GEMINI_API_KEY'    = $(
    if ($envMap.ContainsKey('GEMINI_API_KEY') -and $envMap['GEMINI_API_KEY']) { $envMap['GEMINI_API_KEY'] }
    else { $envMap['GOOGLE_AI_STUDIO_API_KEY'] }
  )
  'FALLBACK_GEMINI_API_KEY' = $(
    if ($envMap.ContainsKey('FALLBACK_GEMINI_API_KEY') -and $envMap['FALLBACK_GEMINI_API_KEY']) { $envMap['FALLBACK_GEMINI_API_KEY'] }
    else { $envMap['FALLBACK_GOOGLE_AI_STUDIO_API_KEY'] }
  )
  'SECOND_FALLBACK_GEMINI_API_KEY' = $(
    if ($envMap.ContainsKey('SECOND_FALLBACK_GEMINI_API_KEY') -and $envMap['SECOND_FALLBACK_GEMINI_API_KEY']) { $envMap['SECOND_FALLBACK_GEMINI_API_KEY'] }
    else { $envMap['SECOND_FALLBACK_GOOGLE_AI_STUDIO_API_KEY'] }
  )
  'THIRD_FALLBACK_GEMINI_API_KEY' = $(
    if ($envMap.ContainsKey('THIRD_FALLBACK_GEMINI_API_KEY') -and $envMap['THIRD_FALLBACK_GEMINI_API_KEY']) { $envMap['THIRD_FALLBACK_GEMINI_API_KEY'] }
    else { $envMap['THIRD_FALLBACK_GOOGLE_AI_STUDIO_API_KEY'] }
  )
  # Optional aliases (some code supports both names)
  'GOOGLE_AI_STUDIO_API_KEY' = $envMap['GOOGLE_AI_STUDIO_API_KEY']
  'FALLBACK_GOOGLE_AI_STUDIO_API_KEY' = $envMap['FALLBACK_GOOGLE_AI_STUDIO_API_KEY']
  'SECOND_FALLBACK_GOOGLE_AI_STUDIO_API_KEY' = $envMap['SECOND_FALLBACK_GOOGLE_AI_STUDIO_API_KEY']
  'THIRD_FALLBACK_GOOGLE_AI_STUDIO_API_KEY' = $envMap['THIRD_FALLBACK_GOOGLE_AI_STUDIO_API_KEY']
  'GH_MODELS_API_KEY' = $envMap['GH_MODELS_API_KEY']

  # Publer
  'PUBLER_API_KEY'    = $envMap['PUBLER_API_KEY']
  'PUBLER_EMAIL'      = $envMap['PUBLER_EMAIL']
  'PUBLER_PASSWORD'   = $envMap['PUBLER_PASSWORD']

  # Shopify (workflow uses SHOPIFY_ADMIN_TOKEN + SHOPIFY_PUBLIC_DOMAIN)
  'SHOPIFY_SHOP'         = $envMap['SHOPIFY_SHOP']
  'SHOPIFY_ADMIN_TOKEN'  = $(
    if ($envMap.ContainsKey('SHOPIFY_ACCESS_TOKEN') -and $envMap['SHOPIFY_ACCESS_TOKEN']) { $envMap['SHOPIFY_ACCESS_TOKEN'] }
    elseif ($envMap.ContainsKey('SHOPIFY_TOKEN') -and $envMap['SHOPIFY_TOKEN']) { $envMap['SHOPIFY_TOKEN'] }
    else { '' }
  )
  'SHOPIFY_BLOG_ID'       = $(
    if ($envMap.ContainsKey('SHOPIFY_VIRALOPS_BLOG_ID') -and $envMap['SHOPIFY_VIRALOPS_BLOG_ID']) { $envMap['SHOPIFY_VIRALOPS_BLOG_ID'] }
    else { $envMap['SHOPIFY_BLOG_ID'] }
  )
  'SHOPIFY_PUBLIC_DOMAIN' = $(
    if ($envMap.ContainsKey('SHOPIFY_CUSTOM_DOMAIN') -and $envMap['SHOPIFY_CUSTOM_DOMAIN']) { $envMap['SHOPIFY_CUSTOM_DOMAIN'] }
    elseif ($envMap.ContainsKey('SHOPIFY_STORE_DOMAIN') -and $envMap['SHOPIFY_STORE_DOMAIN']) { $envMap['SHOPIFY_STORE_DOMAIN'] }
    else { '' }
  )
}

$missing = @()
foreach ($k in $secretPlan.Keys) {
  if (-not $secretPlan[$k]) {
    $missing += $k
  }
}

Write-Host "Target repo: $Repo"
Write-Host "Env file: $EnvPath"
Write-Host "DryRun: $DryRun"
Write-Host "Will set secrets: $($secretPlan.Keys.Count)"

if ($missing.Count) {
  Write-Host "Missing/empty values for: $($missing -join ', ')" -ForegroundColor Yellow
}

foreach ($name in ($secretPlan.Keys | Sort-Object)) {
  $value = $secretPlan[$name]
  if (-not $value) { continue }

  if ($DryRun) {
    Write-Host "[DryRun] gh secret set $name -R $Repo (value hidden)"
    continue
  }

  # Do not echo the value.
  gh secret set $name -R $Repo -b $value | Out-Null
  Write-Host "Set: $name"
}

if ($DryRun) {
  Write-Host "\nRun again with -Apply to actually write secrets." -ForegroundColor Cyan
}
