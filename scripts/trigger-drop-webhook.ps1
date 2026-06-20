# POST a tweet URL to your Cursor Automation webhook (after saving the automation).
# Usage:
#   .\scripts\trigger-drop-webhook.ps1 -WebhookUrl "https://..." -TweetUrl "https://x.com/DoorDash/status/..."

param(
    [Parameter(Mandatory = $true)]
    [string]$WebhookUrl,
    [Parameter(Mandatory = $true)]
    [string]$TweetUrl
)

$body = @{ tweet_url = $TweetUrl } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri $WebhookUrl -Body $body -ContentType "application/json"
