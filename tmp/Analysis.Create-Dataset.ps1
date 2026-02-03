param (
    [Parameter(Mandatory=$false, HelpMessage="Start date from which we should start taking data from.")]
    [string]$startDate = "00000000",
    [Parameter(Mandatory=$false, HelpMessage="End date where we should end the data extraction.")]
    [string]$endDate = "99999999",
    [Parameter(Mandatory=$false, HelpMessage="A comma (,) separated string (without whitespaces) of applications to be filtered.")]
    [string]$app = "",
    [Parameter(Mandatory=$false, HelpMessage="Specifies the name of the output folder. Default location is always root of 'Export' folder.")]
    [string]$outputName = "tls_dataset.json" # PATH will always be the same into .../Datasets/Export
)

<# Python parameters
# --start_date
# --end_date
# --applications
# --output_name
#>
$time = Measure-Command {
    $featureExtract = "$PSScriptRoot\feature_ext.ps1"
    $mergeData = "$PSScriptRoot\merge_jsons.py"
    
    Write-Host "INFO: Beginning feature extraction..."
    & $featureExtract -startDate $startDate -endDate $endDate -app $app
    Write-Host "`nINFO: Beginning dataset aggregation..."
    & "python.exe" $mergeData "--start_date" $startDate "--end_date" $endDate "--applications" $app "--output_name" $outputName
}
Write-Host "`nScript ran for: $time"