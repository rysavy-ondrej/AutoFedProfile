# Main script to run the extraction
# I've encountered an issue where the indices for network interfaces
# changed, so I made a change that it will automatically detect the correct interface number

# Get the list of interfaces Tshark sees
$interfaces = tshark -D

# Filter the interfaces based on "vEthernet (Default Switch)"
$interface = $interfaces | Where-Object { $_ -like "*vEthernet (Default Switch)*" }


# If it found the specified interface, we can run other scripts
if ($interface) {
    # Extract the interface number from tshark output
    $interfaceIndex = $interface -replace "^(\d+)\..*", '$1'
    
    # Running the Sandbox
    pwsh .\Scripts\Winbox.Fingerprint-Winapp.ps1 -OutRootPath .\Datasets\Windows\Samples -InterfaceIndex $interfaceIndex
} else {
    Write-Warning "Tshark could not find the specified vEthernet adapter. Verify that Microsoft Sandbox is installed correctly, enable Hyper-V or try using tshark -D and manually change the regex in this script."
    Write-Error "Failed to find virtual interface for capturing pcap files."
}