./Shark.Export-TlsConnections.ps1 -PcapFolder ../../CSNet/datasets/desktop  -Recurse $true -OutPath ../datasets/desktop.tls/

./Shark.Export-TlsConnections.ps1 -PcapFolder ../../CSNet/datasets/iscx  -Recurse $true -OutPath ../datasets/iscx.tls/

./Shark.Export-TlsConnections.ps1 -PcapFolder ../../CSNet/datasets/mobile   -Recurse $true -OutPath ../datasets/mobile.tls/

./Shark.Export-TlsConnections.ps1 -PcapFolder ../../Datasets/Windows/Captures  -Recurse $true -OutPath ../datasets/windows.tls/

# CIC Adware
./Shark.Export-TlsConnections.ps1 -PcapFolder ../../Datasets/AndroidAdware/Adware  -Recurse $true -OutPath ../datasets/cic-aa.adware.tls/
./Shark.Export-TlsConnections.ps1 -PcapFolder ../../Datasets/AndroidAdware/Malware  -Recurse $true -OutPath ../datasets/cic-aa.malware.tls/
./Shark.Export-TlsConnections.ps1 -PcapFolder ../../Datasets/AndroidAdware/Benign  -Recurse $true -OutPath ../datasets/cic.aa.normal.tls/