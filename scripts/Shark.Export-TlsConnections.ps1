<#
.SYNOPSIS
Uses tshark tool to extract TLS connection information from the given PCAP files.

.DESCRIPTION
This script processes PCAP files in the specified folder, extracts TLS information from them using the tshark tool and a custom Lua script, and exports the data to JSON format.
The script supports running on both Windows and Linux/WSL, and handles file path conversion for WSL compatibility.
The extracted TLS data is written to JSON files in the specified output folder.
It assumes that the tshark tool is available in the system paths.

.PARAMETER PcapFolder
The folder path containing the PCAP files to be processed. This folder should include all the `.pcap` files that you wish to analyze.

.PARAMETER OutPath
The folder path where the JSON output files will be stored. Each PCAP file will be processed, and its corresponding TLS information will be written into a separate JSON file in this folder.

.PARAMETER Recurse
(Optional) Set to true if you need to process the PCAP folder recursively.

.EXAMPLE
It is possible to run the script either in Windows with WSL or in Linux (WSL) with Powershell:

# using Linux/WSL:
./Shark.Export-TlsConnections.ps1 -PcapFolder /mnt/e/FETA/CSNet/datasets/iscx/ -OutPath /mnt/e/FETA/CSNet/datasets/iscx.tls/

# or from Windows PS:
./Shark.Export-TlsConnections.ps1 -PcapFolder E:/FETA/CSNet/datasets/iscx/ -OutPath E:/FETA/CSNet/datasets/iscx.tls/

#>

param (    
    [Parameter(Mandatory = $true, HelpMessage = "Specifies the folder with PCAP files.")]
    [string]$PcapFolder,   
      
    [Parameter(Mandatory = $true, HelpMessage = "The folder path where the Flow JSON files will be exported.")]
    [string]$OutPath,

    [Parameter(Mandatory = $false, HelpMessage = "Set to true if you need to process the pcap folder recursively.")]
    [bool]$Recurse=$false
)

# Lua script is included in this script to remove dependencies on external files
# If original Lua script is updated this section needs to be update too.
# 
$luaScript = @"
--[[
tls.lua

This TShark Lua script processes only TLS packets (using a display filter "tls")
and aggregates bidirectional connection information. For each TLS connection,
the following JSON fields are exported:

"pt" - protocol type (e.g., 6 for TCP)
"sa" - source address (client)
"sp" - source port (client)
"da" - destination address (server)
"dp" - destination port (server)
"ps" - packets from client to server
"pr" - packets from server to client
"bs" - octets (bytes) from client to server
"br" - octets (bytes) from server to client
"ts" - timestamp of the first packet (connection start)
"td" - connection duration (last packet timestamp minus ts)
"tls.cver" - TLS client version (from Client Hello), e.g., "0x0303"
"tls.ccs" - TLS client cipher suites as an array (from Client Hello)
"tls.cext" - TLS client extensions as an array
"tls.csg" - TLS client supported groups as an array
"tls.csv" - TLS client supported versions
"tls.alpn" - TLS ALPN protocols as an array (e.g., ["h2", "http/1.1"])
"tls.sni" - TLS Server Name Indication (SNI) from Client Hello
"tls.sver" - TLS server version (from Server Hello), e.g., "0x0303"
"tls.scs" - TLS server cipher suite (from Server Hello), e.g., "0xc030"
"tls.sext" - TLS server extensions as an array
"tls.ssv" - TLS server supported versions (for TLS 1.3)
"tls.ja3" - JA3 fingerprint for the TLS client
"tls.ja3s" - JA3S fingerprint for the TLS server
"tls.ja4" - JA4 fingerprint for the TLS client (alternative fingerprint)
"tls.ja4s" - JA4S fingerprint for the TLS server (alternative fingerprint)
"tls.rec" - array of individual TLS record lengths observed in the communication
"sample" - sample field (for testing), value can be "nil" if not provided

Usage:
  tshark -q -X lua_script:tls.lua -X lua_script1:sample=SAMPLE_NAME -r your_capture_file.pcap

where 
    SAMPLE_NAME is the value assigned to each flow within file
   
    
Example Output:
{ "pt": 6, "sa": "10.127.0.100", "sp": 49845, "da": "40.126.32.138", "dp": 443, "ps": 6, "pr": 5, "bs": 3219, "br": 5552, "ts": 1717277588.103, "td": 0.460, 
 "tls.cver": "0x0303", "tls.ccs": ["C02C", "C02B", "C030", "C02F", "C024", "C023", "C028", "C027", "C00A", "C009", "C014", "C013", "009D", "009C", 
"003D", "003C", "0035", "002F", "000A"], "tls.cext": ["0000", "0005", "000A", "000B", "000D", "0023", "0010", "0017", "FF01"], 
"tls.csg": ["0804", "0805", "0806", "0401", "0501", "0201", "0403", "0503", "0203", "0202", "0601", "0603"], "tls.csv": [], 
"tls.alpn": ["h2", "http/1.1"], "tls.sni": "login.live.com","tls.sver": "0x0303", "tls.scs": "0xc030", 
"tls.sext": ["0005", "0017", "FF01", "0000"], "tls.ssv": [], 
"tls.ja3": "28a2c9bd18a11de089ef85a160da29e4", "tls.ja3s": "7d8fd34fdb13a7fff30d5a52846b6c4c",
"tls.ja4": "t12d1909h2_d83cc789557e_7af1ed941c26","tls.ja4s": "t120400_c030_09f674154ab3",
"tls.rec": [-226, 3954, -102, -1, -40, 1, 40, -445, -4734, 10766, -26], "sample": "240601-1d3dcafe4y" }

-- See also: https://github.com/nmap/nmap/blob/master/nselib/tls.lua
             https://github.com/fullylegit/ja3/blob/master/ja3.lua
--]]

-- Field extractors for IP addresses, ports, and protocol
local f_ip_src      = Field.new("ip.src")
local f_ip_dst      = Field.new("ip.dst")
local f_tcp_srcport = Field.new("tcp.srcport")
local f_tcp_dstport = Field.new("tcp.dstport")
local f_udp_srcport = Field.new("udp.srcport")
local f_udp_dstport = Field.new("udp.dstport")
local f_ip_proto    = Field.new("ip.proto")

local f_tcp_syn     = Field.new("tcp.flags.syn")
local f_tcp_ack     = Field.new("tcp.flags.ack")

-- TLS-specific field extractors (handshake)
local f_tls_hs_type         = Field.new("tls.handshake.type")
local f_tls_hs_version      = Field.new("tls.handshake.version")
local f_tls_hs_ciphersuite  = Field.new("tls.handshake.ciphersuite")
local f_tls_hs_extension    = Field.new("tls.handshake.extension.type")
local f_tls_hs_groups       = Field.new("tls.handshake.extensions_supported_group")
local f_tls_hs_ec_points    = Field.new("tls.handshake.extensions_ec_point_format")
local f_tls_hs_sni          = Field.new("tls.handshake.extensions_server_name")
local f_tls_hs_alpn         = Field.new("tls.handshake.extensions_alpn_str")
local f_tls_hs_sig_hash     = Field.new("tls.handshake.sig_hash_alg")
local f_tls_hs_sup_version  = Field.new("tls.handshake.extensions.supported_version")

-- TLS-JA hashes field extractors (handshake)
local f_tls_hs_ja3  = Field.new("tls.handshake.ja3")
local f_tls_hs_ja4  = Field.new("tls.handshake.ja4")
local f_tls_hs_ja3s  = Field.new("tls.handshake.ja3s")
local f_tls_hs_ja4s  = Field.new("ja4.ja4s")
local f_tls_hs_ja4x  = Field.new("ja4.ja4x") -- array!

-- TLS-metainformation extraction, e.g., record lengths
local f_tls_rec_length      = Field.new("tls.record.length")

-- Utility function to generate an order-insensitive key for a connection.
local function get_connection_key(src, sp, dst, dp, proto)
    if src < dst or (src == dst and sp < dp) then
        return string.format("%s-%d-%s-%d-%d", src, sp, dst, dp, proto)
    else
        return string.format("%s-%d-%s-%d-%d", dst, dp, src, sp, proto)
    end
end

-- Utility function to convert a Lua array of strings into a JSON array.
local function array_to_json(arr)
    local out = "["
    for i, v in ipairs(arr) do
        if i > 1 then
            out = out .. ", "
        end
        v = v:gsub('"', '\\"')  -- escape any double quotes
        out = out .. string.format('"%s"', v)
    end
    out = out .. "]"
    return out
end

local function int_array_to_json(arr)
    local out = "["
    for i, v in ipairs(arr) do
        if i > 1 then
            out = out .. ", "
        end
        out = out .. string.format('%d', v)
    end
    out = out .. "]"
    return out
end

local function int_array_to_hex16_json(arr)
    local out = "["
    for i, v in ipairs(arr) do
        if i > 1 then
            out = out .. ", "
        end
        out = out .. string.format('"%04X"', v)
    end
    out = out .. "]"
    return out
end

GREASE_VALUES = {
    [0x0A0A] = true,
    [0x1A1A] = true,
    [0x2A2A] = true,
    [0x3A3A] = true,
    [0x4A4A] = true,
    [0x5A5A] = true,
    [0x6A6A] = true,
    [0x7A7A] = true,
    [0x8A8A] = true,
    [0x9A9A] = true,
    [0xAAAA] = true,
    [0xBABA] = true,
    [0xCACA] = true,
    [0xDADA] = true,
    [0xEAEA] = true,
    [0xFAFA] = true
}

function remove_grease(list)
    local clean_list = {}
    for i, entry in ipairs(list) do
        if GREASE_VALUES[entry.value] == nil then
            table.insert(clean_list, entry.value)
        end
    end
    return clean_list
end

function to_array(list)
    local clean_list = {}
    for i, entry in ipairs(list) do
        table.insert(clean_list, entry.value)
    end
    return clean_list
end

-- MAIN
-- Access the arguments passed to the script
local args_array = {...}
local args_map = {}

for _, pairStr in ipairs(args_array) do
    local key, value = pairStr:match("([^=]+)=(.+)")
    if key and value then
        args_map[key] = value
    end
end

local sample = args_map["sample"]
local incomplete = args_map["incomplete"]
if not incomplete then
    incomplete="drop"
end


-- Table to store connections; key is an order-insensitive connection key.
local connections = {}

-- Use a display filter to process only TLS packets.
local tap = Listener.new("frame", "tls")

function tap.packet(pinfo, tvb)
    -- Extract IP and port fields.
    local ip_src   = f_ip_src()
    local ip_dst   = f_ip_dst()
    local ip_proto = f_ip_proto()
    local tcp_src  = f_tcp_srcport()
    local tcp_dst  = f_tcp_dstport()
    local pkt_len = tvb:len()
    local pkt_num = pinfo.number


    local src   = tostring(ip_src)
    local dst   = tostring(ip_dst)
    local proto = tonumber(tostring(ip_proto))
    local sp = tonumber(tostring(tcp_src))
    local dp = tonumber(tostring(tcp_dst))

    -- final check that we have evruhgting that we need:
    if src and dst and proto and sp and dp then
            local key = get_connection_key(src, sp, dst, dp, proto)
            local current_ts = pinfo.abs_ts

            -- first packet then initialize the connection
            if not connections[key] then
                -- First packet seen for this connection: we need to determine if this is client or server packet:
                -- 1. Typically, the client uses an ephemeral (high-numbered) port while the server uses a well-known port.
                -- 2. The first packet should include the SYN flag without the ACK flag, which means that it is the packet 
                -- from the client starting the handshake. On the other hand we use tls filter that does not provide us 
                -- TCP handshake nor empty TCP packets.
                -- Thus the only rule is port number:
                local client_sa, client_da = ""
                local client_sp, client_dp = 0            
                if (sp >= dp)  then
                    client_sa = src
                    client_da = dst
                    client_sp = sp
                    client_dp = dp
                else
                    client_da = src
                    client_sa = dst
                    client_dp = sp
                    client_sp = dp
                end

                connections[key] = {
                    pt = proto,
                    sa = client_sa,   -- client IP
                    sp = client_sp,    -- client port
                    da = client_da,   -- server IP
                    dp = client_dp,    -- server port
                    ps = 0,     -- packets from client to server
                    pr = 0,     -- packets from server to client
                    bs = 0,     -- octets from client to server
                    br = 0,     -- octets from server to client
                    ts = current_ts,  -- first seen timestamp
                    td = 0,           -- duration (will be updated)
                    ["tls.cv"] = "",  -- TLS client version
                    ["tls.sv"] = "",  -- TLS server version
                    ["tls.sc"] = "",  -- TLS server cipher suite
                    ["tls.cc"] = {},  -- TLS client cipher suites
                    ["tls.ce"] = {},  -- TLS client extension types
                    ["tls.se"] = {},  -- TLS server extension types
                    ["tls.sn"] = "",  -- TLS server name
                    ["tls.alpn"] = {},-- TLS ALPN
                    ["tls.sig"] = {},  -- TLS Signature Algorithms
                    ["tls.csv"] = {},  -- TLS Client supported versions
                    ["tls.ssv"] = {},  -- TLS Server supported versions
                    ["tls.ja3"] = "",
                    ["tls.ja4"] = "",
                    ["tls.ja3s"] = "",
                    ["tls.ja4s"] = "",
                    ["tls.ja4x"] = {},
                    ["tls.rec"] = {}  -- array of TLS record length (negative for C->S, positive fo S->C)
                }
            end

            local conn = connections[key]
            -- Update duration with each packet.
            conn.td = current_ts - conn.ts

            -- Determine packet direction.
            if src == conn.sa and sp == conn.sp and dst == conn.da and dp == conn.dp then
                -- Client -> Server
                conn.ps = conn.ps + 1
                conn.bs = conn.bs + pkt_len
            elseif src == conn.da and sp == conn.dp and dst == conn.sa and dp == conn.sp then
                -- Server -> Client
                conn.pr = conn.pr + 1
                conn.br = conn.br + pkt_len
            end


            -- Process TLS record lengths.
            local tls_lengths = { f_tls_rec_length() }
            if tls_lengths and #tls_lengths > 0 then
                for _, rec in ipairs(tls_lengths) do
                    local rec_val = tonumber(tostring(rec))
                    if rec_val then
                        if src == conn.sa and sp == conn.sp and dst == conn.da and dp == conn.dp then
                            table.insert(conn["tls.rec"], -rec_val)
                        elseif src == conn.da and sp == conn.dp and dst == conn.sa and dp == conn.sp then
                            table.insert(conn["tls.rec"], rec_val)
                        end
                    end
                end
            end

            -- Process TLS handshake fields if available.
            local hs_type_field = f_tls_hs_type()
            if hs_type_field then
                local hs_type = tonumber(tostring(hs_type_field))
                -------------------------------------------------
                -- CLIENT HELLO:
                if hs_type == 1 then
                    -- For client hello, record client version and cipher suites.
                    if conn["tls.cv"] == "" then
                        local ver = f_tls_hs_version()
                        if ver then
                            conn["tls.cv"] = tostring(ver)
                        end
                    end
                    local ciphers = { f_tls_hs_ciphersuite() }
                    if ciphers then
                        conn["tls.cc"] = remove_grease(ciphers)
                    end
                    local extensions = { f_tls_hs_extension() }
                    if extensions then
                        conn["tls.ce"] = remove_grease(extensions) 
                    end
                    if conn["tls.sn"] == "" then
                        local server_name = f_tls_hs_sni()
                        if server_name then
                            conn["tls.sn"] = tostring(server_name)
                        end
                    end
                    local alpn_arr = { f_tls_hs_alpn() }
                    if alpn_arr then
                        conn["tls.alpn"] = to_array(alpn_arr)
                    end
                    local sig_algo = { f_tls_hs_sig_hash() }
                    if sig_algo then
                        conn["tls.sig"] = to_array(sig_algo)
                    end
                    local sup_vers = { f_tls_hs_sup_version() }
                    if sup_vers then
                        conn["tls.csv"] = to_array(sup_vers)
                    end
                    if conn["tls.ja3"] == "" then
                        local ja3 = f_tls_hs_ja3()
                        if ja3 then
                            conn["tls.ja3"] = tostring(ja3)
                        end
                    end
                    
                    if conn["tls.ja4"] == "" then
                        local ja4 = f_tls_hs_ja4()
                        if ja4 then
                            conn["tls.ja4"] = tostring(ja4)
                        end
                    end
                -------------------------------------------------   
                -- SERVER HELLO:
                elseif hs_type == 2 then        
                    if conn["tls.sv"] == "" then
                        local ver = f_tls_hs_version()
                        if ver then
                            conn["tls.sv"] = tostring(ver)
                        end
                    end
                    if conn["tls.sc"] == "" then
                        local cipher = f_tls_hs_ciphersuite()
                        if cipher then
                            conn["tls.sc"] = tostring(cipher)
                        end
                    end
                    local extensions = { f_tls_hs_extension() }
                    if extensions then
                        conn["tls.se"] = remove_grease(extensions) 
                    end
                    local sup_vers = { f_tls_hs_sup_version() }
                    if sup_vers then
                        conn["tls.ssv"] = to_array(sup_vers)
                    end
                    if conn["tls.ja3s"] == "" then
                        local ja3s = f_tls_hs_ja3s()
                        if ja3s then
                            conn["tls.ja3s"] = tostring(ja3s)
                        end
                    end
                    if conn["tls.ja4s"] == "" then
                        local ja4s = f_tls_hs_ja4s()
                        if ja4s then
                            conn["tls.ja4s"] = tostring(ja4s)
                        end
                    end
                    local ja4x = f_tls_hs_ja4x()
                    if ja4x then
                        table.insert(conn["tls.ja4x"], tostring(ja4x))
                    end
                end
            end
    end
end

-- After processing all packets, print each TLS connection as a JSON object on its own line.
function tap.draw()
    for _, conn in pairs(connections) do
        if not (incomplete == "drop" and (conn["tls.cv"] == "" or conn["tls.sv"] == "")) then
            local client_ciphers = int_array_to_hex16_json(conn["tls.cc"])
            local client_extensions = int_array_to_hex16_json(conn["tls.ce"])
            local server_extensions = int_array_to_hex16_json(conn["tls.se"])
            local alpn_string = array_to_json(conn["tls.alpn"])
            local sig_string = int_array_to_hex16_json(conn["tls.sig"])
            local cvers = int_array_to_hex16_json(conn["tls.csv"])
            local svers = int_array_to_hex16_json(conn["tls.ssv"])
            local json_line = string.format(
                '{ "pt": %d, "sa": "%s", "sp": %d, "da": "%s", "dp": %d, "ps": %d, "pr": %d, "bs": %d, "br": %d, "ts": %.3f, "td": %.3f, ' ..
                '"tls.cver": "%s", "tls.ccs": %s, "tls.cext": %s, "tls.csg": %s, "tls.csv": %s, "tls.alpn": %s, "tls.sni": "%s",' .. 
                '"tls.sver": "%s", "tls.scs": "%s", "tls.sext": %s, "tls.ssv": %s, ' ..
                '"tls.ja3": "%s", "tls.ja3s": "%s","tls.ja4": "%s","tls.ja4s": "%s",'..
                '"tls.rec": %s, "sample": "%s" }',
                conn.pt, conn.sa, conn.sp, conn.da, conn.dp,
                conn.ps, conn.pr, conn.bs, conn.br,
                conn.ts, conn.td,
                conn["tls.cv"], client_ciphers, client_extensions, sig_string, cvers, alpn_string, conn["tls.sn"], 
                conn["tls.sv"], conn["tls.sc"], server_extensions, svers,
                conn["tls.ja3"], conn["tls.ja3s"], conn["tls.ja4"], conn["tls.ja4s"],
                int_array_to_json(conn["tls.rec"]), sample
            )
            print(json_line)
        end
    end
end

"@

if (-Not (Test-Path -Path $OutPath)) {
    New-Item -ItemType Directory -Path $OutPath -Force
}

# Test if the required tshark tool is installed
if ($IsWindows) {
    if (Get-Command tshark.exe -ErrorAction SilentlyContinue) {
        Write-Host "Checking tshark.exe...Ok!"
    } else {
        Write-Host "Fatal error: tshark.exe is not available!"
        exit
    }
} else {
    if (Get-Command tshark -ErrorAction SilentlyContinue) {
        Write-Host "Checking tshark...Ok!"
    } else {
        Write-Host "Fatal error: tshark is not available!"
        exit
    }   
}
# PREPARE LUA script:
# Generate a temporary file path. Here we combine the TEMP directory with a filename.
$tlsLuaFile = Join-Path $env:TEMP "tshark.tls.lua"
# Write the Lua script content to the temporary file.
Set-Content -Path $tlsLuaFile -Value $luaScript -Encoding UTF8
Write-Host "Lua script deployed to $tlsLuaFile."

<#
.SYNOPSIS
Converts Windows Path to equivalane WSL path.

.DESCRIPTION
The function converts windows-based path to the same path in WSL. 

.PARAMETER windowsPath
WIndows path to convert to the equivalent WSL path.

.EXAMPLE
An example
#>
function Convert-WindowsPathToWSL {
    param (
        [string]$windowsPath
    )

    # Replace backslashes with forward slashes
    $wslPath = $windowsPath -replace '\\', '/'

    if ($wslPath -match '^([A-Za-z]):')
    {   # Replace the drive letter with /mnt/<drive_letter>
        $mntPoint = $($matches[1].ToLower())   
        $wslPath = $wslPath -replace '^([A-Za-z]):', "/mnt/$mntPoint"
    }
    Write-Host "$windowsPath  -> $wslPath"
    return $wslPath
}

function Invoke-Shark {
    param (
        [string]$pcapFile,
        [string]$sampleName
    )   
    # $tempFile = [System.IO.Path]::GetTempFileName()
    if ($IsWindows) {
        $content = & tshark.exe -q -X lua_script:$tlsLuaFile -X lua_script1:sample=$sampleName -r $pcapFile
    } else {
        $content = & tshark -q -X lua_script:$tlsLuaFile -X lua_script1:sample=$sampleName -r $pcapFile
    }
    #$content = Get-Content $tempFile
    #Remove-Item $tempFile
    return $content
}

#-----------------------------------------------------------------------------------------------------------

if ($Recurse)
{
    $PcapFiles = Get-ChildItem -Path "$PcapFolder" -Recurse -Filter "*.pcap??"
} else {
    $PcapFiles = Get-ChildItem -Path "$PcapFolder" -Filter "*.pcap??"
}

$allcaps = $PcapFiles.Length
Write-Host "Found $allcaps capture files. Start processing..."
Write-Host ""


$numcaps = 0
$numrecs = 0
$elapsed = 0.0
foreach ($PcapFile in $PcapFiles) {
    $OutPath = (Get-Item -Path $OutPath).FullName
    $PcapName = $PcapFile.BaseName
    $JsonName = $PcapName + ".json"
    $JsonPath = Join-Path -Path $OutPath -ChildPath $JsonName

    $numcaps += 1
    Write-Host "Processing capture file ($numcaps/$allcaps) $PcapFile ..."

    $timeTaken = Measure-Command {
        $numlines = 0
        $lines = Invoke-Shark $PcapFile $PcapName
        foreach ($line in $lines) {
            $json = $line | ConvertFrom-Json
            $json | ConvertTo-Json -Compress | Add-Content -Path $JsonPath
            $numlines += 1
        }
    }
    Write-Host "Written $numlines records to $JsonPath ($($timeTaken.TotalSeconds) secs)"
    Write-Host ""
    
    $numrecs += $numlines
    $elapsed += $timeTaken.TotalSeconds
}
Write-Host "All done: $numcaps capture files processed, $numrecs connections written. Elapsed time $elapsed secs."