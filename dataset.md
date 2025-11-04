# Dataset

The dataset will consist of JSON records, each representing a single network connection extracted from the original PCAP files.
Each JSON record will include:
1.	Extended flow metadata — selected transport and application-layer features (e.g., TLS, DNS, or HTTP attributes) describing the connection.
2.	Labeling (annotation) — structured semantic labels that characterize the connection from multiple perspectives, such as the underlying system, the running application, the user behavior, or the presence of malicious activity.

This structured approach enables flexible use of the dataset for multi-level learning and analysis, including anomaly detection, device and OS fingerprinting, application and user profiling, and malware communication characterization.

##  Define Dataset Scope and Taxonomy

Each connection (flow) in the dataset is annotated across distinct semantic levels, reflecting progressively higher-level abstractions of network activity.
The taxonomy separates the identity of the communicating system, the software context, the user behavior, and malicious intent.

| Level | Focus | Label Granularity | Typical Classes | Labeling Fields |
|-------|--------|------------------|-----------------|-----------------|
| **System Level** | Network identity & behavior | Device type, OS version | Windows 10/11, Android, iOS, routers, IoT cameras, smart TVs | `device_type`, `os_family`, `os_version`, `network_role` |
| **Application Level** | Software and activity context | Application name, category, and activity | Browser, Teams, Zoom, Steam, Office, YouTube, etc. | `application_name`, `application_category`, `application_activity` |
| **User Level** | Behavioral and identity context | User profile or role, user ID | office_workload, student_use, gaming, streaming | `user_profile`, `user_id` |
| **Malware Level** | Malicious communication | Malware family, campaign, or sample ID | Trickbot, Emotet, Zeus, Mirai, etc. | `malware_family`, `malware_sample` |

Explanation:
* The System Level describes what device and operating system generated the connection, providing the foundation for fingerprinting, device classification, and federated partitioning.
* The Application Level captures which software or service produced the traffic and what it was doing, enabling activity-specific modeling (e.g., “Teams video call” vs “Teams message”).
* The User Level represents the human or behavioral aspect, identifying how and by whom the system was used, allowing user-centric behavioral profiling.
* The Malware Level applies only to malicious traffic, describing known or inferred threat characteristics such as malware family, campaign, and associated sample identifier.

## Label Fields

Each JSON record will contain a hierarchical set of fields organized into logical namespaces (meta, system, application, user, malware).
Only the relevant sections will be included for a given sample — for instance, benign traffic will omit the malware block, while automated sandbox captures may omit the user block.

| **Field** | **Description** | **Example / Allowed Values** |
|------------|-----------------|------------------------------|
| **meta.sample_id** | Sample identifier within the dataset. | Generated or inferred value to identify the sample within the dataset. |
| **meta.environment** | Context or deployment domain of the captured traffic. | `enterprise_lab`, `home_network`, `sandbox`, `public_dataset`, `testbed` |
| **meta.data_origin** | Identifies the concrete source system, dataset, or platform from which the sample was obtained. | Tria.ge, CIC-IDS2017, CTU-13, Stratosphere-IPS |
| **system.device_type** | Physical or virtual type of device generating the flow. | `laptop`, `desktop`, `smartphone`, `tablet`, `router`, `access_point`, `camera`, `printer`, `iot_sensor`, `server`, `virtual_machine`, `sandbox_vm` |
| **system.os_family** | Operating system family or kernel type. | `Windows`, `Linux`, `Android`, `iOS`, `macOS`, `RouterOS`, `EmbeddedLinux`, `RTOS` |
| **system.os_version** | Specific OS version or release string. | Free text (e.g., `11.0.22631`, `Android 14`, `Ubuntu 22.04`) |
| **system.network_role** | Functional role of the device in the network. | `client`, `server`, `gateway`, `peer`, `controller`, `camera_streamer` |
| **application.name** | Application or process associated with the flow. | `Teams`, `Zoom`, `Chrome`, `Firefox`, `Steam`, `Office`, `Dropbox`, `YouTube`, `Spotify`, `Telegram`, `Unknown` |
| **application.category** | Broad class of the application. | `browser`, `collaboration`, `streaming`, `gaming`, `file_sharing`, `social_media`, `system_update`, `background_service` |
| **application.activity** | Functional context of the flow. | Depends on the type of application, values, such as `page_load`, `form_submit`, `file_download` | 
| **user.profile** | Typical user behavior or usage pattern. | `office_workload`, `student_use`, `gaming`, `streaming`, `mobile_user`, `automated_service` |
| **user.id** | Unique pseudonymous identifier for user-level profiling. | Alphanumeric ID (e.g., `usr_0042`) |
| **malware.family** | Identified or assigned malware family name. | `Emotet`, `Trickbot`, `Zeus`, `Dridex`, `Mirai`, `Qakbot`, `RedLine`, `AgentTesla`, `Unknown` |
| **malware.sample** | Specific sample reference (e.g., sandbox ID or hash). | Free text (e.g., `Tria.ge_12345`, SHA256 hash) |

Example:

```json
{
  "label": {
    "meta": {
      "sample_id": "240505-124ajsfc7v.behavioral1",
      "environment": "sandbox",
      "data_origin": "Tria.ge"
    },
  
    "system": {
      "device_type": "sandbox_vm",
      "os_family": "Windows",
      "os_version": "10",
      "network_role": "client"
    },
  
    "application": {
      "name": "malware_executable",
      "category": "unknown",
      "activity": "c2_communication"
    },
  
    "malware": {
      "family": "Emotet",
      "sample": "Tria.ge_240505-124ajsfc7v"
    }
  },

  "connection": {
    "pt": 6,
    "sa": "10.127.0.220",
    "sp": 59894,
    "da": "20.26.156.215",
    "dp": 443,
    "ps": 15,
    "pr": 75,
    "bs": 3424,
    "br": 91076,
    "ts": 1714947049.484,
    "td": 32.623,
    "tls": {
      "cver": "0x0303",
      "ccs": ["1301", "1302", "1303", "C02B", "..."],
      "cext": ["0000", "0017", "FF01", "..."],
      "csg": ["0403", "0804", "0401", "..."],
      "csv": ["3A3A", "0304", "0303", "..."],
      "alpn": ["h2", "http/1.1"],
      "sni": "github.com",
      "sver": "0x0303",
      "scs": "0x1301",
      "sext": ["002B", "0033", "0000", "0010"],
      "ssv": ["0304"],
      "ja3": "cd08e31494f9531f560d64c695473da9",
      "ja3s": "f4febc55ea12b31ae17cfb7e614afda8",
      "ja4": "t13d1516h2_8daaf6152771_e5627efa2ab1",
      "rec": [-512, 122, 1, 36, 3154, "..."]
    }
  }
}
```

Notes
* The dataset structure ensures flexibility — researchers can isolate or combine levels (e.g., train models only on system-level labels or on combined application–user context).
* The meta section ensures reproducibility, traceability, and compatibility with federated learning setups by documenting the capture source and environment.
* The schema can be extended with additional attributes such as confidence or annotation_method for semi-automated labeling pipelines.
