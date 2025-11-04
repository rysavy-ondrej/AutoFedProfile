# Dataset

##  Define Dataset Scope and Taxonomy

| Level | Focus | Label Granularity | Typical Classes | Labeling Fields |
|-------|--------|------------------|-----------------|-----------------|
| **System Level** | Network identity & behavior | Device type, OS version | Windows 10/11, Android, iOS, routers, IoT cameras, smart TVs | `device_type`, `os_family`, `os_version`, `network_role` |
| **Application Level** | Software and activity context | Application name, category, and activity | Browser, Teams, Zoom, Steam, Office, YouTube, etc. | `application_name`, `application_category`, `application_activity` |
| **User Level** | Behavioral and identity context | User profile or role, user ID | office_workload, student_use, gaming, streaming | `user_profile`, `user_id` |
| **Malware Level** | Malicious communication | Malware family, campaign, or sample ID | Trickbot, Emotet, Zeus, Mirai, etc. | `malware_family`, `malware_sample` |

## Label Fields

| **Field** | **Description** | **Example / Allowed Values** |
|------------|-----------------|------------------------------|
| **meta.label_source** | Origin or method of data capture or annotation. | `sandbox`, `enterprise_lab`, `home_network`, `public_dataset`, `testbed` |
| **meta.environment** | Context or deployment domain of the captured traffic. | `enterprise_lab`, `home_network`, `sandbox`, `public_dataset`, `testbed` |
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
