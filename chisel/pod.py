import runpod
import os
from dotenv import load_dotenv
import requests
from typing import Optional, Dict, List, Any, Union
from typing_extensions import TypedDict
from dataclasses import dataclass

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")


class SshDetails(TypedDict):
    host: str
    port: int
    user: str
    connection_string: str


class PortMappings(TypedDict, total=False):
    """Port mappings with string keys (port numbers) and int values (mapped ports)."""

    pass  # This allows any string key with int value


@dataclass
class GpuType:
    id: str
    count: int
    displayName: str
    securePrice: float
    communityPrice: float
    oneMonthPrice: float
    threeMonthPrice: float
    sixMonthPrice: float
    oneWeekPrice: float
    communitySpotPrice: float
    secureSpotPrice: float

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["GpuType"]:
        return cls(**data) if data else None


@dataclass
class Gpu:
    id: str
    count: int
    displayName: str
    securePrice: float
    communityPrice: float
    oneMonthPrice: float
    threeMonthPrice: float
    sixMonthPrice: float
    oneWeekPrice: float
    communitySpotPrice: float
    secureSpotPrice: float

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["Gpu"]:
        return cls(**data) if data else None


@dataclass
class CpuType:
    id: str
    displayName: str
    cores: int
    threadsPerCore: int
    groupId: str

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["CpuType"]:
        return cls(**data) if data else None


@dataclass
class Machine:
    minPodGpuCount: Optional[int] = None
    gpuTypeId: Optional[str] = None
    gpuType: Optional[GpuType] = None
    cpuCount: Optional[int] = None
    cpuTypeId: Optional[str] = None
    cpuType: Optional[CpuType] = None
    location: Optional[str] = None
    dataCenterId: Optional[str] = None
    diskThroughputMBps: Optional[int] = None
    maxDownloadSpeedMbps: Optional[int] = None
    maxUploadSpeedMbps: Optional[int] = None
    supportPublicIp: Optional[bool] = None
    secureCloud: Optional[bool] = None
    maintenanceStart: Optional[str] = None
    maintenanceEnd: Optional[str] = None
    maintenanceNote: Optional[str] = None
    note: Optional[str] = None
    costPerHr: Optional[float] = None
    currentPricePerGpu: Optional[float] = None
    gpuAvailable: Optional[int] = None
    gpuDisplayName: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Machine":
        if not data:
            return cls()

        # Handle nested objects
        gpu_type = None
        if "gpuType" in data and data["gpuType"]:
            gpu_type = GpuType.from_dict(data["gpuType"])

        cpu_type = None
        if "cpuType" in data and data["cpuType"]:
            cpu_type = CpuType.from_dict(data["cpuType"])

        return cls(
            minPodGpuCount=data.get("minPodGpuCount"),
            gpuTypeId=data.get("gpuTypeId"),
            gpuType=gpu_type,
            cpuCount=data.get("cpuCount"),
            cpuTypeId=data.get("cpuTypeId"),
            cpuType=cpu_type,
            location=data.get("location"),
            dataCenterId=data.get("dataCenterId"),
            diskThroughputMBps=data.get("diskThroughputMBps"),
            maxDownloadSpeedMbps=data.get("maxDownloadSpeedMbps"),
            maxUploadSpeedMbps=data.get("maxUploadSpeedMbps"),
            supportPublicIp=data.get("supportPublicIp"),
            secureCloud=data.get("secureCloud"),
            maintenanceStart=data.get("maintenanceStart"),
            maintenanceEnd=data.get("maintenanceEnd"),
            maintenanceNote=data.get("maintenanceNote"),
            note=data.get("note"),
            costPerHr=data.get("costPerHr"),
            currentPricePerGpu=data.get("currentPricePerGpu"),
            gpuAvailable=data.get("gpuAvailable"),
            gpuDisplayName=data.get("gpuDisplayName"),
        )


@dataclass
class NetworkVolume:
    id: str
    name: str
    size: int
    dataCenterId: str

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["NetworkVolume"]:
        return cls(**data) if data else None


@dataclass
class SavingsPlan:
    costPerHr: float
    endTime: str
    gpuTypeId: str
    id: str
    podId: str
    startTime: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SavingsPlan":
        """Create SavingsPlan from dict. Data is required for SavingsPlan."""
        return cls(**data)


@dataclass
class Pod:
    # Required fields
    id: str

    # Optional fields with defaults
    adjustedCostPerHr: Optional[float] = None
    aiApiId: Optional[str] = None
    consumerUserId: Optional[str] = None
    containerDiskInGb: Optional[int] = None
    containerRegistryAuthId: Optional[str] = None
    costPerHr: Optional[Union[str, float]] = None
    cpuFlavorId: Optional[str] = None
    createdAt: Optional[str] = None
    desiredStatus: Optional[str] = None
    dockerEntrypoint: Optional[List[str]] = None
    dockerStartCmd: Optional[List[str]] = None
    endpointId: Optional[str] = None
    env: Optional[Dict[str, str]] = (
        None  # Environment variables as key-value string pairs
    )
    gpu: Optional[Gpu] = None
    gpuCount: Optional[int] = None
    image: Optional[str] = None
    imageName: Optional[str] = None
    interruptible: Optional[bool] = None
    lastStartedAt: Optional[str] = None
    lastStatusChange: Optional[str] = None
    locked: Optional[bool] = None
    machine: Optional[Machine] = None
    machineId: Optional[str] = None
    memoryInGb: Optional[int] = None
    name: Optional[str] = None
    networkVolume: Optional[NetworkVolume] = None
    portMappings: Optional[Dict[str, int]] = (
        None  # Port number as string key, mapped port as int value
    )
    ports: Optional[List[str]] = None
    publicIp: Optional[str] = None
    savingsPlans: Optional[List[SavingsPlan]] = None
    slsVersion: Optional[int] = None
    templateId: Optional[str] = None
    vcpuCount: Optional[int] = None
    volumeEncrypted: Optional[bool] = None
    volumeInGb: Optional[int] = None
    volumeMountPath: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pod":
        """Create a Pod instance from a dictionary (JSON response)."""

        # Handle nested objects
        gpu = None
        if "gpu" in data and data["gpu"]:
            gpu = Gpu.from_dict(data["gpu"])

        machine = Machine.from_dict(data.get("machine", {}))

        network_volume = None
        if "networkVolume" in data and data["networkVolume"]:
            network_volume = NetworkVolume.from_dict(data["networkVolume"])

        savings_plans = []
        if "savingsPlans" in data and data["savingsPlans"]:
            savings_plans = [
                SavingsPlan.from_dict(plan) for plan in data["savingsPlans"]
            ]

        return cls(
            id=data["id"],
            adjustedCostPerHr=data.get("adjustedCostPerHr"),
            aiApiId=data.get("aiApiId"),
            consumerUserId=data.get("consumerUserId"),
            containerDiskInGb=data.get("containerDiskInGb"),
            containerRegistryAuthId=data.get("containerRegistryAuthId"),
            costPerHr=data.get("costPerHr"),
            cpuFlavorId=data.get("cpuFlavorId"),
            createdAt=data.get("createdAt"),
            desiredStatus=data.get("desiredStatus"),
            dockerEntrypoint=data.get("dockerEntrypoint"),
            dockerStartCmd=data.get("dockerStartCmd"),
            endpointId=data.get("endpointId"),
            env=data.get("env"),
            gpu=gpu,
            gpuCount=data.get("gpuCount"),
            image=data.get("image"),
            imageName=data.get("imageName"),
            interruptible=data.get("interruptible"),
            lastStartedAt=data.get("lastStartedAt"),
            lastStatusChange=data.get("lastStatusChange"),
            locked=data.get("locked"),
            machine=machine,
            machineId=data.get("machineId"),
            memoryInGb=data.get("memoryInGb"),
            name=data.get("name"),
            networkVolume=network_volume,
            portMappings=data.get("portMappings"),
            ports=data.get("ports"),
            publicIp=data.get("publicIp"),
            savingsPlans=savings_plans if savings_plans else None,
            slsVersion=data.get("slsVersion"),
            templateId=data.get("templateId"),
            vcpuCount=data.get("vcpuCount"),
            volumeEncrypted=data.get("volumeEncrypted"),
            volumeInGb=data.get("volumeInGb"),
            volumeMountPath=data.get("volumeMountPath"),
        )

    def is_running(self) -> bool:
        """Check if the pod is currently running."""
        return self.desiredStatus == "RUNNING"

    def get_ssh_connection_string(self) -> Optional[str]:
        """Get SSH connection string if available."""
        # Use direct TCP connection (supports SCP/SFTP)
        if self.publicIp and self.portMappings and "22" in self.portMappings:
            port = self.portMappings["22"]
            return f"ssh -p {port} root@{self.publicIp}"
        return None

    def get_ssh_host(self) -> Optional[str]:
        """Get SSH host (public IP) for terminal commands."""
        return self.publicIp

    def get_ssh_port(self) -> Optional[int]:
        """Get SSH port for terminal commands."""
        if self.portMappings and "22" in self.portMappings:
            return self.portMappings["22"]
        return None

    def get_ssh_details(self) -> Optional[SshDetails]:
        """Get all SSH connection details as a typed dictionary."""
        if not self.publicIp or not self.get_ssh_port():
            return None

        host = self.get_ssh_host()
        port = self.get_ssh_port()
        connection_string = self.get_ssh_connection_string()

        if host is None or port is None or connection_string is None:
            return None

        return SshDetails(
            host=host, port=port, user="root", connection_string=connection_string
        )

    def can_ssh(self) -> bool:
        """Check if SSH connection is possible."""
        return bool(self.publicIp and self.portMappings and "22" in self.portMappings)


def get_pods() -> List[Pod]:
    """Fetch all pods and return as Pod objects."""
    url = "https://rest.runpod.io/v1/pods"
    headers = {"Authorization": f"Bearer {runpod.api_key}"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    pods_data = response.json()
    return [Pod.from_dict(pod_data) for pod_data in pods_data]


# Example usage
# if __name__ == "__main__":
#     pods = get_pods()

#     print(f"Found {len(pods)} pod(s)")

#     for pod in pods:
#         print(f"\nPod: {pod.name} ({pod.id})")
#         print(f"Status: {pod.desiredStatus} | Cost: ${pod.costPerHr}/hr")
#         print(f"Resources: {pod.memoryInGb}GB RAM, {pod.vcpuCount} vCPUs, {pod.gpuCount} GPU(s)")

#         if pod.can_ssh():
#             print(f"SSH: {pod.get_ssh_connection_string()}")
