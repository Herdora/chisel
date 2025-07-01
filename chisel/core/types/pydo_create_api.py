from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

"""
This file provides dataclass-based representations for DigitalOcean Droplet creation and response models.

For full details on the DigitalOcean API, see:
https://docs.digitalocean.com/reference/api/digitalocean/#tag/Droplets/operation/droplets_create
"""


@dataclass
class DiskInfo:
    type: str
    size: int
    allocated: bool


@dataclass
class Kernel:
    id: int
    name: str
    version: str


@dataclass
class BackupWindow:
    start: Optional[str] = None
    end: Optional[str] = None


@dataclass
class Image:
    id: int
    name: str
    distribution: str
    slug: Optional[str] = None
    public: bool = False
    regions: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    min_disk_size: Optional[int] = None
    type: Optional[str] = None
    size_gigabytes: Optional[float] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    status: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class Size:
    slug: str
    memory: int
    vcpus: int
    disk: int
    transfer: float
    price_monthly: float
    price_hourly: float
    regions: List[str] = field(default_factory=list)
    available: bool = True
    description: Optional[str] = None


@dataclass
class NetworkV4:
    ip_address: str
    netmask: str
    gateway: str
    type: str  # "public" or "private"


@dataclass
class NetworkV6:
    ip_address: str
    netmask: int
    gateway: str
    type: str  # "public"


@dataclass
class Networks:
    v4: List[NetworkV4] = field(default_factory=list)
    v6: List[NetworkV6] = field(default_factory=list)


@dataclass
class Region:
    name: str
    slug: str
    sizes: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    available: bool = True


@dataclass
class GPUInfo:
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None
    gpu_memory_gb: Optional[int] = None
    gpu_vendor: Optional[str] = None


@dataclass
class ActionLink:
    id: int
    rel: str
    href: str


@dataclass
class PydoDropletObject:
    id: int
    name: str
    memory: int
    vcpus: int
    disk: int
    locked: bool
    status: str  # "new", "active", "off", "archive"
    created_at: str
    features: List[str]
    backup_ids: List[int]
    next_backup_window: Optional[BackupWindow]
    snapshot_ids: List[int]
    image: Image
    volume_ids: List[str]
    size: Size
    size_slug: str
    networks: Networks
    region: Region
    tags: List[str]
    disk_info: List[DiskInfo] = field(default_factory=list)
    kernel: Optional[Kernel] = None  # deprecated
    vpc_uuid: Optional[str] = None
    gpu_info: Optional[GPUInfo] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PydoDropletObject":
        backup_window = None
        if data.get("next_backup_window"):
            backup_window_data = data["next_backup_window"]
            backup_window = BackupWindow(
                start=backup_window_data.get("start"),
                end=backup_window_data.get("end"),
            )

        disk_info = []
        if data.get("disk_info"):
            disk_info = [
                DiskInfo(
                    type=disk.get("type", ""),
                    size=disk.get("size", 0),
                    allocated=disk.get("allocated", False),
                )
                for disk in data["disk_info"]
            ]

        kernel = None
        if data.get("kernel"):
            kernel_data = data["kernel"]
            kernel = Kernel(
                id=kernel_data.get("id", 0),
                name=kernel_data.get("name", ""),
                version=kernel_data.get("version", ""),
            )

        image_data = data["image"]
        image = Image(
            id=image_data.get("id", 0),
            name=image_data.get("name", ""),
            distribution=image_data.get("distribution", ""),
            slug=image_data.get("slug"),
            public=image_data.get("public", False),
            regions=image_data.get("regions", []),
            created_at=image_data.get("created_at"),
            min_disk_size=image_data.get("min_disk_size"),
            type=image_data.get("type"),
            size_gigabytes=image_data.get("size_gigabytes"),
            description=image_data.get("description"),
            tags=image_data.get("tags", []),
            status=image_data.get("status"),
            error_message=image_data.get("error_message"),
        )

        size_data = data["size"]
        size = Size(
            slug=size_data.get("slug", ""),
            memory=size_data.get("memory", 0),
            vcpus=size_data.get("vcpus", 0),
            disk=size_data.get("disk", 0),
            transfer=size_data.get("transfer", 0.0),
            price_monthly=size_data.get("price_monthly", 0.0),
            price_hourly=size_data.get("price_hourly", 0.0),
            regions=size_data.get("regions", []),
            available=size_data.get("available", True),
            description=size_data.get("description"),
        )

        networks_data = data.get("networks", {})
        v4_networks = []
        if networks_data.get("v4"):
            v4_networks = [
                NetworkV4(
                    ip_address=net.get("ip_address", ""),
                    netmask=net.get("netmask", ""),
                    gateway=net.get("gateway", ""),
                    type=net.get("type", ""),
                )
                for net in networks_data["v4"]
            ]

        v6_networks = []
        if networks_data.get("v6"):
            v6_networks = [
                NetworkV6(
                    ip_address=net.get("ip_address", ""),
                    netmask=net.get("netmask", 0),
                    gateway=net.get("gateway", ""),
                    type=net.get("type", ""),
                )
                for net in networks_data["v6"]
            ]

        networks = Networks(v4=v4_networks, v6=v6_networks)

        region_data = data["region"]
        region = Region(
            name=region_data.get("name", ""),
            slug=region_data.get("slug", ""),
            sizes=region_data.get("sizes", []),
            features=region_data.get("features", []),
            available=region_data.get("available", True),
        )

        gpu_info = None
        if data.get("gpu_info"):
            gpu_data = data["gpu_info"]
            gpu_info = GPUInfo(
                gpu_type=gpu_data.get("gpu_type"),
                gpu_count=gpu_data.get("gpu_count"),
                gpu_memory_gb=gpu_data.get("gpu_memory_gb"),
                gpu_vendor=gpu_data.get("gpu_vendor"),
            )

        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            memory=data.get("memory", 0),
            vcpus=data.get("vcpus", 0),
            disk=data.get("disk", 0),
            locked=data.get("locked", False),
            status=data.get("status", ""),
            created_at=data.get("created_at", ""),
            features=data.get("features", []),
            backup_ids=data.get("backup_ids", []),
            next_backup_window=backup_window,
            snapshot_ids=data.get("snapshot_ids", []),
            image=image,
            volume_ids=data.get("volume_ids", []),
            size=size,
            size_slug=data.get("size_slug", ""),
            networks=networks,
            region=region,
            tags=data.get("tags", []),
            disk_info=disk_info,
            kernel=kernel,
            vpc_uuid=data.get("vpc_uuid"),
            gpu_info=gpu_info,
        )


@dataclass
class BackupPolicy:
    plan: str  # "daily", "weekly"
    weekday: Optional[str] = None
    hour: Optional[int] = None  # 0-23


@dataclass
class DropletCreateRequest:
    name: str
    size: str
    image: Union[str, int]
    region: Optional[str] = None
    ssh_keys: List[Union[str, int]] = field(default_factory=list)
    backups: bool = False
    backup_policy: Optional[BackupPolicy] = None
    ipv6: bool = False
    monitoring: bool = False
    tags: List[str] = field(default_factory=list)
    user_data: Optional[str] = None
    private_networking: bool = False  # deprecated
    volumes: List[str] = field(default_factory=list)
    vpc_uuid: Optional[str] = None
    with_droplet_agent: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result


@dataclass
class RateLimitHeaders:
    limit: int
    remaining: int
    reset: int


@dataclass
class DigitalOceanError:
    id: str
    message: str
    request_id: Optional[str] = None


@dataclass
class Links:
    actions: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    pages: Optional[Dict[str, str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Links":
        return cls(actions=data.get("actions", []), pages=data.get("pages"))


@dataclass
class DropletCreateResponse:
    droplet: PydoDropletObject
    links: Links
    rate_limit: Optional[RateLimitHeaders] = None

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> "DropletCreateResponse":
        rate_limit = None
        if headers:
            try:
                rate_limit = RateLimitHeaders(
                    limit=int(headers.get("ratelimit-limit", 0)),
                    remaining=int(headers.get("ratelimit-remaining", 0)),
                    reset=int(headers.get("ratelimit-reset", 0)),
                )
            except (ValueError, TypeError):
                pass

        return cls(
            droplet=PydoDropletObject.from_dict(data["droplet"]),
            links=Links.from_dict(data.get("links", {})),
            rate_limit=rate_limit,
        )


@dataclass
class MultipleDropletCreateResponse:
    droplets: List[PydoDropletObject]
    links: Links
    rate_limit: Optional[RateLimitHeaders] = None

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> "MultipleDropletCreateResponse":
        rate_limit = None
        if headers:
            try:
                rate_limit = RateLimitHeaders(
                    limit=int(headers.get("ratelimit-limit", 0)),
                    remaining=int(headers.get("ratelimit-remaining", 0)),
                    reset=int(headers.get("ratelimit-reset", 0)),
                )
            except (ValueError, TypeError):
                pass

        return cls(
            droplets=[
                PydoDropletObject.from_dict(droplet)
                for droplet in data["droplets"]
            ],
            links=Links.from_dict(data.get("links", {})),
            rate_limit=rate_limit,
        )
