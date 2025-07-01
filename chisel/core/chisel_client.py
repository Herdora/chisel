import requests
from typing import Optional, Dict, Any
import os

class ChiselClient:
    """Client for communicating with Chisel backend API"""
    
    def __init__(self, chisel_token: str, api_url: Optional[str] = None):
        self.chisel_token = chisel_token
        self.api_url = api_url or os.getenv("CHISEL_API_URL", "https://chisel-production-0736.up.railway.app")
        
    def validate_token(self) -> Dict[str, Any]:
        """Validate chisel token and get user info"""
        try:
            response = requests.post(
                f"{self.api_url}/auth/validate",
                json={"chisel_token": self.chisel_token},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"valid": False, "error": str(e)}
    
    def get_do_token(self, gpu_type: str = "nvidia-h100") -> Dict[str, Any]:
        """Exchange chisel token for DigitalOcean token"""
        try:
            response = requests.post(
                f"{self.api_url}/auth/do-token",
                json={"chisel_token": self.chisel_token, "gpu_type": gpu_type},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"do_token": None, "error": str(e)}
    
    def track_usage(self, droplet_id: str, gpu_type: str, duration_minutes: int) -> Dict[str, Any]:
        """Track droplet usage and deduct credits"""
        try:
            response = requests.post(
                f"{self.api_url}/usage/track",
                json={
                    "chisel_token": self.chisel_token,
                    "droplet_id": droplet_id,
                    "gpu_type": gpu_type,
                    "duration_minutes": duration_minutes
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def get_user_status(self) -> Dict[str, Any]:
        """Get user credits and usage info"""
        try:
            response = requests.get(
                f"{self.api_url}/users/{self.chisel_token}/status",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def check_credits(self, estimated_cost: float) -> bool:
        """Check if user has enough credits for operation"""
        status = self.get_user_status()
        if "error" in status:
            return False
        
        credits_remaining = status.get("credits_remaining", 0)
        return credits_remaining >= estimated_cost