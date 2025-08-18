"""
Authentication management for Keys & Caches.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import webbrowser

from .client import APIClient, AuthenticationError


class AuthManager:
    """Manages authentication credentials and settings."""

    def __init__(self):
        self.config_dir = Path.home() / ".kandc"
        self.settings_file = self.config_dir / "settings.json"
        self.config_dir.mkdir(exist_ok=True)

    def load_settings(self) -> Dict[str, Any]:
        """Load settings from config file."""
        if not self.settings_file.exists():
            return {}

        try:
            with open(self.settings_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def save_settings(self, settings: Dict[str, Any]):
        """Save settings to config file."""
        try:
            with open(self.settings_file, "w") as f:
                json.dump(settings, f, indent=2)
        except IOError as e:
            print(f"⚠️  Warning: Could not save settings: {e}")

    def get_api_key(self) -> Optional[str]:
        """Get stored API key."""
        settings = self.load_settings()
        return settings.get("api_key")

    def set_api_key(self, api_key: str, email: str = None):
        """Store API key and email."""
        settings = self.load_settings()
        settings["api_key"] = api_key
        if email:
            settings["email"] = email
        self.save_settings(settings)

    def clear_credentials(self):
        """Clear stored credentials."""
        settings = self.load_settings()
        settings.pop("api_key", None)
        settings.pop("email", None)
        self.save_settings(settings)

    def verify_api_key(self, api_key: str) -> bool:
        """Verify if API key is valid by testing with backend."""
        if not api_key:
            print("⚠️  No API key provided")
            return False

        try:
            client = APIClient(api_key=api_key)
            # Test API key by trying to access projects endpoint
            print("🔍 Verifying API key with backend...")
            client._request("GET", "/api/v1/projects")
            print("✅ API key verified successfully")
            return True
        except AuthenticationError as e:
            print("❌ API key validation failed")
            print(f"   Error: {str(e)}")
            print("   Please check your credentials and try again")
            return False
        except Exception as e:
            # Only assume valid if it's a network error
            if "Connection" in str(e):
                print(f"⚠️  Network error during validation:")
                print(f"   Error: {str(e)}")
                print("   Assuming API key is valid (offline mode)")
                return True
            print(f"❌ API key validation error:")
            print(f"   Error: {str(e)}")
            print("   Backend URL:", client.base_url)
            print("   Please check your network connection and backend status")
            return False

    def ensure_authenticated(self) -> APIClient:
        """
        Ensure user is authenticated and return API client.

        Returns:
            APIClient: Authenticated API client

        Raises:
            AuthenticationError: If authentication fails
        """
        # Try existing API key
        api_key = self.get_api_key()

        if api_key and self.verify_api_key(api_key):
            print(f"✅ Using existing authentication")
            settings = self.load_settings()
            if email := settings.get("email"):
                print(f"   Logged in as: {email}")
            return APIClient(api_key=api_key)

        # Need to authenticate
        print("🔐 Authentication required")

        if api_key:
            print("   Existing credentials are invalid")
            self.clear_credentials()

        # Use the backend auth init endpoint to get proper session ID and auth URL
        client = APIClient()  # No API key yet
        try:
            # Get auth URL from backend (this creates the session ID)
            auth_response = client._request("GET", "/api/v1/auth/init")
            auth_url = auth_response.get("auth_url")
            session_id = auth_response.get("session_id")

            if not auth_url or not session_id:
                raise AuthenticationError("Invalid response from auth init endpoint")

            print("🌐 Please sign in with your Google account in the browser")
            print(f"   Opening: {auth_url}")
            print("   After signing in, your CLI will be automatically authenticated")

            try:
                webbrowser.open(auth_url)
            except Exception as e:
                print(f"⚠️  Could not open browser automatically: {e}")
                print(f"   Please manually visit: {auth_url}")

            # Poll for authentication completion (like the old flow)
            print("⏳ Waiting for authentication...")
            max_attempts = 60  # 5 minutes with 5-second intervals

            for attempt in range(max_attempts):
                remaining = max_attempts - attempt
                # Only show progress every 10 attempts to reduce noise
                if attempt % 10 == 0:
                    print(f"   ⏰ {remaining} attempts remaining ({remaining * 5}s)...")

                try:
                    # Check auth status
                    auth_status = client._request("GET", f"/api/v1/auth/check/{session_id}")

                    if auth_status.get("authenticated"):
                        api_key = auth_status.get("api_key")
                        if api_key:
                            print("🎉 Authentication successful!")
                            print(f"   Email: {auth_status.get('email', 'Unknown')}")

                            # Store the API key
                            email = auth_status.get("email")
                            if email:
                                self.set_api_key(api_key, email)
                                print(f"✅ Authenticated as {email}")
                            else:
                                self.set_api_key(api_key)

                            print(f"💾 Credentials saved to {self.settings_file}")
                            return APIClient(api_key=api_key)
                        else:
                            raise AuthenticationError(
                                "Authentication succeeded but no API key returned"
                            )

                    # Wait before next check
                    import time

                    time.sleep(5)

                except Exception:
                    # Continue polling on API errors
                    import time

                    time.sleep(5)
                    continue

            raise AuthenticationError("Authentication timed out. Please try again.")

        except AuthenticationError as e:
            raise AuthenticationError(f"Authentication failed: {e}")

    def get_dashboard_url(self, project_id: str = None, run_id: str = None) -> str:
        """Get dashboard URL."""
        api_key = self.get_api_key()
        if not api_key:
            raise AuthenticationError("Not authenticated")

        client = APIClient(api_key=api_key)
        return client.get_dashboard_url(project_id, run_id)

    def open_dashboard(self, project_id: str = None, run_id: str = None):
        """Open dashboard in browser."""
        try:
            url = self.get_dashboard_url(project_id, run_id)
            print(f"🌐 Opening dashboard: {url}")
            webbrowser.open(url)
        except Exception as e:
            print(f"⚠️  Could not open dashboard: {e}")


# Global auth manager instance
_auth_manager = AuthManager()


def get_auth_manager() -> AuthManager:
    """Get the global auth manager instance."""
    return _auth_manager


def get_api_key() -> Optional[str]:
    """Get the current API key."""
    return _auth_manager.get_api_key()


def ensure_authenticated() -> APIClient:
    """Ensure user is authenticated and return API client."""
    return _auth_manager.ensure_authenticated()
