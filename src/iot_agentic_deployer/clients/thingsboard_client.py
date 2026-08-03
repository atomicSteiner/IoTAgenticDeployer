import os
import requests


class ThingsBoardClient:
    """Thin wrapper around the ThingsBoard REST API
    Handles auth once, exposes small purpose-built methods
    """

    def __init__(self, base_url=None, username=None, password=None):
        self.base_url = (base_url or os.getenv("TB_BASE_URL", "http://localhost:9090")).rstrip("/")
        self.username = username or os.getenv("TB_USERNAME", "tenant@thingsboard.org")
        self.password = password or os.getenv("TB_PASSWORD", "tenant")
        self._token = None

    @property
    def _headers(self):
        if self._token is None:
            self._authenticate()
        return {"X-Authorization": f"Bearer {self._token}"}

    def _authenticate(self):
        resp = requests.post(
            f"{self.base_url}/api/auth/login",
            json={"username": self.username, "password": self.password},
        )
        resp.raise_for_status()
        self._token = resp.json()["token"]

    def _get(self, path, **kwargs):
        resp = requests.get(f"{self.base_url}{path}", headers=self._headers, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path, json=None):
        resp = requests.post(f"{self.base_url}{path}", json=json, headers=self._headers)
        resp.raise_for_status()
        return resp.json() if resp.content else None

    #device operations used by the workflow nodes

    def get_all_devices(self):
        devices, page = [], 0
        while True:
            data = self._get("/api/tenant/devices", params={"pageSize": 100, "page": page})
            devices.extend(data["data"])
            if not data["hasNext"]:
                return devices
            page += 1

    def get_device_attributes(self, device_id):
        return self._get(f"/api/plugins/telemetry/DEVICE/{device_id}/values/attributes")

    def get_device_telemetry(self, device_id):
        return self._get(f"/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries")

    def create_device(self, name, device_type="default", label=None):
        payload = {"name": name, "type": device_type}
        if label:
            payload["label"] = label
        return self._post("/api/device", json=payload)

    def push_attributes(self, device_id, scope, attributes):
        self._post(f"/api/plugins/telemetry/DEVICE/{device_id}/attributes/{scope}", json=attributes)
