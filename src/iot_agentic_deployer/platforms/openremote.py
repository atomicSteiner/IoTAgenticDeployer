"""OpenRemote adapter

The platform that puts the boundary to the test: it has neither a device
entity nor a relation entity. Everything is an asset, and containment is the
child's own parentId, so `create_relation` creates nothing and updates the
child instead. The plan above does not know, and does not change.

`scope` and `entity_type` are ThingsBoard's vocabulary, which planning still
speaks: ignored here on purpose, and a truer account of how far the
abstraction reaches than renaming them everywhere would be
"""

import os

import requests

from iot_agentic_deployer.platforms.base import PlatformAdapter, register_adapter

# Planning names the spatial levels; OpenRemote names asset types. A floor is
# not a GroupAsset: that is a set of assets of one type and insists on saying
# which. OpenRemote has no floor, so it falls back to the generic thing
SPATIAL_TYPES = {
    "building": "BuildingAsset",
    "floor": "ThingAsset",
    "space": "RoomAsset",
    "access_point": "ThingAsset",
}
FALLBACK_TYPE = "ThingAsset"

TIMEOUT = 20


class OpenRemoteError(RuntimeError):
    """An operation the platform refused. Raised so realisation stops there
    with the reason, instead of marking it done with an id it never got"""


@register_adapter
class OpenRemoteAdapter(PlatformAdapter):

    name = "openremote"

    def __init__(self, endpoint: str | None = None, realm: str | None = None, **_ignored):
        # _ignored takes the options meant for the other adapter: the workflow
        # hands the same dict to whichever platform the architect picked
        self.endpoint = (endpoint or os.getenv("OPENREMOTE_URL",
                                               "http://localhost:8080")).rstrip("/")
        # Keycloak answers on its own port, not behind the manager: the manager
        # advertises authServerUrl '/auth', but nothing serves it there
        self.auth_url = os.getenv("OPENREMOTE_AUTH_URL",
                                  "http://localhost:8081/auth").rstrip("/")
        self.realm = realm or os.getenv("OPENREMOTE_REALM", "master")
        self.client_id = os.getenv("OPENREMOTE_CLIENT_ID", "")
        self.client_secret = os.getenv("OPENREMOTE_CLIENT_SECRET", "")
        self._token = None
        self._model = None      # the platform's own asset model, fetched once

    # -- http plumbing ----------------------------------------------------

    def _authenticate(self) -> str:
        """A service user's client credentials, exchanged for a token. Kept
        until something comes back 401, which is when it has expired"""
        if self._token:
            return self._token
        if not self.client_id:
            raise OpenRemoteError(
                "No OPENREMOTE_CLIENT_ID set. Create a service user in the OpenRemote "
                "console and put its client id and secret in .env.")

        response = requests.post(
            f"{self.auth_url}/realms/{self.realm}/protocol/openid-connect/token",
            data={"grant_type": "client_credentials",
                  "client_id": self.client_id,
                  "client_secret": self.client_secret},
            timeout=TIMEOUT)
        if response.status_code != 200:
            raise OpenRemoteError(
                f"Authentication refused ({response.status_code}): {response.text[:200]}")
        self._token = response.json()["access_token"]
        return self._token

    def _call(self, method: str, path: str, **kwargs):
        """One request, retried once without the cached token: the only
        failure worth distinguishing here is an expired one"""
        for attempt in (1, 2):
            response = requests.request(
                method, f"{self.endpoint}/api/{self.realm}{path}",
                headers={"Authorization": f"Bearer {self._authenticate()}"},
                timeout=TIMEOUT, **kwargs)
            if response.status_code == 401 and attempt == 1:
                self._token = None
                continue
            if response.status_code >= 400:
                raise OpenRemoteError(
                    f"{method} {path} refused ({response.status_code}): {response.text[:200]}")
            return response.json() if response.content else None

    def _find_by_name(self, name: str):
        """The existing asset of that name, if there is one. Reusing it is
        what lets an interrupted run be confirmed again without colliding
        with what it already created"""
        try:
            found = self._call("POST", "/asset/query",
                               json={"names": [{"predicateType": "string", "value": name}]})
        except OpenRemoteError:
            return None
        return found[0] if found else None

    # -- contract ---------------------------------------------------------

    def check_reachability(self) -> tuple[bool, str]:
        try:
            # Authenticate first: reading is allowed without a token, so a
            # query alone would call it reachable and then fail on the first
            # write, halfway through the plan
            self._authenticate()
            self._call("POST", "/asset/query", json={"limit": 1})
        except Exception as e:
            return False, (
                f"OpenRemote at {self.endpoint} did not answer: {type(e).__name__}: {e}\n\n"
                "Start it with `docker compose -f docker-compose.openremote.yaml up -d` "
                "from src/iot_agentic_deployer, and check that OPENREMOTE_CLIENT_ID and "
                "OPENREMOTE_CLIENT_SECRET in .env belong to a service user of the "
                f"'{self.realm}' realm."
            )
        return True, f"OpenRemote reachable at {self.endpoint} (realm '{self.realm}')."

    def _required_attributes(self, asset_type: str) -> dict:
        """What this asset type will not be created without.

        A BuildingAsset wants a postal code, a PeopleCounterAsset seven
        counters. The model above knows none of that and must not invent it,
        so they are declared and left empty. Read from the manager, so a new
        device type needs no change here"""
        if self._model is None:
            self._model = {
                info["assetDescriptor"]["name"]: {
                    a["name"]: a.get("type")
                    for a in info.get("attributeDescriptors", []) if not a.get("optional")
                }
                for info in self._call("GET", "/model/assetInfos")
                if info.get("assetDescriptor", {}).get("name")
            }
        return self._model.get(asset_type, {})

    def create_asset(self, name: str, asset_type: str) -> str:
        existing = self._find_by_name(name)
        if existing:
            return existing["id"]

        kind = SPATIAL_TYPES.get(asset_type, asset_type)
        created = self._call("POST", "/asset", json={
            "name": name,
            "type": kind,
            "realm": self.realm,
            "attributes": {n: {"name": n, "type": t}
                           for n, t in self._required_attributes(kind).items()},
        })
        return created["id"]

    def create_device(self, name: str, device_type: str, label: str | None = None) -> str:
        # Never reached while the catalogue maps every device to an asset,
        # but a platform is not free to answer only the questions it likes
        return self.create_asset(name, device_type or FALLBACK_TYPE)

    def set_attributes(self, entity_id: str, entity_type: str, scope: str,
                       attributes: dict) -> None:
        """The whole asset, not one attribute at a time: writing an attribute
        the type never declared is refused, and ours are the model's own
        metadata, which no OpenRemote type has heard of. Declared as text on
        the asset they go through, and what landed stays traceable"""
        if not attributes:
            return
        asset = self._call("GET", f"/asset/{entity_id}")
        asset.setdefault("attributes", {})
        for key, value in attributes.items():
            asset["attributes"][key] = {"name": key, "type": "text", "value": str(value)}
        self._call("PUT", f"/asset/{entity_id}", json=asset)

    def create_relation(self, from_id: str, from_type: str,
                        to_id: str, to_type: str) -> None:
        """Containment is not a thing you create here, it is where the child
        sits: read it, move it under its parent, write it back"""
        child = self._call("GET", f"/asset/{to_id}")
        child["parentId"] = from_id
        self._call("PUT", f"/asset/{to_id}", json=child)

    # -- reading ----------------------------------------------------------
    # Not provisioning: just for looking at what ended up on the platform

    def get_all_assets(self) -> list:
        return self._call("POST", "/asset/query", json={}) or []
