"""Restricted transport for one explicitly configured replacement GPU pod.

This module never retries a paid create, discovers a different recipe, deletes
resources, or supplies a new budget. The caller must persist a unique attempt
name and arm a watcher against its existing deadline before calling ``create``.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request

import runpod_budget as guard


GPU_ID = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
IMAGE = "runpod/pytorch:1.4.1-cu1320-torch280-ubuntu2404"
MIN_CUDA = "13.2"
RECIPE_FIELDS = frozenset({"gpu_id", "data_center_id", "volume_id", "image",
                           "public_key", "max_hourly_rate"})
EU_DATA_CENTER = re.compile(r"(?:EU|EUR)-[A-Z0-9]+-[0-9]+\Z")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
USABLE_AVAILABILITY = frozenset({"LOW", "MEDIUM", "HIGH"})
MAX_PAGES = 20
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
CREATION_CLOCK_SKEW_SECONDS = 120


class ReplacementError(RuntimeError):
    """An error with a fixed message safe to show without provider content."""


class ReplacementProtocolError(ReplacementError):
    pass


class ReplacementPreflightError(ReplacementError):
    pass


class ReplacementVerificationError(ReplacementError):
    def __init__(self, findings, *, owned_identity):
        self.findings = tuple(findings)
        self.owned_identity = bool(owned_identity)
        super().__init__("Replacement pod did not match its approved recipe: " +
                         ", ".join(self.findings) + ".")


def _positive_rate(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplacementPreflightError("Replacement hourly rate is invalid.")
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise ReplacementPreflightError("Replacement hourly rate is invalid.")
    return rate


def _public_key(value):
    # Only one ordinary Ed25519 public key, with an optional one-line comment.
    # No authorized_keys options, private keys, or environment substitutions.
    if (not isinstance(value, str) or len(value) > 1024 or
            any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ReplacementPreflightError("Replacement SSH public key is invalid.")
    fields = value.split()
    if len(fields) < 2 or fields[0] != "ssh-ed25519":
        raise ReplacementPreflightError("Replacement SSH public key is invalid.")
    try:
        decoded = base64.b64decode(fields[1], validate=True)
    except (ValueError, binascii.Error):
        raise ReplacementPreflightError("Replacement SSH public key is invalid.") from None
    if (len(decoded) != 51 or decoded[:19] !=
            b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20"):
        raise ReplacementPreflightError("Replacement SSH public key is invalid.")
    return value


def validate_recipe(recipe):
    """Validate a private operator recipe; browser input is never a recipe."""
    if not isinstance(recipe, dict) or set(recipe) != RECIPE_FIELDS:
        raise ReplacementPreflightError("Replacement recipe fields are invalid.")
    if recipe["gpu_id"] != GPU_ID or recipe["image"] != IMAGE:
        raise ReplacementPreflightError("Replacement GPU or container image is not approved.")
    if (not isinstance(recipe["data_center_id"], str) or
            not EU_DATA_CENTER.fullmatch(recipe["data_center_id"])):
        raise ReplacementPreflightError("Replacement requires an explicit European data center.")
    if (not isinstance(recipe["volume_id"], str) or
            not guard.POD_ID.fullmatch(recipe["volume_id"])):
        raise ReplacementPreflightError("Replacement volume identity is invalid.")
    rate = _positive_rate(recipe["max_hourly_rate"])
    if rate > guard.MAX_SESSION_HOURLY_RATE_USD:
        raise ReplacementPreflightError("Replacement hourly rate exceeds the budget ceiling.")
    public_key = _public_key(recipe["public_key"])
    return {**recipe, "max_hourly_rate": rate, "public_key": public_key}


def _name(name):
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ReplacementPreflightError("Replacement attempt name is invalid.")
    return name


def build_create_body(recipe, name):
    recipe = validate_recipe(recipe)
    return {
        "name": _name(name), "image": recipe["image"],
        "gpu": {"id": recipe["gpu_id"], "count": 1, "minCudaVersion": MIN_CUDA},
        "cloud": "SECURE", "disk": 30,
        "dataCenterIds": [recipe["data_center_id"]], "ports": ["22/tcp"],
        "mounts": {"network": [{"volumeId": recipe["volume_id"], "path": "/workspace"}]},
        "startSsh": True, "startJupyter": False, "globalNetworking": False,
        "env": {"PUBLIC_KEY": recipe["public_key"]},
    }


def ownership_match(pod, name, created_after):
    """Recognize only the exact persisted nonce name created by this attempt.

    There is deliberately no upper time bound: a delayed provider create must
    still be discovered and stopped after the original session deadline. This
    is not sufficient by itself to authorize adoption; verify the full recipe.
    """
    if (not isinstance(pod, dict) or not isinstance(name, str) or
            not NAME.fullmatch(name) or pod.get("name") != name or
            not isinstance(pod.get("id"), str) or
            not guard.POD_ID.fullmatch(pod["id"])):
        return False
    try:
        after = float(created_after)
        created = guard.timestamp(pod["createdAt"])
        return (math.isfinite(after) and math.isfinite(created) and
                created >= after - CREATION_CLOCK_SKEW_SECONDS)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return False


def verify_created(pod, recipe, name, created_after):
    """Return a matching pod or raise with fixed findings and ownership status.

    A mismatched pod may be stopped only if ``owned_identity`` is true. Its
    resource settings must never be used to mutate an unrelated account pod.
    ``gpu.memory`` is host RAM, so VRAM is checked in the catalog preflight.
    """
    recipe = validate_recipe(recipe)
    _name(name)
    owned = ownership_match(pod, name, created_after)
    if not isinstance(pod, dict):
        raise ReplacementVerificationError(["identity"], owned_identity=False)
    findings = []
    if not owned:
        findings.append("identity")
    gpu = pod.get("gpu")
    if (not isinstance(gpu, dict) or gpu.get("id") != recipe["gpu_id"] or
            type(gpu.get("count")) is not int or gpu["count"] != 1 or pod.get("cpu") is not None):
        findings.append("GPU")
    if pod.get("cloud") != "SECURE" or pod.get("dataCenterId") != recipe["data_center_id"]:
        findings.append("location")
    cuda = pod.get("cudaVersion")
    if cuda is not None:
        parsed = re.fullmatch(r"([0-9]{1,3})\.([0-9]{1,3})", cuda) if isinstance(cuda, str) else None
        if parsed is None or tuple(map(int, parsed.groups())) < (13, 2):
            findings.append("CUDA version")
    if pod.get("image") != recipe["image"]:
        findings.append("image")
    if type(pod.get("disk")) is not int or pod["disk"] != 30:
        findings.append("disk")
    expected_mounts = {"network": [{"volumeId": recipe["volume_id"], "path": "/workspace"}]}
    if pod.get("mounts") != expected_mounts:
        findings.append("network volume")
    if pod.get("ports") != ["22/tcp"]:
        findings.append("ports")
    if pod.get("env") != {"PUBLIC_KEY": recipe["public_key"]}:
        findings.append("environment")
    if (pod.get("args") != "" or pod.get("entrypoint") not in (None, []) or
            pod.get("cmd") not in (None, []) or pod.get("registry") is not None or
            pod.get("template") is not None):
        findings.append("container settings")
    networking = pod.get("globalNetworking")
    if not isinstance(networking, dict) or networking.get("enabled") is not False:
        findings.append("networking")
    if pod.get("cluster") is not None or pod.get("locked") is not False:
        findings.append("management settings")
    try:
        rate = _positive_rate(pod.get("cost"))
        if rate > recipe["max_hourly_rate"]:
            findings.append("hourly rate")
    except ReplacementPreflightError:
        findings.append("hourly rate")
    if findings:
        raise ReplacementVerificationError(findings, owned_identity=owned)
    return pod


class ReplacementAPI:
    """Use the guard's token and no-redirect opener without paid retries."""

    def __init__(self, api, *, preflight_diagnostic=None):
        self.api = api
        self.preflight_diagnostic = preflight_diagnostic

    def _catalog_for_preflight(self, recipe):
        gpu = None
        try:
            gpu = self.get_gpu(recipe)
            return gpu
        finally:
            if self.preflight_diagnostic is not None:
                # Diagnostics are optional and cannot change a paid-operation
                # decision or hide the original provider error.
                try:
                    def scalar(value):
                        if isinstance(value, str):
                            return ''.join(c for c in value[:128] if c.isprintable())
                        if value is None or isinstance(value, bool):
                            return value
                        if isinstance(value, (int, float)) and math.isfinite(value):
                            return value
                        return None

                    catalog = gpu if isinstance(gpu, dict) else {}
                    price = catalog.get('price')
                    centers = catalog.get('dataCenters')
                    self.preflight_diagnostic({
                        'recipe_data_center_id': recipe['data_center_id'],
                        'catalog_received': gpu is not None,
                        'catalog': {
                            'id': scalar(catalog.get('id')),
                            'memory': scalar(catalog.get('memory')),
                            'availability': scalar(catalog.get('availability')),
                            'price': {key: scalar(price.get(key)) for key in ('secure', 'community')}
                                     if isinstance(price, dict) else None,
                            'dataCenters': [
                                {key: scalar(center.get(key)) for key in ('id', 'availability')}
                                if isinstance(center, dict) else None for center in centers[:64]
                            ] if isinstance(centers, list) else None,
                            'dataCenters_count': len(centers) if isinstance(centers, list) else None,
                            'dataCenters_truncated': isinstance(centers, list) and len(centers) > 64,
                        },
                    })
                except Exception:
                    pass

    def _request(self, method, path, *, query=None, payload=None):
        endpoint = guard.API + path
        if query:
            endpoint += "?" + urlencode(query)
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(endpoint, data=data, method=method, headers={
            "Authorization": f"Bearer {self.api.key}", "Content-Type": "application/json",
        })
        try:
            with self.api.opener.open(request, timeout=30) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            status = exc.code
            try:
                body = exc.read(16385)
            except Exception:
                body = b""
            finally:
                exc.close()
            raise guard.RunpodHTTPError(status, guard.safe_http_message(status, body)) from None
        except (URLError, TimeoutError, OSError):
            raise ReplacementError("Runpod API is unreachable.") from None
        if len(body) > MAX_RESPONSE_BYTES:
            raise ReplacementProtocolError("Runpod response exceeds the permitted size.")
        try:
            result = json.loads(body)
        except (ValueError, UnicodeError):
            raise ReplacementProtocolError("Runpod returned an invalid response.") from None
        if not isinstance(result, dict):
            raise ReplacementProtocolError("Runpod returned an invalid response.")
        return result

    def get_gpu(self, recipe):
        recipe = validate_recipe(recipe)
        return self._request("GET", "/catalog/gpus/" + quote(recipe["gpu_id"], safe=""), query={
            "include": "AVAILABILITY", "product": "POD", "count": 1,
            "cloud": "SECURE", "minCudaVersion": MIN_CUDA,
        })

    def get_volume(self, volume_id):
        if not isinstance(volume_id, str) or not guard.POD_ID.fullmatch(volume_id):
            raise ReplacementPreflightError("Replacement volume identity is invalid.")
        return self._request("GET", "/network-volumes/" + quote(volume_id, safe=""))

    def list_pods(self):
        """Return a complete account listing or fail; never return a partial list."""
        pods, ids, cursors = [], set(), set()
        cursor = None
        for _ in range(MAX_PAGES):
            query = {"limit": 1000, "includeClusterPods": "true"}
            if cursor is not None:
                query["cursor"] = cursor
            page = self._request("GET", "/pods", query=query)
            items, pagination = page.get("pods"), page.get("pagination")
            if (not isinstance(items, list) or not isinstance(pagination, dict) or
                    type(pagination.get("hasNextPage")) is not bool):
                raise ReplacementProtocolError("Runpod pod listing is incomplete.")
            for pod in items:
                if (not isinstance(pod, dict) or not isinstance(pod.get("id"), str) or
                        not guard.POD_ID.fullmatch(pod["id"]) or pod["id"] in ids):
                    raise ReplacementProtocolError("Runpod pod listing is inconsistent.")
                ids.add(pod["id"])
                pods.append(pod)
            cursor = pagination.get("nextCursor")
            if not pagination["hasNextPage"]:
                if cursor is not None:
                    raise ReplacementProtocolError("Runpod pod listing is inconsistent.")
                return pods
            if not isinstance(cursor, str) or not cursor or len(cursor) > 4096 or cursor in cursors:
                raise ReplacementProtocolError("Runpod pod listing is incomplete.")
            cursors.add(cursor)
        raise ReplacementProtocolError("Runpod pod listing exceeded the safe page limit.")

    def preflight(self, recipe):
        recipe = validate_recipe(recipe)
        gpu = self._catalog_for_preflight(recipe)
        volume = self.get_volume(recipe["volume_id"])
        if (gpu.get("id") != recipe["gpu_id"] or gpu.get("secure") is not True or
                gpu.get("manufacturer") != "NVIDIA" or type(gpu.get("memory")) is not int or
                gpu["memory"] != 96):
            raise ReplacementPreflightError("Approved GPU is unavailable in the secure catalog.")
        price = gpu.get("price")
        rate = _positive_rate(price.get("secure") if isinstance(price, dict) else None)
        if rate > recipe["max_hourly_rate"]:
            raise ReplacementPreflightError("Replacement quote exceeds its configured hourly ceiling.")
        centers = gpu.get("dataCenters")
        matching = [center for center in centers if isinstance(center, dict) and
                    center.get("id") == recipe["data_center_id"]] if isinstance(centers, list) else []
        overall = gpu.get("availability")
        regional = matching[0].get("availability") if len(matching) == 1 else None
        # Placement is pinned to the volume's data center. The catalog exposes
        # overall and regional stock separately and does not guarantee they
        # agree, so only that exact region determines whether it can be tried.
        if overall is not None and (not isinstance(overall, str) or
                                    overall not in USABLE_AVAILABILITY | {"NONE"}):
            raise ReplacementPreflightError("Runpod returned invalid GPU availability.")
        if not isinstance(regional, str) or regional not in USABLE_AVAILABILITY:
            raise ReplacementPreflightError("No GPU is available right now. Try again shortly.")
        if volume.get("id") != recipe["volume_id"] or volume.get("dataCenter") != recipe["data_center_id"]:
            raise ReplacementPreflightError("Replacement network volume does not match its data center.")
        if type(volume.get("size")) is not int or volume["size"] != 150 or volume.get("type") != "STANDARD":
            raise ReplacementPreflightError("Replacement requires its existing 150 GB standard network volume.")
        return {"gpu_id": recipe["gpu_id"], "data_center_id": recipe["data_center_id"],
                "volume_id": recipe["volume_id"], "quoted_rate": rate,
                "memory_gb": 96, "availability": matching[0]["availability"]}

    def create(self, recipe, name):
        # One request only. A lost response may still mean a billed pod exists.
        return self._request("POST", "/pods", payload=build_create_body(recipe, name))
