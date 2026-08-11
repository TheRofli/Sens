"""URL capture: screenshot, DOM/CSS styles, CSS animations, scroll motion events."""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import re
import socket
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np


SOURCE_RASTER_MAX_ASSETS = 12
SOURCE_RASTER_MAX_BYTES = 12 * 1024 * 1024
SOURCE_RASTER_TOTAL_MAX_BYTES = 32 * 1024 * 1024
SOURCE_RASTER_MEDIA_SUFFIXES = {
    "image/avif": ".avif",
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
SOURCE_VECTOR_MAX_ASSETS = 12
SOURCE_VECTOR_MAX_BYTES = 512 * 1024
SOURCE_VECTOR_TOTAL_MAX_BYTES = 3 * 1024 * 1024
SOURCE_VECTOR_MIN_VIEWPORT_AREA_RATIO = 0.003
SOURCE_VECTOR_MIN_VISIBLE_RATIO = 0.75
SOURCE_FONT_MAX_ASSETS = 16
SOURCE_FONT_MAX_BYTES = 4 * 1024 * 1024
SOURCE_FONT_TOTAL_MAX_BYTES = 24 * 1024 * 1024
SOURCE_FONT_MEDIA = {
    "font/woff2": (".woff2", "woff2"),
    "application/font-woff2": (".woff2", "woff2"),
    "font/woff": (".woff", "woff"),
    "application/font-woff": (".woff", "woff"),
    "application/x-font-woff": (".woff", "woff"),
    "font/ttf": (".ttf", "truetype"),
    "application/x-font-ttf": (".ttf", "truetype"),
    "font/otf": (".otf", "opentype"),
    "application/x-font-opentype": (".otf", "opentype"),
}
SOURCE_FONT_CANONICAL_MEDIA = {
    "woff2": "font/woff2",
    "woff": "font/woff",
    "truetype": "font/ttf",
    "opentype": "font/otf",
}
_SOURCE_VECTOR_ALLOWED_TAGS = {
    "circle",
    "clippath",
    "defs",
    "ellipse",
    "g",
    "lineargradient",
    "line",
    "mask",
    "path",
    "polygon",
    "polyline",
    "radialgradient",
    "rect",
    "stop",
    "svg",
    "use",
}
_SOURCE_VECTOR_ALLOWED_ATTRIBUTES = {
    "aria-hidden",
    "class",
    "clip-path",
    "clip-rule",
    "color",
    "cx",
    "cy",
    "d",
    "fill",
    "fill-opacity",
    "fill-rule",
    "filter",
    "focusable",
    "fx",
    "fy",
    "gradienttransform",
    "gradientunits",
    "height",
    "href",
    "id",
    "mask",
    "maskcontentunits",
    "maskunits",
    "offset",
    "opacity",
    "orient",
    "pathlength",
    "points",
    "preserveaspectratio",
    "r",
    "refx",
    "refy",
    "role",
    "rx",
    "ry",
    "spreadmethod",
    "stop-color",
    "stop-opacity",
    "stroke",
    "stroke-dasharray",
    "stroke-dashoffset",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-miterlimit",
    "stroke-opacity",
    "stroke-width",
    "transform",
    "vector-effect",
    "viewbox",
    "width",
    "x",
    "x1",
    "x2",
    "y",
    "y1",
    "y2",
}


def validate_network_url(url: str, *, policy: str = "explicit") -> str:
    """Validate one browser network destination for the selected trust policy."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("capture URL must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("capture URL must not contain credentials")
    if policy not in {"explicit", "public", "candidate"}:
        raise ValueError("network policy must be explicit, public, or candidate")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("capture URL must include a hostname")
    if policy == "explicit":
        return hostname

    try:
        addresses = {ipaddress.ip_address(hostname)}
    except ValueError:
        try:
            addresses = {
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(
                    hostname,
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            }
        except (socket.gaierror, ValueError) as error:
            raise ValueError(
                f"capture URL hostname could not be resolved: {hostname}"
            ) from error
    if not addresses:
        raise ValueError(f"capture URL hostname could not be resolved: {hostname}")

    def is_non_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        return (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        )

    if policy == "public" and any(is_non_public(address) for address in addresses):
        raise ValueError("public capture URL resolved to a private or local address")
    if policy == "candidate" and any(
        is_non_public(address) and not address.is_loopback for address in addresses
    ):
        raise ValueError("candidate capture URL resolved to a private network address")
    return hostname


def _guard_browser_request(
    route: Any,
    request: Any,
    *,
    policy: str,
    blocked: list[dict[str, str]],
) -> None:
    """Apply the selected network policy to every browser request."""
    request_url = str(request.url)
    if request_url.startswith(("data:", "blob:", "about:")):
        route.continue_()
        return
    try:
        validate_network_url(request_url, policy=policy)
    except ValueError as error:
        blocked.append({"url": request_url, "reason": str(error)})
        route.abort()
        return
    route.continue_()


def _prepare_network_capture(
    context: Any,
    url: str,
    settings: dict[str, Any],
) -> list[dict[str, str]]:
    """Validate the entry URL and install redirect/subresource protection."""
    policy = str(settings.get("networkPolicy") or "explicit")
    validate_network_url(url, policy=policy)
    blocked: list[dict[str, str]] = []
    if policy != "explicit":
        context.route(
            "**/*",
            lambda route, request: _guard_browser_request(
                route,
                request,
                policy=policy,
                blocked=blocked,
            ),
        )
    return blocked


def _source_raster_urls(value: Any) -> list[str]:
    candidate = str(value or "").strip()
    if candidate.startswith(("http://", "https://")):
        return [candidate.split("#", 1)[0]]
    urls: list[str] = []
    for match in re.finditer(
        r"url\(\s*(?:['\"](?P<quoted>.*?)['\"]|(?P<plain>[^)]+))\s*\)",
        candidate,
        flags=re.IGNORECASE,
    ):
        url = str(match.group("quoted") or match.group("plain") or "").strip()
        if url.startswith(("http://", "https://")):
            urls.append(url.split("#", 1)[0])
    return list(dict.fromkeys(urls))


def _boxes_overlap(first: Any, second: Any) -> bool:
    if not (
        isinstance(first, (list, tuple))
        and isinstance(second, (list, tuple))
        and len(first) == len(second) == 4
    ):
        return False
    try:
        ax0, ay0, ax1, ay1 = (float(value) for value in first)
        bx0, by0, bx1, by1 = (float(value) for value in second)
    except (TypeError, ValueError):
        return False
    return min(ax1, bx1) > max(ax0, bx0) and min(ay1, by1) > max(ay0, by0)


def _xml_local_name(value: Any) -> str:
    return str(value or "").rsplit("}", 1)[-1]


def _sanitize_source_svg(markup: Any, *, id_prefix: str) -> bytes | None:
    """Return inert standalone SVG bytes from an observed live DOM root."""
    candidate = str(markup or "").strip()
    if not candidate or len(candidate.encode("utf-8")) > SOURCE_VECTOR_MAX_BYTES:
        return None
    try:
        root = ET.fromstring(candidate)
    except ET.ParseError:
        return None
    if _xml_local_name(root.tag).casefold() != "svg":
        return None

    def prune(parent: ET.Element) -> None:
        for child in list(parent):
            if _xml_local_name(child.tag).casefold() not in _SOURCE_VECTOR_ALLOWED_TAGS:
                parent.remove(child)
                continue
            prune(child)

    prune(root)
    identifiers: dict[str, str] = {}
    for element in root.iter():
        source_id = str(element.attrib.get("id") or "").strip()
        if source_id:
            suffix = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]
            identifiers[source_id] = f"{id_prefix}{suffix}"

    internal_url = re.compile(r"url\(\s*['\"]?#([^)'\"\s]+)['\"]?\s*\)", re.I)
    unsafe_value = re.compile(r"(?:javascript:|expression\s*\(|@import|https?://|//)", re.I)
    for element in root.iter():
        sanitized_attributes: dict[str, str] = {}
        for raw_name, raw_value in list(element.attrib.items()):
            name = _xml_local_name(raw_name)
            folded_name = name.casefold()
            value = str(raw_value or "").strip()
            if folded_name.startswith("on") or folded_name not in _SOURCE_VECTOR_ALLOWED_ATTRIBUTES:
                continue
            if unsafe_value.search(value):
                continue
            if folded_name == "id":
                replacement = identifiers.get(value)
                if replacement:
                    sanitized_attributes["id"] = replacement
                continue
            if folded_name == "href":
                replacement = identifiers.get(value[1:]) if value.startswith("#") else None
                if replacement:
                    sanitized_attributes["href"] = f"#{replacement}"
                continue
            if "url(" in value.casefold():
                match = internal_url.fullmatch(value)
                replacement = identifiers.get(match.group(1)) if match else None
                if replacement is None:
                    continue
                value = f"url(#{replacement})"
            sanitized_attributes[name] = value
        element.attrib.clear()
        element.attrib.update(sanitized_attributes)
    try:
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        content = ET.tostring(root, encoding="utf-8", short_empty_elements=True)
    except (TypeError, ValueError):
        return None
    if not content or len(content) > SOURCE_VECTOR_MAX_BYTES:
        return None
    return content


def _persist_source_vector_assets(
    vector_elements: list[dict[str, Any]],
    viewport: dict[str, Any],
    work: Path,
    *,
    no_store: bool,
) -> list[dict[str, Any]]:
    """Persist bounded, viewport-visible, sanitized SVG DOM observations."""
    if no_store:
        return []
    try:
        viewport_width = max(1, int(viewport.get("width") or 0))
        viewport_height = max(1, int(viewport.get("height") or 0))
    except (AttributeError, TypeError, ValueError):
        return []
    viewport_area = viewport_width * viewport_height
    persisted: list[dict[str, Any]] = []
    total_bytes = 0
    for vector_index, vector in enumerate(vector_elements[:40]):
        if len(persisted) >= SOURCE_VECTOR_MAX_ASSETS:
            break
        if not isinstance(vector, dict) or vector.get("visible") is not True:
            continue
        box = vector.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x0, y0, x1, y1 = (round(float(value), 3) for value in box)
        except (TypeError, ValueError):
            continue
        width = x1 - x0
        height = y1 - y0
        area = width * height
        if width <= 0 or height <= 0 or area / viewport_area < SOURCE_VECTOR_MIN_VIEWPORT_AREA_RATIO:
            continue
        intersection_width = max(0, min(viewport_width, x1) - max(0, x0))
        intersection_height = max(0, min(viewport_height, y1) - max(0, y0))
        visible_ratio = (intersection_width * intersection_height) / max(1, area)
        if visible_ratio < SOURCE_VECTOR_MIN_VISIBLE_RATIO:
            continue
        content = _sanitize_source_svg(
            vector.get("markup"),
            id_prefix=f"sens-vector-{vector_index}-",
        )
        if content is None or total_bytes + len(content) > SOURCE_VECTOR_TOTAL_MAX_BYTES:
            continue
        digest = hashlib.sha256(content).hexdigest()
        destination = work / f"source-vector-{digest[:20]}.svg"
        if not destination.exists():
            temporary = work / f".{destination.name}.tmp"
            temporary.write_bytes(content)
            temporary.replace(destination)
        persisted.append(
            {
                "vectorIndex": vector_index,
                "domIndex": vector.get("domIndex"),
                "path": str(destination),
                "sha256": digest,
                "sizeBytes": len(content),
                "mediaType": "image/svg+xml",
                "box": [x0, y0, x1, y1],
                "visible": True,
                "viewportCoverage": round(visible_ratio, 5),
                "source": "observed",
                "method": "sanitized-live-dom-svg",
            }
        )
        total_bytes += len(content)
    return persisted


def _persist_source_raster_assets(
    raster_elements: list[dict[str, Any]],
    text_nodes: list[dict[str, Any]],
    responses_by_url: dict[str, Any],
    work: Path,
    *,
    no_store: bool,
) -> list[dict[str, Any]]:
    """Persist bounded image bodies already loaded by the guarded page.

    This function never performs a request. It only reads response objects
    emitted by Playwright for the current capture and referenced by a visible
    DOM raster element.
    """
    if no_store:
        return []
    persisted: list[dict[str, Any]] = []
    body_cache: dict[str, tuple[bytes, str, str] | None] = {}
    total_bytes = 0
    for raster_index, raster in enumerate(raster_elements[:40]):
        if len(persisted) >= SOURCE_RASTER_MAX_ASSETS:
            break
        if not isinstance(raster, dict) or not raster.get("visible"):
            continue
        box = raster.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        for source_url in _source_raster_urls(raster.get("src")):
            response_key = source_url.split("#", 1)[0]
            response = responses_by_url.get(response_key)
            if response is None:
                continue
            if response_key not in body_cache:
                try:
                    headers = getattr(response, "headers", {}) or {}
                    media_type = str(headers.get("content-type") or "")
                    media_type = media_type.split(";", 1)[0].strip().lower()
                    suffix = SOURCE_RASTER_MEDIA_SUFFIXES.get(media_type)
                    if suffix is None:
                        body_cache[response_key] = None
                        continue
                    content = bytes(response.body())
                except Exception:  # noqa: BLE001 - one unavailable body is non-fatal
                    body_cache[response_key] = None
                    continue
                if not content or len(content) > SOURCE_RASTER_MAX_BYTES:
                    body_cache[response_key] = None
                    continue
                body_cache[response_key] = (content, media_type, suffix)
            cached = body_cache[response_key]
            if cached is None:
                continue
            content, media_type, suffix = cached
            if total_bytes + len(content) > SOURCE_RASTER_TOTAL_MAX_BYTES:
                continue
            digest = hashlib.sha256(content).hexdigest()
            destination = work / f"source-raster-{digest[:20]}{suffix}"
            if not destination.exists():
                temporary = work / f".{destination.name}.tmp"
                temporary.write_bytes(content)
                temporary.replace(destination)
            overlapping_text = sum(
                1
                for text_node in text_nodes
                if isinstance(text_node, dict)
                and str(text_node.get("text") or "").strip()
                and _boxes_overlap(box, text_node.get("box"))
            )
            persisted.append(
                {
                    "rasterIndex": raster_index,
                    "domIndex": raster.get("domIndex"),
                    "kind": raster.get("kind"),
                    "path": str(destination),
                    "sha256": digest,
                    "sizeBytes": len(content),
                    "mediaType": media_type,
                    "box": box,
                    "visible": True,
                    "objectFit": raster.get("objectFit"),
                    "backgroundSize": raster.get("backgroundSize"),
                    "backdropColor": raster.get("backdropColor"),
                    "overlappingLiveTextCount": overlapping_text,
                    "source": "observed",
                    "method": "playwright-response-body",
                }
            )
            total_bytes += len(content)
            break
    return persisted


def _persist_source_font_assets(
    font_faces: Any,
    responses_by_url: dict[str, Any],
    work: Path,
    *,
    no_store: bool,
) -> list[dict[str, Any]]:
    """Persist bounded public font responses already loaded by the guarded page."""
    if no_store or not isinstance(font_faces, dict):
        return []
    faces = font_faces.get("faces")
    if not isinstance(faces, list):
        return []
    persisted: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    body_cache: dict[str, tuple[bytes, str, str, str] | None] = {}
    total_bytes = 0
    for face in faces[:64]:
        if len(persisted) >= SOURCE_FONT_MAX_ASSETS or not isinstance(face, dict):
            break
        family = str(face.get("family") or "").strip().strip("'\"")[:128]
        if not family or any(ord(character) < 32 for character in family):
            continue
        weight = str(face.get("weight") or "normal").strip()[:32]
        style = str(face.get("style") or "normal").strip()[:32]
        stretch = str(face.get("stretch") or "normal").strip()[:32]
        sources = face.get("sources")
        if not isinstance(sources, list):
            continue
        for source_url in sources[:8]:
            response_key = str(source_url or "").split("#", 1)[0]
            response = responses_by_url.get(response_key)
            if response is None:
                continue
            if response_key not in body_cache:
                try:
                    headers = getattr(response, "headers", {}) or {}
                    media_type = str(headers.get("content-type") or "")
                    media_type = media_type.split(";", 1)[0].strip().lower()
                    media = SOURCE_FONT_MEDIA.get(media_type)
                    resource_type = str(
                        getattr(getattr(response, "request", None), "resource_type", "")
                        or ""
                    ).casefold()
                    if media is None and resource_type == "font":
                        suffix = Path(urlparse(response_key).path).suffix.casefold()
                        media = next(
                            (
                                value
                                for value in SOURCE_FONT_MEDIA.values()
                                if value[0] == suffix
                            ),
                            None,
                        )
                        if media is not None:
                            media_type = SOURCE_FONT_CANONICAL_MEDIA[media[1]]
                    if media is None:
                        body_cache[response_key] = None
                        continue
                    suffix, font_format = media
                    content = bytes(response.body())
                except Exception:  # noqa: BLE001 - one unavailable body is non-fatal
                    body_cache[response_key] = None
                    continue
                if not content or len(content) > SOURCE_FONT_MAX_BYTES:
                    body_cache[response_key] = None
                    continue
                body_cache[response_key] = (
                    content,
                    media_type or SOURCE_FONT_CANONICAL_MEDIA[font_format],
                    suffix,
                    font_format,
                )
            cached = body_cache[response_key]
            if cached is None:
                continue
            content, media_type, suffix, font_format = cached
            if total_bytes + len(content) > SOURCE_FONT_TOTAL_MAX_BYTES:
                continue
            digest = hashlib.sha256(content).hexdigest()
            identity = (family.casefold(), weight.casefold(), style.casefold(), digest)
            if identity in seen:
                continue
            destination = work / f"source-font-{digest[:20]}{suffix}"
            if not destination.exists():
                temporary = work / f".{destination.name}.tmp"
                temporary.write_bytes(content)
                temporary.replace(destination)
            persisted.append(
                {
                    "family": family,
                    "weight": weight,
                    "style": style,
                    "stretch": stretch,
                    "path": str(destination),
                    "sha256": digest,
                    "sizeBytes": len(content),
                    "mediaType": media_type,
                    "format": font_format,
                    "source": "observed",
                    "method": "playwright-loaded-font-response",
                }
            )
            seen.add(identity)
            total_bytes += len(content)
    return persisted


STYLES_JS = """() => {
  const cs = getComputedStyle(document.body);
  const fonts = new Set();
  document.querySelectorAll("h1,h2,h3,p,a,button").forEach(el =>
    fonts.add(getComputedStyle(el).fontFamily));
  return { bodyBackground: cs.backgroundColor, fonts: [...fonts].slice(0, 12) };
}"""

FONT_FACES_JS = r"""() => {
  const cleanFamily = value => String(value || '').trim().replace(/^['"]|['"]$/g, '');
  const loaded = [...(document.fonts || [])]
    .filter(face => face.status === 'loaded')
    .map(face => ({family: cleanFamily(face.family), style: face.style || 'normal',
      weight: face.weight || 'normal', stretch: face.stretch || 'normal', status: face.status}));
  const loadedFamilies = new Set(loaded.map(face => face.family.toLowerCase()));
  const faces = [];
  const sourceUrls = (value, base) => {
    const out = [];
    const pattern = /url\(\s*(?:['"]([^'"]+)['"]|([^)'"\s]+))\s*\)/gi;
    for (const match of String(value || '').matchAll(pattern)) {
      try { out.push(new URL(match[1] || match[2], base || document.baseURI).href); }
      catch (error) {}
    }
    return [...new Set(out)];
  };
  const visit = (rules, base) => {
    for (const rule of rules || []) {
      if (rule.type === CSSRule.FONT_FACE_RULE) {
        const family = cleanFamily(rule.style.getPropertyValue('font-family'));
        const sources = sourceUrls(rule.style.getPropertyValue('src'), base);
        if (family && sources.length && loadedFamilies.has(family.toLowerCase())) {
          faces.push({family, sources,
            style: rule.style.getPropertyValue('font-style') || 'normal',
            weight: rule.style.getPropertyValue('font-weight') || 'normal',
            stretch: rule.style.getPropertyValue('font-stretch') || 'normal',
            status: 'loaded'});
        }
      } else if (rule.cssRules) {
        visit(rule.cssRules, base);
      }
    }
  };
  for (const sheet of document.styleSheets) {
    try { visit(sheet.cssRules, sheet.href || document.baseURI); } catch (error) {}
  }
  return {loaded, faces};
}"""

DOM_JS = r"""() => [...document.querySelectorAll('body *')].slice(0, 300).map((el, index) => {
  const box = el.getBoundingClientRect();
  const style = getComputedStyle(el);
  return {
    index, tag: el.tagName.toLowerCase(), id: el.id || null,
    classes: [...el.classList].slice(0, 8), role: el.getAttribute('role'),
    ariaLabel: el.getAttribute('aria-label'),
    text: (el.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 300),
    directText: [...el.childNodes].filter(node => node.nodeType === Node.TEXT_NODE)
      .map(node => node.textContent || '').join(' ').trim().replace(/\s+/g, ' ').slice(0, 300),
    box: [Math.round(box.x), Math.round(box.y), Math.round(box.right), Math.round(box.bottom)],
    style: { display: style.display, position: style.position, color: style.color,
      background: style.backgroundColor, font: style.font, fontFamily: style.fontFamily,
      fontSize: style.fontSize, fontWeight: style.fontWeight, lineHeight: style.lineHeight,
      letterSpacing: style.letterSpacing, textTransform: style.textTransform,
      textAlign: style.textAlign, border: style.border,
      borderTop: style.borderTop, borderRight: style.borderRight,
      borderBottom: style.borderBottom, borderLeft: style.borderLeft,
      borderRadius: style.borderRadius, gap: style.gap, padding: style.padding,
      visibility: style.visibility, opacity: style.opacity, cursor: style.cursor,
      pointerEvents: style.pointerEvents,
      userSelect: style.userSelect || style.webkitUserSelect || 'auto' }
  };
}).filter(item => item.box[2] > item.box[0] && item.box[3] > item.box[1])"""

TEXT_NODES_JS = r"""() => {
  const elements = [...document.querySelectorAll('body *')];
  const indexes = new Map(elements.map((element, index) => [element, index]));
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent || ['script', 'style', 'noscript', 'template'].includes(parent.tagName.toLowerCase()))
        return NodeFilter.FILTER_REJECT;
      return (node.textContent || '').trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    }
  });
  const out = [];
  const seenSemanticParents = new Set();
  while (walker.nextNode() && out.length < 500) {
    const node = walker.currentNode;
    const visualParent = node.parentElement;
    const semanticParent = visualParent.closest('[data-sens-text-box="true"]');
    if (semanticParent && seenSemanticParents.has(semanticParent)) continue;
    if (semanticParent) seenSemanticParents.add(semanticParent);
    const parent = semanticParent || visualParent;
    const range = document.createRange();
    range.selectNodeContents(node);
    const rect = semanticParent
      ? semanticParent.getBoundingClientRect()
      : range.getBoundingClientRect();
    const style = getComputedStyle(parent);
    const rawText = (
      semanticParent ? semanticParent.textContent : node.textContent || ''
    ).replace(/\r\n?/g, '\n');
    const wordBoxes = [];
    if (!semanticParent) {
      for (const match of (node.textContent || '').matchAll(/\S+/gu)) {
        const start = match.index;
        if (!Number.isInteger(start)) continue;
        const wordRange = document.createRange();
        wordRange.setStart(node, start);
        wordRange.setEnd(node, start + match[0].length);
        const wordRect = wordRange.getBoundingClientRect();
        if (wordRect.width <= 0 || wordRect.height <= 0) continue;
        wordBoxes.push({
          text: match[0],
          box: [Math.round(wordRect.x), Math.round(wordRect.y),
            Math.round(wordRect.right), Math.round(wordRect.bottom)]
        });
      }
    }
    const visible = rect.width > 0 && rect.height > 0 && style.display !== 'none' &&
      style.visibility !== 'hidden' && Number(style.opacity || 1) > 0;
    out.push({
      text: rawText.trim().replace(/\s+/g, ' ').slice(0, 500),
      rawText: rawText.slice(0, 20000), whiteSpace: style.whiteSpace,
      wordBoxes,
      box: [Math.round(rect.x), Math.round(rect.y), Math.round(rect.right), Math.round(rect.bottom)],
      parentIndex: indexes.get(parent) ?? null,
      parentTag: parent.tagName.toLowerCase(), id: parent.id || null,
      classes: [...parent.classList].slice(0, 8),
      style: { color: style.color, fontFamily: style.fontFamily,
        fontSize: style.fontSize, fontWeight: style.fontWeight,
        fontStyle: style.fontStyle, lineHeight: style.lineHeight,
        letterSpacing: style.letterSpacing, textTransform: style.textTransform,
        textAlign: style.textAlign },
      userSelect: style.userSelect || style.webkitUserSelect || 'auto',
      pointerEvents: style.pointerEvents, visibility: style.visibility,
      opacity: style.opacity, visible
    });
  }
  return out;
}"""

SEMANTIC_CONTROLS_JS = r"""() => {
  const elements = [...document.querySelectorAll('body *')];
  const indexes = new Map(elements.map((element, index) => [element, index]));
  const selector = [
    'a[href]', 'button', 'input', 'select', 'textarea', 'summary',
    '[role="button"]', '[role="link"]', '[role="checkbox"]', '[role="radio"]',
    '[role="switch"]', '[role="tab"]', '[role="menuitem"]'
  ].join(',');
  return [...document.querySelectorAll(selector)].slice(0, 300).map((element, index) => {
    const rect = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    const tag = element.tagName.toLowerCase();
    const inferredRole = tag === 'a' ? 'link' : ['button', 'summary'].includes(tag) ? 'button' : tag;
    return {
      index, domIndex: indexes.get(element) ?? null, tag,
      role: element.getAttribute('role') || inferredRole,
      name: (element.getAttribute('aria-label') || element.innerText || element.value ||
        element.getAttribute('title') || '').trim().replace(/\s+/g, ' ').slice(0, 300),
      box: [Math.round(rect.x), Math.round(rect.y), Math.round(rect.right), Math.round(rect.bottom)],
      href: element.href || null, disabled: Boolean(element.disabled || element.getAttribute('aria-disabled') === 'true'),
      tabIndex: element.tabIndex, cursor: style.cursor, pointerEvents: style.pointerEvents,
      style: { color: style.color, background: style.backgroundColor,
        fontFamily: style.fontFamily, fontSize: style.fontSize,
        fontWeight: style.fontWeight, lineHeight: style.lineHeight,
        letterSpacing: style.letterSpacing, border: style.border,
        borderRadius: style.borderRadius, padding: style.padding },
      userSelect: style.userSelect || style.webkitUserSelect || 'auto',
      visible: rect.width > 0 && rect.height > 0 && style.display !== 'none' &&
        style.visibility !== 'hidden' && Number(style.opacity || 1) > 0
    };
  });
}"""

STRUCTURAL_LINES_JS = r"""() => {
  const elements = [...document.querySelectorAll('body *')];
  const out = [];
  const seen = new Set();
  const visibleColor = value => value && value !== 'transparent' &&
    !/^rgba\([^)]*,\s*0(?:\.0+)?\)$/.test(value);
  const push = (orientation, box, thickness, color, source, domIndex) => {
    const rounded = box.map(value => Math.round(value));
    const key = `${orientation}:${rounded.join(',')}:${color}`;
    if (seen.has(key) || rounded[2] <= rounded[0] || rounded[3] <= rounded[1]) return;
    seen.add(key);
    out.push({orientation, box: rounded, thickness: Math.round(thickness * 100) / 100,
      color, source, domIndex});
  };
  for (let domIndex = 0; domIndex < elements.length && out.length < 300; domIndex += 1) {
    const element = elements[domIndex];
    const rect = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    if (rect.width <= 0 || rect.height <= 0 || style.display === 'none' ||
        style.visibility === 'hidden' || Number(style.opacity || 1) <= 0) continue;
    if (element.dataset.sensLine === 'true') {
      const orientation = rect.width >= rect.height ? 'horizontal' : 'vertical';
      const thickness = orientation === 'horizontal' ? rect.height : rect.width;
      push(orientation, [rect.x, rect.y, rect.right, rect.bottom], thickness,
        element.dataset.sensLineColor || style.backgroundColor,
        `sens-${element.dataset.sensLineStyle || 'solid'}-line`, domIndex);
      continue;
    }
    const sides = [
      ['horizontal', style.borderTopWidth, style.borderTopColor,
        [rect.x, rect.y, rect.right, rect.y + parseFloat(style.borderTopWidth || 0)], 'border-top'],
      ['horizontal', style.borderBottomWidth, style.borderBottomColor,
        [rect.x, rect.bottom - parseFloat(style.borderBottomWidth || 0), rect.right, rect.bottom], 'border-bottom'],
      ['vertical', style.borderLeftWidth, style.borderLeftColor,
        [rect.x, rect.y, rect.x + parseFloat(style.borderLeftWidth || 0), rect.bottom], 'border-left'],
      ['vertical', style.borderRightWidth, style.borderRightColor,
        [rect.right - parseFloat(style.borderRightWidth || 0), rect.y, rect.right, rect.bottom], 'border-right']
    ];
    for (const [orientation, widthValue, color, box, source] of sides) {
      const width = parseFloat(widthValue || 0);
      if (width > 0 && visibleColor(color)) push(orientation, box, width, color, source, domIndex);
    }
    if (rect.height <= 8 && rect.width >= 20 && visibleColor(style.backgroundColor)) {
      push('horizontal', [rect.x, rect.y, rect.right, rect.bottom], rect.height,
        style.backgroundColor, 'thin-fill', domIndex);
    } else if (rect.width <= 8 && rect.height >= 20 && visibleColor(style.backgroundColor)) {
      push('vertical', [rect.x, rect.y, rect.right, rect.bottom], rect.width,
        style.backgroundColor, 'thin-fill', domIndex);
    }
  }
  return out;
}"""

RASTER_ELEMENTS_JS = r"""() => {
  const elements = [...document.querySelectorAll('body *')];
  const out = [];
  const nearestBackdropColor = (element) => {
    for (let current = element; current; current = current.parentElement) {
      const color = getComputedStyle(current).backgroundColor;
      if (!color || color.toLowerCase() === 'transparent') continue;
      const rgba = color.match(/^rgba\([^,]+,[^,]+,[^,]+,\s*([0-9.]+)\s*\)$/i);
      if (!rgba || Number(rgba[1]) > 0.01) return color;
    }
    return null;
  };
  for (let domIndex = 0; domIndex < elements.length && out.length < 100; domIndex += 1) {
    const element = elements[domIndex];
    const tag = element.tagName.toLowerCase();
    const style = getComputedStyle(element);
    let kind = null; let src = null;
    if (tag === 'img') { kind = 'img'; src = element.currentSrc || element.src; }
    else if (tag === 'canvas') kind = 'canvas';
    else if (tag === 'video') { kind = 'video'; src = element.poster || element.currentSrc || element.src; }
    else if (tag === 'picture') kind = 'picture';
    else if (tag === 'object' || tag === 'embed') { kind = tag; src = element.data || element.src || null; }
    else if (tag === 'svg') kind = 'vector-svg';
    else if (style.backgroundImage && style.backgroundImage !== 'none') {
      src = style.backgroundImage;
      kind = /(?:url|image-set)\(/i.test(src) ? 'background-image' : 'css-gradient';
    } else {
      const mask = style.maskImage || style.webkitMaskImage;
      if (mask && mask !== 'none') { kind = 'css-mask'; src = mask; }
    }
    if (!kind) continue;
    const rect = element.getBoundingClientRect();
    out.push({
      domIndex, tag, kind, src, alt: element.alt || null,
      sensRasterRole: element.dataset.sensRasterRole || null,
      sensArtifactId: element.dataset.sensArtifactId || null,
      text: (element.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 300),
      box: [Math.round(rect.x), Math.round(rect.y), Math.round(rect.right), Math.round(rect.bottom)],
      visible: rect.width > 0 && rect.height > 0 && style.display !== 'none' &&
        style.visibility !== 'hidden' && Number(style.opacity || 1) > 0,
      objectFit: style.objectFit, backgroundSize: style.backgroundSize,
      backdropColor: nearestBackdropColor(element),
      pointerEvents: style.pointerEvents
    });
  }
  return out;
}"""

VECTOR_ELEMENTS_JS = r"""() => {
  const elements = [...document.querySelectorAll('body *')];
  const indexes = new Map(elements.map((element, index) => [element, index]));
  const out = [];
  const subpixel = value => Math.round(value * 1000) / 1000;
  const inlineProperties = [
    ['color', 'color'], ['fill', 'fill'], ['fill-opacity', 'fillOpacity'],
    ['fill-rule', 'fillRule'], ['stroke', 'stroke'],
    ['stroke-opacity', 'strokeOpacity'], ['stroke-width', 'strokeWidth'],
    ['stroke-linecap', 'strokeLinecap'], ['stroke-linejoin', 'strokeLinejoin'],
    ['stroke-miterlimit', 'strokeMiterlimit'], ['opacity', 'opacity'],
    ['clip-rule', 'clipRule']
  ];
  for (const element of document.querySelectorAll('svg')) {
    if (out.length >= 40 || element.ownerSVGElement) continue;
    const rect = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    const visible = rect.width > 0 && rect.height > 0 && style.display !== 'none' &&
      style.visibility !== 'hidden' && Number(style.opacity || 1) > 0;
    if (!visible) continue;
    const intersectionWidth = Math.max(0, Math.min(innerWidth, rect.right) - Math.max(0, rect.left));
    const intersectionHeight = Math.max(0, Math.min(innerHeight, rect.bottom) - Math.max(0, rect.top));
    const viewportCoverage = intersectionWidth * intersectionHeight /
      Math.max(1, rect.width * rect.height);
    const clone = element.cloneNode(true);
    const sourceNodes = [element, ...element.querySelectorAll('*')];
    const cloneNodes = [clone, ...clone.querySelectorAll('*')];
    for (let index = 0; index < Math.min(sourceNodes.length, cloneNodes.length); index += 1) {
      const computed = getComputedStyle(sourceNodes[index]);
      for (const [attribute, property] of inlineProperties) {
        const value = computed[property];
        if (value && value !== 'none' && value !== 'normal')
          cloneNodes[index].setAttribute(attribute, value);
      }
    }
    out.push({
      domIndex: indexes.get(element),
      box: [subpixel(rect.x), subpixel(rect.y), subpixel(rect.right), subpixel(rect.bottom)],
      visible: true,
      viewportCoverage: Math.round(viewportCoverage * 100000) / 100000,
      markup: clone.outerHTML
    });
  }
  return out;
}"""

ASSETS_JS = """() => ({
  images: [...document.images].slice(0, 100).map(img => ({src: img.currentSrc || img.src, alt: img.alt,
    size: [img.naturalWidth, img.naturalHeight]})),
  stylesheets: [...document.styleSheets].slice(0, 100).map(sheet => sheet.href).filter(Boolean),
  links: [...document.querySelectorAll('a[href]')].slice(0, 100).map(a => a.href)
})"""

VARS_JS = """() => {
  const style = getComputedStyle(document.documentElement); const out = {};
  for (const name of style) if (name.startsWith('--')) out[name] = style.getPropertyValue(name).trim();
  return out;
}"""

ANIM_JS = """() => {
  const out = { keyframes: [], animated: [], live: [] };
  for (const sheet of document.styleSheets) {
    let rules; try { rules = sheet.cssRules; } catch (e) { continue; }
    for (const r of rules) {
      if (r.type === CSSRule.KEYFRAMES_RULE)
        out.keyframes.push({ name: r.name, steps: r.cssRules.length });
      if (r.style && (r.style.animationName || r.style.transitionProperty))
        out.animated.push({
          selector: r.selectorText,
          animation: r.style.animationName,
          duration: r.style.animationDuration || r.style.transitionDuration,
          easing: r.style.animationTimingFunction || r.style.transitionTimingFunction
        });
    }
  }
  out.live = (document.getAnimations ? document.getAnimations() : []).slice(0, 50)
    .map(a => ({ name: a.animationName || "", state: a.playState,
      duration: a.effect && a.effect.getTiming ? a.effect.getTiming().duration : null }));
  return out;
}"""

FREEZE_VISUAL_STATE_JS = """() => {
  const animations = document.getAnimations ? document.getAnimations() : [];
  let paused = 0;
  for (const animation of animations) {
    try { animation.pause(); paused += 1; } catch (error) {}
  }
  let style = document.getElementById('sens-capture-freeze-style');
  if (!style) {
    style = document.createElement('style');
    style.id = 'sens-capture-freeze-style';
    (document.head || document.documentElement).appendChild(style);
  }
  style.textContent = `
    *, *::before, *::after {
      animation-play-state:paused !important;
      transition-property:none !important;
      caret-color:transparent !important;
      scroll-behavior:auto !important;
    }
  `;
  document.documentElement.setAttribute('data-sens-capture-frozen', 'true');
  return new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve({
    animationsObserved: animations.length,
    animationsPaused: paused,
    source: 'observed',
    method: 'paused-web-animations-plus-capture-style'
  }))));
}"""


def normalize_capture_options(options: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = options or {}
    viewport = raw.get("viewport") or {}
    if not isinstance(viewport, dict):
        raise ValueError("viewport must be an object with width and height")
    width = int(viewport.get("width", 1440))
    height = int(viewport.get("height", 900))
    if not 320 <= width <= 3840:
        raise ValueError("viewport width must be between 320 and 3840")
    if not 240 <= height <= 2160:
        raise ValueError("viewport height must be between 240 and 2160")
    dpr = float(raw.get("dpr") if raw.get("dpr") is not None else 1.0)
    if not 0.5 <= dpr <= 3.0:
        raise ValueError("dpr must be between 0.5 and 3.0")
    theme = str(raw.get("theme") or "light")
    if theme not in {"light", "dark", "no-preference"}:
        raise ValueError("theme must be light, dark, or no-preference")
    wait_until = str(raw.get("waitUntil") or "networkidle")
    if wait_until not in {"commit", "domcontentloaded", "load", "networkidle"}:
        raise ValueError("waitUntil must be commit, domcontentloaded, load, or networkidle")
    timeout_ms = int(
        raw.get("timeoutMs") if raw.get("timeoutMs") is not None else 30_000
    )
    if not 1_000 <= timeout_ms <= 60_000:
        raise ValueError("timeoutMs must be between 1000 and 60000")
    settle_ms = int(raw.get("settleMs") if raw.get("settleMs") is not None else 250)
    if not 0 <= settle_ms <= 5_000:
        raise ValueError("settleMs must be between 0 and 5000")
    scroll_steps = int(
        raw.get("scrollSteps") if raw.get("scrollSteps") is not None else 0
    )
    if not 0 <= scroll_steps <= 10:
        raise ValueError("scrollSteps must be between 0 and 10")
    network_policy = str(raw.get("networkPolicy") or "explicit")
    if network_policy not in {"explicit", "public", "candidate"}:
        raise ValueError("networkPolicy must be explicit, public, or candidate")
    return {
        "viewport": {"width": width, "height": height},
        "dpr": dpr,
        "theme": theme,
        "locale": str(raw.get("locale") or "en-US"),
        "waitUntil": wait_until,
        "fullPage": bool(raw.get("fullPage", False)),
        "timeoutMs": timeout_ms,
        "settleMs": settle_ms,
        "scrollSteps": scroll_steps,
        "networkPolicy": network_policy,
    }


def capture_request_id(url: str, options: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(url.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(options, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()[:20]


def _changed_boxes(prev: Any, cur: Any, min_area: int = 400) -> list[tuple[int, int, int, int]]:
    diff = cv2.absdiff(prev, cur)
    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]


def motion_events(frame_paths: list[str], step_seconds: float = 0.7) -> list[dict[str, Any]]:
    """Frame-diff motion events.

    A moving object shows up in a consecutive-frame diff as TWO changed
    regions: where it left and where it arrived. We greedily pair changed
    boxes within each transition (nearest centers) and report the pair's
    center-to-center delta as dx/dy — an approximation of the shift. Boxes
    that pair with nothing are emitted with dx/dy None.
    """
    events: list[dict[str, Any]] = []
    prev_gray = None
    for i, fp in enumerate(frame_paths):
        gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if prev_gray is not None:
            boxes = _changed_boxes(prev_gray, gray)
            centers = [(x + w / 2, y + h / 2) for (x, y, w, h) in boxes]
            order = sorted(range(len(boxes)), key=lambda k: (boxes[k][0], boxes[k][1]))
            used: set[int] = set()
            for a in order:
                if a in used:
                    continue
                best, best_dist = None, None
                for b in order:
                    if b == a or b in used:
                        continue
                    dist = (centers[a][0] - centers[b][0]) ** 2 + (centers[a][1] - centers[b][1]) ** 2
                    if best_dist is None or dist < best_dist:
                        best, best_dist = b, dist
                if best is None:
                    x, y, w, h = boxes[a]
                    events.append(
                        {"frame": i, "box": [x, y, x + w, y + h], "dx": None, "dy": None,
                         "seconds": round(i * step_seconds, 1)}
                    )
                    used.add(a)
                    continue
                used.add(a)
                used.add(best)
                x1, y1, w1, h1 = boxes[a]
                x2, y2, w2, h2 = boxes[best]
                events.append(
                    {
                        "frame": i,
                        "box": [
                            min(x1, x2), min(y1, y2),
                            max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2),
                        ],
                        "dx": round(centers[best][0] - centers[a][0]),
                        "dy": round(centers[best][1] - centers[a][1]),
                        "seconds": round(i * step_seconds, 1),
                    }
                )
        prev_gray = gray
    return events


def _content_address(path: Path, prefix: str) -> tuple[Path, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    target = path.with_name(f"{prefix}-{digest[:20]}{path.suffix}")
    path.replace(target)
    return target, digest


def _launch_browser(chromium: Any) -> Any:
    """Prefer Windows' system Edge so the Sens runtime ships no browser blob."""
    try:
        return chromium.launch(channel="msedge", headless=True)
    except Exception:  # noqa: BLE001 - Playwright has backend-specific errors
        try:
            return chromium.launch(headless=True)
        except Exception as bundled_error:  # noqa: BLE001
            raise RuntimeError(
                "URL capture needs Microsoft Edge or a Playwright Chromium installation"
            ) from bundled_error


def _best_effort_close(resource: Any | None) -> None:
    """Close a Playwright resource without masking the capture failure.

    Playwright stops its private event loop when ``sync_playwright`` exits. If
    an exception unwinds the context manager before the browser/context were
    cleared below, a second ``close`` raises ``Event loop is closed`` and hides
    the actionable navigation or capture error that caused the unwind.
    """
    if resource is None:
        return
    try:
        resource.close()
    except Exception:  # noqa: BLE001 - cleanup must preserve the primary failure
        pass


def _freeze_visual_state(page: Any) -> dict[str, Any]:
    """Pause visual motion before binding a screenshot to DOM measurements."""
    evidence = page.evaluate(FREEZE_VISUAL_STATE_JS)
    return evidence if isinstance(evidence, dict) else {
        "animationsObserved": 0,
        "animationsPaused": 0,
        "source": "observed",
        "method": "paused-web-animations-plus-capture-style",
    }


def _navigate_page(
    page: Any,
    url: str,
    settings: dict[str, Any],
    *,
    timeout_error_type: type[BaseException],
) -> dict[str, Any]:
    """Attempt network-idle without discarding an already usable document."""
    requested_wait = settings["waitUntil"]
    timeout_ms = settings["timeoutMs"]
    if requested_wait != "networkidle":
        page.goto(url, wait_until=requested_wait, timeout=timeout_ms)
        return {
            "requestedWaitUntil": requested_wait,
            "navigationWaitUntil": requested_wait,
            "observedWaitState": requested_wait,
            "fallbackUsed": False,
        }

    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    try:
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except timeout_error_type:
        ready = bool(
            page.evaluate(
                "() => document.readyState !== 'loading' && "
                "Boolean(document.body && document.body.childElementCount > 0)"
            )
        )
        if not ready:
            raise
        return {
            "requestedWaitUntil": requested_wait,
            "navigationWaitUntil": "domcontentloaded",
            "observedWaitState": "dom-ready-after-networkidle-timeout",
            "fallbackUsed": True,
        }
    return {
        "requestedWaitUntil": requested_wait,
        "navigationWaitUntil": "domcontentloaded",
        "observedWaitState": "networkidle",
        "fallbackUsed": False,
    }


def capture_url(
    url: str,
    out_dir: str | Path,
    options: dict[str, Any] | None = None,
    *,
    no_store: bool = False,
) -> dict[str, Any]:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("capture URL must use http or https")
    settings = normalize_capture_options(options)
    validate_network_url(url, policy=settings["networkPolicy"])
    request_id = capture_request_id(url, settings)
    root = Path(out_dir)
    if no_store:
        work = Path(tempfile.mkdtemp(prefix="sens-capture-"))
    else:
        work = root / f"capture-{request_id}-{time.time_ns()}"
        work.mkdir(parents=True, exist_ok=False)

    browser = None
    context = None
    blocked_requests: list[dict[str, str]] = []
    final_url = url
    navigation: dict[str, Any] | None = None
    try:
        with sync_playwright() as playwright:
            browser = _launch_browser(playwright.chromium)
            context = browser.new_context(
                viewport=settings["viewport"],
                device_scale_factor=settings["dpr"],
                color_scheme=settings["theme"],
                locale=settings["locale"],
            )
            blocked_requests = _prepare_network_capture(context, url, settings)
            page = context.new_page()
            loaded_responses: dict[str, Any] = {}
            page.on(
                "response",
                lambda response: loaded_responses.__setitem__(
                    str(response.url).split("#", 1)[0], response
                ),
            )
            try:
                navigation = _navigate_page(
                    page,
                    url,
                    settings,
                    timeout_error_type=PlaywrightTimeoutError,
                )
            except Exception as error:  # noqa: BLE001 - Playwright backend errors vary
                if blocked_requests:
                    latest = blocked_requests[-1]
                    raise ValueError(
                        "capture blocked a browser request: "
                        f"{latest['url']} ({latest['reason']})"
                    ) from error
                raise
            final_url = page.url
            page.wait_for_function(
                "() => document.readyState !== 'loading' && document.body && document.body.childElementCount > 0",
                timeout=settings["timeoutMs"],
            )
            page.evaluate("() => document.fonts ? document.fonts.ready : Promise.resolve()")
            if settings["settleMs"]:
                page.wait_for_timeout(settings["settleMs"])

            animations = page.evaluate(ANIM_JS)
            visual_freeze = _freeze_visual_state(page)
            shot = work / "screenshot.png"
            page.screenshot(path=str(shot), full_page=settings["fullPage"])
            shot, screenshot_hash = _content_address(shot, "screenshot")
            styles = page.evaluate(STYLES_JS)
            font_faces = page.evaluate(FONT_FACES_JS)
            dom = page.evaluate(DOM_JS)
            text_nodes = page.evaluate(TEXT_NODES_JS)
            semantic_controls = page.evaluate(SEMANTIC_CONTROLS_JS)
            structural_lines = page.evaluate(STRUCTURAL_LINES_JS)
            raster_elements = page.evaluate(RASTER_ELEMENTS_JS)
            vector_elements = page.evaluate(VECTOR_ELEMENTS_JS)
            assets = page.evaluate(ASSETS_JS)
            css_variables = page.evaluate(VARS_JS)
            source_raster_assets = _persist_source_raster_assets(
                raster_elements,
                text_nodes,
                loaded_responses,
                work,
                no_store=no_store,
            )
            source_vector_assets = _persist_source_vector_assets(
                vector_elements,
                settings["viewport"],
                work,
                no_store=no_store,
            )
            source_font_assets = _persist_source_font_assets(
                font_faces,
                loaded_responses,
                work,
                no_store=no_store,
            )
            try:
                accessibility = page.locator("body").aria_snapshot(timeout=5_000)
            except Exception:  # noqa: BLE001 - browser versions vary
                accessibility = None

            element_paths = []
            regions = page.locator("header, nav, main, section, article, aside, footer")
            for index in range(min(regions.count(), 10)):
                locator = regions.nth(index)
                try:
                    if not locator.is_visible():
                        continue
                    path = work / f"element-{index}.png"
                    locator.screenshot(path=str(path), timeout=5_000)
                    path, digest = _content_address(path, f"element-{index}")
                    element_paths.append(
                        {"path": str(path), "sha256": digest, "index": index}
                    )
                except Exception:  # noqa: BLE001 - one unstable element is non-fatal
                    continue

            raster_paths = []
            all_elements = page.locator("body *")
            for raster_index, raster in enumerate(raster_elements[:40]):
                if not raster.get("visible"):
                    continue
                dom_index = raster.get("domIndex")
                if not isinstance(dom_index, int) or dom_index >= all_elements.count():
                    continue
                try:
                    locator = all_elements.nth(dom_index)
                    path = work / f"raster-{raster_index}.png"
                    locator.screenshot(path=str(path), timeout=5_000)
                    path, digest = _content_address(path, f"raster-{raster_index}")
                    raster_paths.append(
                        {
                            "path": str(path),
                            "sha256": digest,
                            "rasterIndex": raster_index,
                            "domIndex": dom_index,
                        }
                    )
                except Exception:  # noqa: BLE001 - unstable assets stay observable in DOM
                    continue

            frames = [shot]
            for index in range(settings["scrollSteps"]):
                page.evaluate(
                    "amount => window.scrollBy({top: amount, behavior: 'instant'})",
                    round(settings["viewport"]["height"] * 0.75),
                )
                page.wait_for_timeout(max(100, settings["settleMs"]))
                frame = work / f"frame-{index}.png"
                page.screenshot(path=str(frame))
                frame, _ = _content_address(frame, f"frame-{index}")
                frames.append(frame)

            context.close()
            context = None
            browser.close()
            browser = None

        frame_strings = [str(path) for path in frames]
        result = {
            "schemaVersion": "2.0.0",
            "captureId": request_id,
            "source": {
                "url": url,
                "finalUrl": final_url,
                "source": "observed",
                "method": "playwright-instrumented-capture",
            },
            "settings": settings,
            "visualFreeze": visual_freeze,
            "navigation": navigation,
            "blockedRequests": blocked_requests,
            "screenshot": None if no_store else str(shot),
            "screenshotSha256": screenshot_hash,
            "styles": styles,
            "fontFaces": font_faces,
            "cssVariables": css_variables,
            "dom": dom,
            "textNodes": text_nodes,
            "semanticControls": semantic_controls,
            "structuralLines": structural_lines,
            "rasterElements": raster_elements,
            "sourceRasterAssets": source_raster_assets,
            "sourceVectorAssets": source_vector_assets,
            "sourceFontAssets": source_font_assets,
            "accessibility": accessibility,
            "assets": assets,
            "animations": animations,
            "elementScreenshots": [] if no_store else element_paths,
            "rasterElementScreenshots": [] if no_store else raster_paths,
            "frames": [] if no_store else frame_strings,
            "motion": motion_events(frame_strings),
            "artifacts": (
                []
                if no_store
                else [
                    {
                        "id": f"sha256:{screenshot_hash}",
                        "kind": "web-screenshot",
                        "uri": str(shot),
                    },
                    *[
                        {
                            "id": f"sha256:{asset['sha256']}",
                            "kind": "web-source-raster",
                            "uri": asset["path"],
                            "mediaType": asset["mediaType"],
                        }
                        for asset in source_raster_assets
                    ],
                    *[
                        {
                            "id": f"sha256:{asset['sha256']}",
                            "kind": "web-source-vector",
                            "uri": asset["path"],
                            "mediaType": asset["mediaType"],
                        }
                        for asset in source_vector_assets
                    ],
                    *[
                        {
                            "id": f"sha256:{asset['sha256']}",
                            "kind": "web-source-font",
                            "uri": asset["path"],
                            "mediaType": asset["mediaType"],
                        }
                        for asset in source_font_assets
                    ],
                ]
            ),
        }
        if no_store:
            result["screenshotDataUri"] = (
                "data:image/png;base64," + base64.b64encode(shot.read_bytes()).decode("ascii")
            )
        return result
    finally:
        _best_effort_close(context)
        _best_effort_close(browser)
        if no_store:
            shutil.rmtree(work, ignore_errors=True)
