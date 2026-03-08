import json
import os
import base64
from pathlib import Path
from typing import Any

import requests
from flask import jsonify, request, send_from_directory


def _default_data_path() -> str:
    return "/Users/json/Developer/projects/senior-design-data"


def _resolve_data_paths() -> tuple[Path, Path]:
    data_root = Path(os.getenv("DATA_PATH", _default_data_path())).expanduser()
    labels_dir = data_root / "test" / "labels"
    images_dir = data_root / "test" / "images"
    return labels_dir, images_dir


def _parse_scene_from_filename(filename: str) -> tuple[str | None, str | None]:
    if filename.endswith("_pre_disaster.json"):
        return filename.removesuffix("_pre_disaster.json"), "pre"
    if filename.endswith("_post_disaster.json"):
        return filename.removesuffix("_post_disaster.json"), "post"
    return None, None


def _parse_polygon_wkt_bounds(wkt: str) -> list[float] | None:
    # Supports simple POLYGON ((lng lat, ...)) strings used in xView2 labels.
    if not wkt.startswith("POLYGON"):
        return None

    content = wkt.replace("POLYGON", "", 1).strip()
    if not (content.startswith("((") and content.endswith("))")):
        return None

    points = content[2:-2].split(",")
    min_lng = float("inf")
    min_lat = float("inf")
    max_lng = float("-inf")
    max_lat = float("-inf")

    for point in points:
        parts = point.strip().split()
        if len(parts) < 2:
            continue
        lng = float(parts[0])
        lat = float(parts[1])
        min_lng = min(min_lng, lng)
        min_lat = min(min_lat, lat)
        max_lng = max(max_lng, lng)
        max_lat = max(max_lat, lat)

    if min_lng == float("inf"):
        return None
    return [min_lng, min_lat, max_lng, max_lat]


def _merge_bounds(base: list[float] | None, nxt: list[float] | None) -> list[float] | None:
    if base is None:
        return nxt
    if nxt is None:
        return base
    return [
        min(base[0], nxt[0]),
        min(base[1], nxt[1]),
        max(base[2], nxt[2]),
        max(base[3], nxt[3]),
    ]


def _extract_label_bounds(label_data: dict[str, Any]) -> list[float] | None:
    bounds: list[float] | None = None
    lng_lat_features = label_data.get("features", {}).get("lng_lat", [])
    for feature in lng_lat_features:
        feature_bounds = _parse_polygon_wkt_bounds(feature.get("wkt", ""))
        bounds = _merge_bounds(bounds, feature_bounds)
    return bounds


def _load_label(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _encode_image_to_data_url(path: Path) -> str | None:
    if not path.exists():
        return None
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _extract_feature_by_uid(scene: dict[str, Any], feature_id: str) -> dict[str, Any] | None:
    for phase in ("pre", "post"):
        phase_data = scene.get(phase)
        if not phase_data:
            continue
        for feature in phase_data.get("features", {}).get("lng_lat", []):
            uid = feature.get("properties", {}).get("uid")
            if uid == feature_id:
                return feature
    return None


def _openrouter_analyze(
    feature: dict[str, Any] | None,
    pre_data_url: str | None,
    post_data_url: str | None,
    disaster_id: str,
    scene_id: str,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    feature_properties = (feature or {}).get("properties", {})
    feature_wkt = (feature or {}).get("wkt")

    prompt_text = (
        "You are a disaster damage analyst. Compare pre-disaster and post-disaster satellite images "
        "for one structure and provide:\n"
        "1) damage classification\n"
        "2) confidence (0-100)\n"
        "3) brief justification\n"
        "4) immediate response recommendation\n\n"
        f"Disaster: {disaster_id}\n"
        f"Scene: {scene_id}\n"
        f"Feature metadata: {json.dumps(feature_properties)}\n"
        f"Feature geometry (WKT): {feature_wkt}\n"
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    if pre_data_url:
        content.append({"type": "text", "text": "Pre-disaster image:"})
        content.append({"type": "image_url", "image_url": {"url": pre_data_url}})
    if post_data_url:
        content.append({"type": "text", "text": "Post-disaster image:"})
        content.append({"type": "image_url", "image_url": {"url": post_data_url}})

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()

    content_value = body["choices"][0]["message"]["content"]
    if isinstance(content_value, str):
        return content_value
    if isinstance(content_value, list):
        parts: list[str] = []
        for item in content_value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts).strip()
    return str(content_value)


def _build_dataset_index() -> dict[str, Any]:
    labels_dir, _ = _resolve_data_paths()
    if not labels_dir.exists():
        return {"disasters": {}, "errors": [f"Labels directory not found: {labels_dir}"]}

    disasters: dict[str, Any] = {}
    errors: list[str] = []

    for label_path in labels_dir.glob("*.json"):
        scene_id, phase = _parse_scene_from_filename(label_path.name)
        if scene_id is None or phase is None:
            continue

        try:
            label_data = _load_label(label_path)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            errors.append(f"Failed to read {label_path.name}: {exc}")
            continue

        metadata = label_data.get("metadata", {})
        disaster_id = metadata.get("disaster")
        if not disaster_id:
            errors.append(f"Missing metadata.disaster in {label_path.name}")
            continue

        disaster = disasters.setdefault(
            disaster_id,
            {
                "id": disaster_id,
                "disasterType": metadata.get("disaster_type"),
                "scenes": {},
            },
        )

        scene = disaster["scenes"].setdefault(
            scene_id,
            {
                "sceneId": scene_id,
                "pre": None,
                "post": None,
                "bounds": None,
                "hasFeatures": False,
            },
        )

        feature_count = len(label_data.get("features", {}).get("lng_lat", []))
        scene["hasFeatures"] = bool(scene["hasFeatures"] or feature_count > 0)
        scene["bounds"] = _merge_bounds(scene["bounds"], _extract_label_bounds(label_data))

        scene[phase] = {
            "fileName": label_path.name,
            "metadata": metadata,
            "features": label_data.get("features", {}),
            "imgName": metadata.get("img_name"),
        }

    for disaster in disasters.values():
        scenes = list(disaster["scenes"].values())
        scenes.sort(key=lambda s: s["sceneId"])
        recommended = next((s for s in scenes if s["hasFeatures"]), scenes[0] if scenes else None)
        disaster["sceneCount"] = len(scenes)
        disaster["recommendedSceneId"] = recommended["sceneId"] if recommended else None
        disaster["recommendedBounds"] = recommended["bounds"] if recommended else None

    return {"disasters": disasters, "errors": errors}


def register_disaster_routes(app):
    @app.route("/disasters", methods=["GET"])
    def get_disasters():
        dataset = _build_dataset_index()
        disasters = sorted(dataset["disasters"].values(), key=lambda d: d["id"])

        response = {
            "disasters": [
                {
                    "id": disaster["id"],
                    "disasterType": disaster["disasterType"],
                    "sceneCount": disaster["sceneCount"],
                    "recommendedSceneId": disaster["recommendedSceneId"],
                    "recommendedBounds": disaster["recommendedBounds"],
                }
                for disaster in disasters
            ],
            "errors": dataset["errors"],
        }
        return jsonify(response), 200

    @app.route("/disasters/<disaster_id>/scenes", methods=["GET"])
    def get_disaster_scenes(disaster_id: str):
        dataset = _build_dataset_index()
        disaster = dataset["disasters"].get(disaster_id)
        if disaster is None:
            return jsonify({"error": f"Disaster '{disaster_id}' not found"}), 404

        scenes = sorted(disaster["scenes"].values(), key=lambda s: s["sceneId"])
        return (
            jsonify(
                {
                    "disasterId": disaster_id,
                    "scenes": [
                        {
                            "sceneId": scene["sceneId"],
                            "hasFeatures": scene["hasFeatures"],
                            "bounds": scene["bounds"],
                            "hasPre": scene["pre"] is not None,
                            "hasPost": scene["post"] is not None,
                        }
                        for scene in scenes
                    ],
                    "recommendedSceneId": disaster["recommendedSceneId"],
                    "recommendedBounds": disaster["recommendedBounds"],
                    "errors": dataset["errors"],
                }
            ),
            200,
        )

    @app.route("/disasters/<disaster_id>/scenes/<scene_id>/labels", methods=["GET"])
    def get_scene_labels(disaster_id: str, scene_id: str):
        dataset = _build_dataset_index()
        disaster = dataset["disasters"].get(disaster_id)
        if disaster is None:
            return jsonify({"error": f"Disaster '{disaster_id}' not found"}), 404

        scene = disaster["scenes"].get(scene_id)
        if scene is None:
            return jsonify({"error": f"Scene '{scene_id}' not found for '{disaster_id}'"}), 404

        return (
            jsonify(
                {
                    "disasterId": disaster_id,
                    "sceneId": scene_id,
                    "bounds": scene["bounds"],
                    "hasFeatures": scene["hasFeatures"],
                    "pre": scene["pre"],
                    "post": scene["post"],
                    "errors": dataset["errors"],
                }
            ),
            200,
        )

    @app.route("/disasters/images/<path:filename>", methods=["GET"])
    def get_disaster_image(filename: str):
        _, images_dir = _resolve_data_paths()
        if not images_dir.exists():
            return jsonify({"error": f"Images directory not found: {images_dir}"}), 404
        return send_from_directory(images_dir, filename)

    @app.route("/analyze", methods=["POST"])
    def analyze_with_openrouter():
        payload = request.get_json(silent=True) or {}
        disaster_id = payload.get("disasterId")
        scene_id = payload.get("sceneId")
        feature_id = payload.get("featureId")

        if not disaster_id or not scene_id:
            return jsonify({"status": "error", "error": "disasterId and sceneId are required"}), 400

        dataset = _build_dataset_index()
        disaster = dataset["disasters"].get(disaster_id)
        if disaster is None:
            return jsonify({"status": "error", "error": f"Disaster '{disaster_id}' not found"}), 404

        scene = disaster["scenes"].get(scene_id)
        if scene is None:
            return (
                jsonify({"status": "error", "error": f"Scene '{scene_id}' not found for '{disaster_id}'"}),
                404,
            )

        labels_dir, images_dir = _resolve_data_paths()
        pre_img_name = (scene.get("pre") or {}).get("imgName")
        post_img_name = (scene.get("post") or {}).get("imgName")

        pre_data_url = _encode_image_to_data_url(images_dir / pre_img_name) if pre_img_name else None
        post_data_url = _encode_image_to_data_url(images_dir / post_img_name) if post_img_name else None

        feature = _extract_feature_by_uid(scene, feature_id) if feature_id else None

        try:
            analysis_text = _openrouter_analyze(
                feature=feature,
                pre_data_url=pre_data_url,
                post_data_url=post_data_url,
                disaster_id=disaster_id,
                scene_id=scene_id,
            )
        except requests.RequestException as exc:
            return jsonify({"status": "error", "error": f"OpenRouter request failed: {exc}"}), 502
        except Exception as exc:  # pragma: no cover - runtime guard
            return jsonify({"status": "error", "error": str(exc)}), 500

        return (
            jsonify(
                {
                    "status": "ok",
                    "result": {
                        "text": analysis_text,
                        "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                        "sceneId": scene_id,
                        "disasterId": disaster_id,
                        "featureId": feature_id,
                        "hasPreImage": bool(pre_data_url),
                        "hasPostImage": bool(post_data_url),
                        "labelsDir": str(labels_dir),
                    },
                }
            ),
            200,
        )
