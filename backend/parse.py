'''
This file contains functions for parsing the image JSON files.
This includes calculating the polygons for the bounding boxes and creating the data schema to store VLM predictions in the database. Below covers some important context for the document structure in detail.

Each document has the following structure:

{
  _id: ObjectId('69a208eb04f01085d5bff036'),
  features: {
    lng_lat: [],
    xy: []
  },
  metadata: {
    ...
    id: 'MjU4MDIxNw.xYGyMrFQmA85YrswxICQOFc4RxM',
    img_name: 'socal-fire_00000559_post_disaster.png'
  }
}

There can be multiple features. The location data for each feature is in the lng_lat array. For example:

features: {
    lng_lat: [
      {
        properties: {
          feature_type: 'building',
          uid: '527cf440-4206-4260-9c8a-02327e3c4f1b'
        },
        wkt: 'POLYGON ((-122.673630620297 38.47121671756256, -122.6734791612574 38.4712694327815, -122.6734598363071 38.47123605554015, -122.6735183472852 38.47121730535241, -122.6735127045632 38.47119837635874, -122.6734140325462 38.471231424698, -122.673397990059 38.4712322806053, -122.673388735864 38.47125206870411, -122.6733589095785 38.47126099186175, -122.6733269871311 38.47122667884069, -122.6733751877231 38.47120789997298, -122.6733411406869 38.47113665512795, -122.6735293074611 38.47107323480782, -122.6735702932238 38.471130088825, -122.673510648913 38.47114613398511, -122.6735265573654 38.47117499853837, -122.6735850642166 38.47115714893732, -122.673630620297 38.47121671756256))'
      },
      {
        properties: {
          feature_type: 'building',
          uid: '496e9c26-01b7-45f5-90f2-79ead28ebe77'
        },
        wkt: 'POLYGON ((-122.6735237703349 38.47103088968107, -122.6734839302621 38.47097403884007, -122.6734897272956 38.47095874425628, -122.673455639643 38.47089650564676, -122.6734338692481 38.47089734559275, -122.6734134313202 38.47085676018446, -122.6733951424134 38.4708477028786, -122.6732608923117 38.47089506200524, -122.6733757321547 38.47108721682918, -122.6735237703349 38.47103088968107))'
      },
      ...

On the other hand, the xy array contains the pixel coordinates for the feature in the image. For example:

xy: [
      {
        properties: {
          feature_type: 'building',
          uid: '527cf440-4206-4260-9c8a-02327e3c4f1b'
        },
        wkt: 'POLYGON ((146.5860632463575 68.49001019221352, 176.0339632236028 55.32769126299026, 179.8264957964298 63.58202686267264, 168.4488980779487 68.26692004087074, 169.5643488346625 72.95181321906887, 188.7501018501405 64.69747761938648, 191.8733639689393 64.4743874680437, 193.6580851796814 59.56640413850283, 199.4584291145933 57.33550262507516, 205.7049533521908 65.81292837610032, 196.3351669957946 70.49782155429843, 203.0278715360776 88.12194351037704, 166.4410867158638 103.9613442557135, 158.4098412675242 89.90666472111917, 170.0105291373481 85.89104199694935, 166.8872670185493 78.7521571539808, 155.5096693000682 83.21396018083615, 146.5860632463575 68.49001019221352))'
      },
      {
        properties: {
          feature_type: 'building',
          uid: '496e9c26-01b7-45f5-90f2-79ead28ebe77'
        },
        wkt: 'POLYGON ((167.5565374725776 114.4465813688235, 175.3646927695745 128.5012609034179, 174.2492420128607 132.2937934762449, 180.9419465531437 147.6870139188959, 185.1806594286562 147.4639237675531, 189.1962821528261 157.5029805779776, 192.7657245743103 159.7338820914053, 218.8672722814141 147.9101040702386, 196.3351669957946 100.3919018342292, 167.5565374725776 114.4465813688235))'
      },
    ...


'''

from typing import Any

# Parse the wkt node
def parse_polygon_wkt_bounds(wkt: str) -> list[float] | None:
    # Simple POLYGON ((lng lat, ...)) as in xView2 labels → [min_lng, min_lat, max_lng, max_lat].
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

# Merge 2 bounding boxes
def merge_bounds(base: list[float] | None, nxt: list[float] | None) -> list[float] | None:
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

# Get the bounding boxes for all features in a document
def extract_label_bounds(label_data: dict[str, Any]) -> list[float] | None:
    bounds: list[float] | None = None
    lng_lat_features = label_data.get("features", {}).get("lng_lat", [])
    for feature in lng_lat_features:
        feature_bounds = parse_polygon_wkt_bounds(feature.get("wkt", ""))
        bounds = merge_bounds(bounds, feature_bounds)
    return bounds

# Fetch the feature given its uid
def find_feature_by_uid(
    pre_phase: dict[str, Any] | None,
    post_phase: dict[str, Any] | None,
    feature_id: str,
) -> dict[str, Any] | None:
    for phase_data in (pre_phase, post_phase):
        if not phase_data:
            continue
        for feature in phase_data.get("features", {}).get("lng_lat", []):
            uid = feature.get("properties", {}).get("uid")
            if uid == feature_id:
                return feature
    return None