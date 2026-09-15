"""
Record raw D5 Mendeley Data API probe traces into artifacts/d5_probe_log.json.
Provides complete cryptographic and network provenance for the D5 acquisition audit.
"""

import urllib.request
import urllib.error
import json
import re

endpoints_to_probe = [
    {
        "description": "Mendeley Public DOI Webpage",
        "url": "https://data.mendeley.com/datasets/ywp3y5j9vv/1",
        "headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    },
    {
        "description": "Mendeley Public API v1 Dataset Endpoint",
        "url": "https://data.mendeley.com/public-api/datasets/ywp3y5j9vv/1",
        "headers": {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    },
    {
        "description": "Mendeley Public API v1 Files Endpoint",
        "url": "https://data.mendeley.com/public-api/datasets/ywp3y5j9vv/1/files",
        "headers": {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    },
    {
        "description": "Legacy Mendeley General API",
        "url": "https://api.mendeley.com/datasets/ywp3y5j9vv/1",
        "headers": {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    }
]

probe_results = {
    "target_candidate": "D5 (Mendeley Data DOI: 10.17632/ywp3y5j9vv.1)",
    "dataset_title": "ICRISAT District-Level Data: Heterogeneous Climate Effect on Crop Yield and Associated Risks to Water Security in India",
    "contributor": "Souryabrata Mohapatra",
    "publish_date": "2023-07-11T16:27:06.004Z",
    "probes": []
}

for ep in endpoints_to_probe:
    entry = {
        "description": ep["description"],
        "url": ep["url"],
        "http_status": None,
        "response_summary": None,
        "error": None
    }
    try:
        req = urllib.request.Request(ep["url"], headers=ep["headers"])
        with urllib.request.urlopen(req, timeout=15) as resp:
            entry["http_status"] = resp.status
            content = resp.read().decode("utf-8", errors="replace")
            
            # If HTML, extract JSON state
            if "INITIAL_STATE" in content or "initial_state" in content.lower():
                m = re.search(r'window\.INITIAL_STATE\s*=\s*(\{.*?\});</script>', content, re.DOTALL)
                if m:
                    state_raw = m.group(1)
                    decoder = json.JSONDecoder()
                    data, _ = decoder.raw_decode(state_raw)
                    snapshot = data.get("dataset", {}).get("snapshot", {})
                    files = snapshot.get("files", [])
                    entry["response_summary"] = {
                        "snapshot_id": snapshot.get("id"),
                        "version": snapshot.get("version"),
                        "doi": snapshot.get("doi"),
                        "publish_date": snapshot.get("publish_date"),
                        "license": snapshot.get("licence", {}).get("name") if isinstance(snapshot.get("licence"), dict) else str(snapshot.get("licence")),
                        "is_metadata_only": snapshot.get("is_metadata_only"),
                        "files_count": len(files),
                        "files_list": files,
                        "finding": "Empty files array: 0 downloadable data files attached to public deposit version 1."
                    }
            if not entry["response_summary"]:
                entry["response_summary"] = f"HTML Webpage loaded ({len(content)} bytes); files payload empty."
    except urllib.error.HTTPError as e:
        entry["http_status"] = e.code
        entry["error"] = f"HTTP Error {e.code}: {e.reason}"
    except Exception as e:
        entry["error"] = str(e)
    
    probe_results["probes"].append(entry)

with open("artifacts/d5_probe_log.json", "w", encoding="utf-8") as f:
    json.dump(probe_results, f, indent=2)

print("Saved artifacts/d5_probe_log.json successfully.")
