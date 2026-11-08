"""Build empirical input ranges for the active 113-row model cohort."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "input_ranges.json"


def summarize(values: pd.Series, source_ids: pd.Series) -> dict:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    aligned = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "source": source_ids.astype(str)})
    aligned = aligned.dropna(subset=["value"])
    positive = numeric[numeric > 1e-12]
    return {
        "min": float(numeric.min()) if len(numeric) else None,
        "positive_min": float(positive.min()) if len(positive) else None,
        "positive_max": float(positive.max()) if len(positive) else None,
        "p05": float(numeric.quantile(.05)) if len(numeric) else None,
        "p95": float(numeric.quantile(.95)) if len(numeric) else None,
        "max": float(numeric.max()) if len(numeric) else None,
        "n_rows": int(len(numeric)),
        "n_papers": int(aligned["source"].nunique()),
        "n_nonzero_rows": int((numeric.abs() > 1e-12).sum()),
        "n_nonzero_papers": int(aligned.loc[aligned["value"].abs() > 1e-12, "source"].nunique()),
    }


def main() -> None:
    refs = pd.read_csv(DATA / "model_reference_records.csv", low_memory=False).fillna("")
    manifest = json.loads((ROOT / "outputs" / "taxonomy_model_manifest.json").read_text(encoding="utf-8"))
    catalog = json.loads((DATA / "substance_catalog.json").read_text(encoding="utf-8"))
    features = manifest["numeric_features"]
    ranges = {}
    families = {str(x["id"]): str(x.get("family") or "unknown") for x in catalog}
    role_map = {str(k).strip().lower(): str(v) for k, v in manifest.get("role_map", {}).items()}
    def normalize_role(value: object) -> str:
        text = str(value or "").strip().lower()
        return role_map.get(text, text.replace(" ", "_"))

    for feature in features:
        if feature.startswith("family_wt__"):
            family = feature.removeprefix("family_wt__")
            vals = []
            for raw in refs["components_json"]:
                try:
                    comps = json.loads(raw) if raw else []
                except (TypeError, json.JSONDecodeError):
                    comps = []
                vals.append(sum(float(c.get("wt_pct") or 0) for c in comps
                                if families.get(str(c.get("substance_id")), "unknown") == family))
            series = pd.Series(vals)
        elif feature.startswith("function_wt__"):
            role = feature.removeprefix("function_wt__")
            vals = []
            for raw in refs["components_json"]:
                try:
                    comps = json.loads(raw) if raw else []
                except (TypeError, json.JSONDecodeError):
                    comps = []
                vals.append(sum(float(c.get("wt_pct") or 0) for c in comps
                                if normalize_role(c.get("role")) == role))
            series = pd.Series(vals)
        else:
            series = refs[feature] if feature in refs.columns else pd.Series([None] * len(refs))
        ranges[feature] = summarize(series, refs["source_id"])

    payload = {
        "schema_version": "1.0",
        "basis": "Current active model cohort only; empirical per-feature ranges, not guaranteed valid combinations.",
        "n_rows": int(len(refs)),
        "n_papers": int(refs["source_id"].nunique()),
        "scope": manifest.get("scope", {}),
        "ranges": ranges,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} from {len(refs)} rows / {refs['source_id'].nunique()} papers")


if __name__ == "__main__":
    main()
