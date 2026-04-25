import json
from pathlib import Path

import joblib


def main():
    here = Path(__file__).resolve().parent
    payload = joblib.load(here / "model_svm.pkl")
    classes = [str(x) for x in payload["classes"]]

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing export dependencies. Run:\n"
            "  python -m pip install -U skl2onnx onnx\n"
        ) from exc

    model = payload["model"]
    onnx_model = convert_sklearn(
        model,
        name="asl_pipeline",
        initial_types=[("input", FloatTensorType([None, 63]))],
        options={id(model): {"zipmap": False}},
    )

    public_dir = here / "public"
    public_dir.mkdir(exist_ok=True)
    (public_dir / "model.onnx").write_bytes(onnx_model.SerializeToString())
    (public_dir / "classes.json").write_text(
        json.dumps({"classes": classes}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Wrote public/model.onnx and public/classes.json")


if __name__ == "__main__":
    main()

