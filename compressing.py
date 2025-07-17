import joblib
from pathlib import Path

# Load the existing uncompressed model
model_path = Path("final_uber_ensemble_model.pkl")
model = joblib.load(model_path)

# Save with compression (level 3 is a good balance)
compressed_path = Path("final_uber_ensemble_model_compressed.pkl")
joblib.dump(model, compressed_path, compress=3)

print(f"✅ Compressed model saved to: {compressed_path.resolve()}")
