"""
Model Registry
==============
Maintains versioning, metadata tracking, and loading of serialized models 
promoted to staging or production for live paper trading.
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import shutil

class ModelRegistry:
    def __init__(self, registry_path: str = "model_registry"):
        self.registry_dir = Path(registry_path)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "registry_index.json"
        
        if not self.metadata_file.exists():
            with open(self.metadata_file, "w") as f:
                json.dump({"models": {}}, f)
                
    def register_model(self, model_zip_source: str, algorithm: str, oos_metrics: Dict[str, Any], hyperparams: Dict[str, Any]) -> str:
        """Saves model to registry and logs the provenance."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_id = f"{algorithm.lower()}_{timestamp}"
        
        target_dir = self.registry_dir / model_id
        target_dir.mkdir(exist_ok=True)
        
        target_zip = target_dir / "model.zip"
        shutil.copy2(model_zip_source, target_zip)
        
        meta = {
            "model_id": model_id,
            "algorithm": algorithm,
            "hyperparameters": hyperparams,
            "oos_metrics": oos_metrics,
            "registered_at": timestamp,
            "status": "candidate" # can be promoted to 'staging' or 'production' later.
        }
        
        with open(target_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=4)
            
        self._update_index(model_id, meta)
        
        print(f"[Registry] Model {model_id} successfully saved to registry.")
        return model_id
        
    def promote_model(self, model_id: str, new_status: str):
        """Update model status (e.g. promoting from candidate to production)."""
        idx = self._load_index()
        if model_id in idx["models"]:
            idx["models"][model_id]["status"] = new_status
            idx["models"][model_id]["promoted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._write_index(idx)
            
            # Sub-meta
            meta_path = self.registry_dir / model_id / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                meta["status"] = new_status
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=4)
                    
            print(f"[Registry] {model_id} promoted to {new_status.upper()}")
        else:
            raise KeyError(f"Model ID {model_id} not found in registry.")

    def get_production_model(self) -> Optional[Path]:
        """Returns the path to the currently active production model."""
        idx = self._load_index()
        for m_id, meta in idx.get("models", {}).items():
            if meta.get("status") == "production":
                return self.registry_dir / m_id / "model.zip"
        return None

    def _load_index(self) -> Dict[str, Any]:
        with open(self.metadata_file, "r") as f:
            return json.load(f)
            
    def _write_index(self, data: Dict[str, Any]):
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=4)
            
    def _update_index(self, model_id: str, meta: Dict[str, Any]):
        idx = self._load_index()
        idx["models"][model_id] = meta
        self._write_index(idx)
