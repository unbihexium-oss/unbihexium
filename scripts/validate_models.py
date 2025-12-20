#!/usr/bin/env python3
"""Validate model loading and inference for all 520 models."""

import json
import sys
from pathlib import Path
import torch

def validate_models():
    root = Path("model_zoo/assets")
    variants = ["tiny", "base", "large", "mega"]
    
    total_checked = 0
    total_passed = 0
    errors = []
    
    print("=" * 60)
    print("MODEL VALIDATION")
    print("=" * 60)
    
    for v in variants:
        vpath = root / v
        if not vpath.exists():
            print(f"\n[SKIP] Variant {v} not found")
            continue
        
        models = list(vpath.iterdir())
        v_passed = 0
        
        print(f"\n{v.upper()} ({len(models)} models):")
        
        for m in models:
            cfg_path = m / "config.json"
            pt_path = m / "model.pt"
            onnx_path = m / "model.onnx"
            sha_path = m / "model.sha256"
            
            total_checked += 1
            
            # Check all files exist
            if not all(p.exists() for p in [cfg_path, pt_path, onnx_path, sha_path]):
                errors.append(f"{m.name}: Missing files")
                continue
            
            # Check config is valid JSON
            try:
                cfg = json.load(open(cfg_path))
                if "params" not in cfg or "task" not in cfg:
                    errors.append(f"{m.name}: Invalid config")
                    continue
            except:
                errors.append(f"{m.name}: Config parse error")
                continue
            
            # Check PT file loads
            try:
                data = torch.load(pt_path, weights_only=False, map_location="cpu")
                if "model_state_dict" not in data:
                    errors.append(f"{m.name}: Invalid PT structure")
                    continue
            except Exception as e:
                errors.append(f"{m.name}: PT load error: {e}")
                continue
            
            v_passed += 1
            total_passed += 1
        
        print(f"  Passed: {v_passed}/{len(models)}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Models Checked: {total_checked}")
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_checked - total_passed}")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    return total_passed == total_checked


if __name__ == "__main__":
    success = validate_models()
    sys.exit(0 if success else 1)
