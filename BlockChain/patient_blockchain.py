import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

"""
patient_blockchain.py

A simple patient-owned blockchain that:
- Stores prediction records.
- Allows only the patient to grant/revoke access.
- Ensures tamper-evident linkage using hashes.
- Persists to disk as 'chain.json'.

Rithish-style private chain ;)


How it works:

pred 1 -- pred 1 access update -- pred 2 -- pred 2 access update -- pred 1 access update -- pred 2 access update ...



"""


def utc_now_iso() -> str:
    """Return UTC ISO string without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def hash_block(block: Dict[str, Any]) -> str:
    """Compute SHA-256 over the block excluding its own hash."""
    temp = deepcopy(block)
    temp.pop("hash", None)
    block_bytes = json.dumps(temp, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(block_bytes).hexdigest()


class PatientBlockchain:
    def __init__(self, storage_path: str = "chain.json"):
        self.storage_path = storage_path
        self.chain: List[Dict[str, Any]] = []

        if os.path.exists(self.storage_path):
            self._load_chain()
        else:
            self.create_genesis_block()

    # ---------------- Persistence ----------------

    def _save_chain(self) -> None:
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.chain, f, indent=2)

    def _load_chain(self) -> None:
        with open(self.storage_path, "r", encoding="utf-8") as f:
            self.chain = json.load(f)
        print(f"Loaded blockchain with {len(self.chain)} block(s).")
        if not self.validate_chain(verbose=True):
            raise ValueError("Blockchain file tampered or corrupted!")

    # ---------------- Genesis ----------------

    def create_genesis_block(self) -> None:
        genesis = {
            "prediction_name": "genesis_block",
            "prediction_data": None,
            "access": [],
            "current_transac_ts": utc_now_iso(),
            "prev_hash": "0",
        }
        genesis["hash"] = hash_block(genesis)
        self.chain.append(genesis)
        self._save_chain()
        print("Genesis block created.")

    # ---------------- Helpers ----------------

    def _get_latest_block(self, prediction_name: str) -> Optional[Dict[str, Any]]:
        """Return the latest block for the given prediction."""
        for b in reversed(self.chain):
            if b.get("prediction_name") == prediction_name:
                return b
        return None

    # ---------------- Patient-only actions ----------------

    def record_prediction(self, patient_id: str, prediction_name: str, prediction_data: Dict[str, Any]) -> None:
        """Patient creates a new prediction record."""
        prediction_data = deepcopy(prediction_data)
        prediction_data["patient_id"] = patient_id
        access_list = [patient_id]  # by default only patient can view

        prev_hash = self.chain[-1]["hash"]
        block = {
            "prediction_name": prediction_name,
            "prediction_data": prediction_data,
            "access": access_list,
            "current_transac_ts": utc_now_iso(),
            "prev_hash": prev_hash,
        }
        block["hash"] = hash_block(block)
        self.chain.append(block)
        self._save_chain()
        print(f"[+] Patient '{patient_id}' recorded prediction '{prediction_name}'.")

    def grant_access(self, patient_id: str, prediction_name: str, account: str) -> None:
        """Patient grants access to another account."""
        latest = self._get_latest_block(prediction_name)
        if not latest:
            print("[!] Prediction not found.")
            return

        pd = latest.get("prediction_data") or {}
        owner = pd.get("patient_id")
        if patient_id != owner:
            print(f"Only patient '{owner}' can change access.")
            return

        access = deepcopy(latest["access"])
        if account in access:
            print(f"[!] '{account}' already has access.")
            return

        access.append(account)
        self._add_access_block(prediction_name, pd, access)
        print(f"[+] Patient '{patient_id}' granted access to '{account}'.")

    def revoke_access(self, patient_id: str, prediction_name: str, account: str) -> None:
        """Patient revokes access from an account."""
        latest = self._get_latest_block(prediction_name)
        if not latest:
            print("[!] Prediction not found.")
            return

        pd = latest.get("prediction_data") or {}
        owner = pd.get("patient_id")
        if patient_id != owner:
            print(f"Only patient '{owner}' can change access.")
            return

        access = deepcopy(latest["access"])
        if account not in access:
            print(f"[!] '{account}' has no access to revoke.")
            return

        access.remove(account)
        self._add_access_block(prediction_name, pd, access)
        print(f"[-] Patient '{patient_id}' revoked access from '{account}'.")

    def _add_access_block(self, prediction_name: str, prediction_data: Dict[str, Any], access: List[str]) -> None:
        """Helper for adding new access-state blocks."""
        prev_hash = self.chain[-1]["hash"]
        block = {
            "prediction_name": prediction_name,
            "prediction_data": prediction_data,
            "access": sorted(list(set(access))),
            "current_transac_ts": utc_now_iso(),
            "prev_hash": prev_hash,
        }
        block["hash"] = hash_block(block)
        self.chain.append(block)
        self._save_chain()

    # ---------------- Reading ----------------

    def get_prediction(self, prediction_name: str, requester: str) -> Optional[Dict[str, Any]]:
        """Return the latest prediction data if requester has access."""
        latest = self._get_latest_block(prediction_name)
        if not latest:
            print("[!] Prediction not found.")
            return None

        pd = latest.get("prediction_data") or {}
        access = latest.get("access", [])
        owner = pd.get("patient_id")

        if requester == owner or requester in access:
            return deepcopy(pd)
        else:
            print(f"'{requester}' does not have access to '{prediction_name}'.")
            return None

    def get_current_access(self, prediction_name: str) -> List[str]:
        latest = self._get_latest_block(prediction_name)
        return deepcopy(latest["access"]) if latest else []

    # ---------------- Validation ----------------

    def validate_chain(self, verbose=False) -> bool:
        for i, block in enumerate(self.chain):
            recalculated = hash_block(block)
            if block["hash"] != recalculated:
                if verbose:
                    print(f"[!] Block {i} hash mismatch.")
                return False
            if i > 0 and block["prev_hash"] != self.chain[i - 1]["hash"]:
                if verbose:
                    print(f"[!] Block {i} prev_hash mismatch.")
                return False
        if verbose:
            print("Blockchain integrity OK.")
        return True

    def show_chain(self) -> None:
        print(json.dumps(self.chain, indent=2))


# ---------------- Demo run ----------------
if __name__ == "__main__":
    bc = PatientBlockchain("chain.json")

    patient = "patient123"
    pred_name = "diabetes_pred_2025-11-11T18:00:00Z"
    data = {
        "result": "Positive",
        "confidence": 0.81,
        "timestamp": utc_now_iso(),
    }

    # patient adds a new prediction
    bc.record_prediction(patient, pred_name, data)

    # patient grants doctor access
    bc.grant_access(patient, pred_name, "doctorA")

    # doctor tries to view
    print("\nDoctorA reads:")
    print(bc.get_prediction(pred_name, requester="doctorA"))

    # patient revokes doctor
    bc.revoke_access(patient, pred_name, "doctorA")

    # doctor tries again
    print("\nDoctorA reads after revoke:")
    print(bc.get_prediction(pred_name, requester="doctorA"))

    # verify chain
    bc.validate_chain(verbose=True)

    # show final chain
    bc.show_chain()

