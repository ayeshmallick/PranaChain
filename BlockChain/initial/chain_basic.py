# 1. The "Ledger"
ledger = []

# -----------------------------
# 2. Chaincode (Smart Contracts)
# -----------------------------
def grant_consent(patient_id, doctor_id):
    tx = {
        "action": "GRANT",
        "patient": patient_id,
        "doctor": doctor_id
    }
    ledger.append(tx)
    return f"Consent granted: {patient_id} -> {doctor_id}"

def revoke_consent(patient_id, doctor_id):
    tx = {
        "action": "REVOKE",
        "patient": patient_id,
        "doctor": doctor_id
    }
    ledger.append(tx)
    return f"Consent revoked: {patient_id} -> {doctor_id}"

def check_consent(patient_id, doctor_id):
    # Scan ledger from start to end (immutability!)
    status = "NO CONSENT"
    for tx in ledger:
        if tx["patient"] == patient_id and tx["doctor"] == doctor_id:
            if tx["action"] == "GRANT":
                status = "CONSENT GRANTED"
            elif tx["action"] == "REVOKE":
                status = "CONSENT REVOKED"
    return status

# 3. Simulation
print("=== Blockchain-Powered EHR Simulation ===")

# Patient grants consent
print(grant_consent("patient123", "doctor456"))

# Another grant
print(grant_consent("patient123", "lab789"))

# Doctor tries to check
print("Doctor456 status:", check_consent("patient123", "doctor456"))

# Patient revokes consent
print(revoke_consent("patient123", "doctor456"))

# Check again
print("Doctor456 status:", check_consent("patient123", "doctor456"))

# Print entire ledger (like blockchain blocks)
print("\n=== Ledger (Blockchain History) ===")
for i, tx in enumerate(ledger, 1):
    print(f"Block {i}: {tx}")
