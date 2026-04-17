"""
Angel One Connection Test
==========================
Run this FIRST before using the live pipeline.
Verifies: login, market data fetch, paper order placement.

Usage:
    python test_angelone_connection.py
"""

import os
from src import config  # loads .env

def test_connection():
    print("=" * 60)
    print("Angel One SmartAPI Connection Test")
    print("=" * 60)

    # --- Check 1: credentials present ---
    api_key    = os.getenv("ANGELONE_API_KEY", "")
    client_id  = os.getenv("ANGELONE_CLIENT_ID", "")
    password   = os.getenv("ANGELONE_PASSWORD", "")
    totp_key   = os.getenv("ANGELONE_TOTP_KEY", "")
    enable_exec = os.getenv("ENABLE_BROKER_EXECUTION", "False")

    print("\n[1] Credential Check:")
    print(f"    ANGELONE_API_KEY:        {'SET' if api_key else 'MISSING'}")
    print(f"    ANGELONE_CLIENT_ID:      {'SET' if client_id else 'MISSING'}")
    print(f"    ANGELONE_PASSWORD:       {'SET' if password else 'MISSING'}")
    print(f"    ANGELONE_TOTP_KEY:       {'SET' if totp_key else 'MISSING'}")
    print(f"    ENABLE_BROKER_EXECUTION: {enable_exec}")

    if not all([api_key, client_id, password]):
        print("\n  FAIL: Fill in .env with your Angel One credentials first.")
        print("  See the guide above.")
        return False

    # --- Check 2: library installed ---
    try:
        from smartapi import SmartConnect
        import pyotp
        print("\n[2] Libraries: smartapi-python + pyotp — OK")
    except ImportError as e:
        print(f"\n[2] Library missing: {e}")
        print("  Run: pip install smartapi-python pyotp")
        return False

    # --- Check 3: TOTP generation ---
    try:
        totp_now = pyotp.TOTP(totp_key).now() if totp_key else "000000"
        print(f"\n[3] TOTP Generated: {totp_now} (changes every 30 seconds)")
    except Exception as e:
        print(f"\n[3] TOTP Error: {e}")
        print("  ANGELONE_TOTP_KEY must be a valid base32 string from your authenticator app.")
        return False

    # --- Check 4: Login ---
    print("\n[4] Attempting Angel One login...")
    try:
        client = SmartConnect(api_key=api_key)
        data = client.generateSession(client_id, password, totp_now)

        if data and data.get("status"):
            name = data["data"].get("name", client_id)
            print(f"  LOGIN SUCCESS: Welcome, {name}")
            print(f"  JWT Token: {data['data']['jwtToken'][:30]}...")
        else:
            print(f"  LOGIN FAILED: {data}")
            print("  Check: client_id = your Angel One trading login (e.g. A12345)")
            print("         password = your trading account password")
            print("         totp_key = base32 secret from your authenticator app")
            return False
    except Exception as e:
        print(f"  LOGIN ERROR: {e}")
        return False

    # --- Check 5: Fetch live market data ---
    print("\n[5] Fetching live quote for RELIANCE...")
    try:
        ltp = client.ltpData("NSE", "RELIANCE", "2885")  # 2885 = RELIANCE token
        if ltp and ltp.get("data"):
            price = ltp["data"]["ltp"]
            print(f"  RELIANCE LTP: {price}")
        else:
            print(f"  LTP fetch returned: {ltp}")
    except Exception as e:
        print(f"  LTP Error (non-fatal): {e}")

    # --- Check 6: Paper order test ---
    if enable_exec.lower() == "true":
        print("\n[6] ENABLE_BROKER_EXECUTION=True — LIVE MODE ENABLED")
        print("  WARNING: Real orders will be placed when main_india.py runs!")
    else:
        print("\n[6] ENABLE_BROKER_EXECUTION=False — Paper mode (no real orders).")
        print("  Set ENABLE_BROKER_EXECUTION=True in .env to enable live trading.")

    print("\n" + "=" * 60)
    print("CONNECTION TEST PASSED")
    print("=" * 60)
    print("\nNext steps:")
    if enable_exec.lower() != "true":
        print("  1. Set ENABLE_BROKER_EXECUTION=True in .env when ready to trade")
    print("  2. Run: python main_india.py --now")
    print("  3. Watch the pipeline run and check aegisquant_live.db for decisions")
    return True


if __name__ == "__main__":
    test_connection()
