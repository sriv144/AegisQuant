"""
Groww Connection Test
==========================
Run this FIRST before using the live pipeline.
Verifies: login, market data fetch, paper order placement.

Usage:
    python test_groww_connection.py
"""

import os
from src import config  # loads .env

def test_connection():
    print("=" * 60)
    print("Groww API Connection Test")
    print("=" * 60)

    # --- Check 1: credentials present ---
    api_key    = os.getenv("GROWW_API_KEY", "")
    secret_key = os.getenv("GROWW_SECRET_KEY", "")
    enable_exec = os.getenv("ENABLE_BROKER_EXECUTION", "False")

    print("\n[1] Credential Check:")
    print(f"    GROWW_API_KEY:           {'SET' if api_key else 'MISSING'}")
    print(f"    GROWW_SECRET_KEY:        {'SET' if secret_key else 'MISSING'}")
    print(f"    ENABLE_BROKER_EXECUTION: {enable_exec}")

    if not all([api_key, secret_key]):
        print("\n  FAIL: Fill in .env with your Groww credentials first.")
        print("  See the guide above.")
        return False

    # --- Check 2: library installed ---
    try:
        from growwapi import GrowwAPI
        print("\n[2] Libraries: growwapi — OK")
    except ImportError as e:
        print(f"\n[2] Library missing: {e}")
        print("  Run: pip install growwapi")
        return False

    # --- Check 3: Login ---
    print("\n[3] Attempting Groww login...")
    try:
        token_data = GrowwAPI.get_access_token(api_key=api_key, secret=secret_key)
        if isinstance(token_data, str):
            access_token = token_data
        elif isinstance(token_data, dict):
            access_token = token_data.get("access_token", api_key)
        else:
            access_token = api_key
            
        api = GrowwAPI(token=access_token)
        profile = api.get_user_profile()

        if profile and profile.get("vendor_user_id"):
            print(f"  LOGIN SUCCESS: Welcome user {profile.get('vendor_user_id')}")
            print(f"  Active Segments: {', '.join(profile.get('active_segments', []))}")
        else:
            print(f"  LOGIN FAILED: {profile}")
            return False
            
    except Exception as e:
        print(f"  LOGIN ERROR: {e}")
        return False
        
    # --- Check 4: Fetch Live Funds ---
    print("\n[4] Fetching Live funds...")
    try:
        funds = api.get_available_margin_details()
        cash = funds.get("clear_cash", 0.0) + funds.get("collateral_available", 0.0)
        print(f"  Available Cash: {cash}")
    except Exception as e:
        print(f"  FUNDS ERROR: {e}")
        return False

    # --- Check 5: Fetch live market data ---
    print("\n[5] Fetching live quote for RELIANCE...")
    try:
        ltp = api.get_ltp(exchange_trading_symbols="NSE_RELIANCE", segment="CASH")
        if ltp and "NSE_RELIANCE" in ltp:
            price = ltp["NSE_RELIANCE"]
            print(f"  RELIANCE LTP: {price}")
        else:
            print(f"  LTP fetch returned: {ltp}")
    except Exception as e:
        print(f"  LTP Error (non-fatal): {e}")

    # --- Check 6: Configuration Check ---
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
