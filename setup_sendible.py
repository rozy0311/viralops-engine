#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  ⚠️  DEPRECATED — Replaced by setup_publer.py (v4.0)       ║
║  Sendible ($199/mo) → Publer (~$10/mo per account)          ║
║  This file is kept for reference only. Do NOT use.          ║
║  Run: python setup_publer.py                                ║
╚══════════════════════════════════════════════════════════════╝

ViralOps Engine — Sendible Setup Wizard 🧙‍♂️ [DEPRECATED]
Tự động hướng dẫn + setup Sendible API credentials.

Usage:
    python setup_publer.py  # ← Use this instead

Wizard sẽ:
1. Mở trình duyệt → Sendible signup (14-day free trial, NO credit card)
2. Hướng dẫn tạo Developer App → lấy application_id, shared_key, shared_iv
3. Lấy username + api_key
4. Test connection
5. Tự động ghi vào .env
6. List connected social services (TikTok, IG, FB, etc.)
"""

from __future__ import annotations

import os
import re
import sys
import io
import json
import webbrowser
from pathlib import Path
from datetime import datetime

# Fix Windows terminal encoding (cp1252 can't handle Unicode)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Load .env if exists
ENV_FILE = Path(__file__).parent / ".env"

def load_env():
    """Load existing .env values."""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


def save_env(env: dict):
    """Save values back to .env, preserving comments and structure."""
    if not ENV_FILE.exists():
        print("❌ .env file not found! Run from project root.")
        sys.exit(1)
    
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in env and env[key]:
                new_lines.append(f"{key}={env[key]}")
                continue
        new_lines.append(line)
    
    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def print_banner():
    print()
    print("=" * 60)
    print("  🚀 ViralOps Engine — Sendible Setup Wizard")
    print("=" * 60)
    print()
    print("  Sendible = bridge to post TikTok, Instagram, Facebook")
    print("  Official REST API — zero bot detection risk!")
    print()
    print("  📋 What you need:")
    print("     1. Sendible account (14-day FREE trial)")
    print("     2. Developer App credentials")
    print("     3. Connect your social accounts in Sendible")
    print()


def step1_signup():
    """Step 1: Sign up for Sendible."""
    print("-" * 60)
    print("  📌 STEP 1: Sendible Account")
    print("-" * 60)
    
    env = load_env()
    
    # Check if already configured
    if env.get("SENDIBLE_APPLICATION_ID") and env.get("SENDIBLE_USERNAME"):
        print("  ✅ Sendible credentials already found in .env!")
        choice = input("  ↳ Overwrite? (y/N): ").strip().lower()
        if choice != "y":
            return True
    
    print()
    print("  Sendible has a 14-day FREE trial (no credit card needed).")
    print("  Plan: Creator ($29/mo) = 6 social profiles, unlimited scheduling.")
    print()
    
    choice = input("  Do you have a Sendible account? (y/N): ").strip().lower()
    
    if choice != "y":
        print()
        print("  🌐 Opening Sendible signup page...")
        print("     URL: https://app.sendible.com/signup")
        print()
        webbrowser.open("https://app.sendible.com/signup?accountTypeId=1937&country=US&freq=1")
        print("  📝 Instructions:")
        print("     1. Sign up with your email")
        print("     2. Verify your email")
        print("     3. Connect TikTok, Instagram, Facebook in Sendible dashboard")
        print("     4. Come back here when done!")
        print()
        input("  Press ENTER when you've created your Sendible account... ")
    
    print("  ✅ Great! Now let's get API credentials.")
    return True


def step2_developer_app():
    """Step 2: Create Developer App and get credentials."""
    print()
    print("-" * 60)
    print("  📌 STEP 2: Create Developer App")
    print("-" * 60)
    print()
    print("  🌐 Opening Sendible Developer Apps page...")
    print()
    
    # Try to open the developer apps PDF guide
    webbrowser.open("https://app.sendible.com/settings/developer-apps")
    
    print("  📝 In Sendible dashboard:")
    print("     1. Go to Settings → Developer Apps")
    print("        (or Settings → scroll down → 'Developer Apps')")
    print("     2. Click 'Create New Application'")
    print("     3. Give it a name (e.g., 'ViralOps Engine')")
    print("     4. After creation, you'll see:")
    print("        - Application ID")
    print("        - Shared Key (Base64)")
    print("        - Shared IV (Base64)")
    print()
    print("  ⚠️  If you can't find Developer Apps:")
    print("     → Try: https://sendible.com/settings")
    print("     → Or contact Sendible support to enable API access")
    print()
    
    # Collect credentials
    print("  📋 Enter your credentials (paste from Sendible):")
    print()
    
    app_id = input("  Application ID: ").strip()
    shared_key = input("  Shared Key (Base64): ").strip()
    shared_iv = input("  Shared IV (Base64): ").strip()
    
    if not all([app_id, shared_key, shared_iv]):
        print("  ❌ All 3 values are required!")
        print("     If you can't find them, check Settings → Developer Apps")
        return None
    
    return {
        "application_id": app_id,
        "shared_key": shared_key,
        "shared_iv": shared_iv,
    }


def step3_user_credentials():
    """Step 3: Get username and API key."""
    print()
    print("-" * 60)
    print("  📌 STEP 3: User Credentials")
    print("-" * 60)
    print()
    print("  Your username = Sendible login email")
    print("  Your API key = from Settings → Account → API section")
    print()
    
    username = input("  Sendible Username (email): ").strip()
    api_key = input("  Sendible API Key: ").strip()
    
    if not all([username, api_key]):
        print("  ❌ Both username and API key are required!")
        return None
    
    return {
        "username": username,
        "api_key": api_key,
    }


def step4_test_connection(creds: dict):
    """Step 4: Test the Sendible API connection."""
    print()
    print("-" * 60)
    print("  📌 STEP 4: Testing Connection...")
    print("-" * 60)
    print()
    
    try:
        import httpx
    except ImportError:
        print("  ⚠️ httpx not installed. Installing...")
        os.system(f"{sys.executable} -m pip install httpx")
        import httpx
    
    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import pad
        import base64
        
        # Build access_key (AES-256-CBC encrypted)
        timestamp = int(datetime.now().timestamp())
        payload = json.dumps({
            "user_login": creds["username"],
            "user_api_key": creds["api_key"],
            "timestamp": timestamp,
        })
        
        key_bytes = base64.b64decode(creds["shared_key"])
        iv_bytes = base64.b64decode(creds["shared_iv"])
        
        cipher = AES.new(key_bytes, AES.MODE_CBC, iv_bytes)
        encrypted = cipher.encrypt(pad(payload.encode("utf-8"), AES.block_size))
        access_key = base64.b64encode(encrypted).decode("utf-8")
        
        # Call /v1/auth
        print("  🔐 Encrypting access_key (AES-256-CBC)...")
        auth_url = f"https://api.sendible.com/api/v1/auth?app_id={creds['application_id']}&access_key={access_key}"
        
        print("  📡 Calling /v1/auth endpoint...")
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(auth_url)
        
        token = resp.text.strip()
        
        # Check if error (XML)
        if token.startswith("<error") or token.startswith("<?xml"):
            print(f"  ❌ Auth failed: {token}")
            return None
        
        if not token or len(token) < 5:
            print(f"  ❌ Invalid token received: '{token}'")
            print(f"     HTTP Status: {resp.status_code}")
            return None
        
        print(f"  ✅ Got access_token: {token[:20]}...")
        
        # Test: get services
        print("  📡 Testing: list connected services...")
        services_url = f"https://api.sendible.com/api/v1/services.json?application_id={creds['application_id']}&access_token={token}"
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(services_url)
        
        if resp.status_code == 200:
            data = resp.json()
            services = data if isinstance(data, list) else data.get("services", data.get("result", []))
            print(f"  ✅ Connected services: {len(services) if isinstance(services, list) else 'OK'}")
            
            if isinstance(services, list):
                for svc in services[:10]:
                    name = svc.get("service_name", svc.get("name", "Unknown"))
                    stype = svc.get("service_type", svc.get("type", ""))
                    sid = svc.get("id", svc.get("service_id", ""))
                    print(f"     • {name} ({stype}) — ID: {sid}")
        else:
            print(f"  ⚠️ Services endpoint returned {resp.status_code}")
            print(f"     Body: {resp.text[:200]}")
        
        return token
    
    except ImportError:
        print("  ⚠️ PyCryptodome not installed. Installing...")
        os.system(f"{sys.executable} -m pip install pycryptodome")
        print("  🔄 Please run setup_sendible.py again!")
        return "NEED_RETRY"
    
    except Exception as e:
        print(f"  ❌ Connection test failed: {e}")
        print()
        print("  Possible causes:")
        print("  • Wrong credentials (double-check in Sendible dashboard)")
        print("  • API access not enabled on your plan")
        print("  • Network issue")
        return None


def step5_save_env(creds: dict, token: str | None):
    """Step 5: Save credentials to .env."""
    print()
    print("-" * 60)
    print("  📌 STEP 5: Saving to .env")
    print("-" * 60)
    print()
    
    env = load_env()
    env["SENDIBLE_APPLICATION_ID"] = creds["application_id"]
    env["SENDIBLE_SHARED_KEY"] = creds["shared_key"]
    env["SENDIBLE_SHARED_IV"] = creds["shared_iv"]
    env["SENDIBLE_USERNAME"] = creds["username"]
    env["SENDIBLE_API_KEY"] = creds["api_key"]
    if token:
        env["SENDIBLE_ACCESS_TOKEN"] = token
    
    save_env(env)
    
    print("  ✅ Credentials saved to .env!")
    print()
    print("  Saved values:")
    print(f"     SENDIBLE_APPLICATION_ID = {creds['application_id'][:10]}...")
    print(f"     SENDIBLE_SHARED_KEY     = {creds['shared_key'][:10]}...")
    print(f"     SENDIBLE_SHARED_IV      = {creds['shared_iv'][:10]}...")
    print(f"     SENDIBLE_USERNAME       = {creds['username']}")
    print(f"     SENDIBLE_API_KEY        = {creds['api_key'][:10]}...")
    if token:
        print(f"     SENDIBLE_ACCESS_TOKEN   = {token[:15]}...")


def step6_next_steps():
    """Step 6: Show what to do next."""
    print()
    print("=" * 60)
    print("  🎉 SENDIBLE SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("  📋 Next steps:")
    print()
    print("  1. Connect social accounts in Sendible dashboard:")
    print("     • TikTok → Settings → Social Profiles → Add TikTok")
    print("     • Instagram → Add Instagram Business")
    print("     • Facebook → Add Facebook Page")
    print("     URL: https://app.sendible.com/settings/services")
    print()
    print("  2. Start ViralOps Engine:")
    print("     python main.py")
    print()
    print("  3. Open dashboard:")
    print("     http://localhost:8000")
    print()
    print("  4. Navigate to '🔗 Sendible Bridge' page")
    print("     → Click 'Test Connection' to verify")
    print("     → See your connected services")
    print("     → Try posting!")
    print()
    print("  💡 Sendible routes posts to TikTok/IG/FB automatically.")
    print("     6 free platforms (Bluesky, Mastodon, Reddit, Medium,")
    print("     Tumblr, Shopify Blog) go through direct API.")
    print()


def main():
    print_banner()
    
    # Step 1: Signup
    step1_signup()
    
    # Step 2: Developer App
    dev_creds = step2_developer_app()
    if not dev_creds:
        print("\n  ❌ Setup cancelled. Run again when ready!")
        return
    
    # Step 3: User Credentials
    user_creds = step3_user_credentials()
    if not user_creds:
        print("\n  ❌ Setup cancelled. Run again when ready!")
        return
    
    # Merge
    creds = {**dev_creds, **user_creds}
    
    # Step 4: Test
    token = step4_test_connection(creds)
    
    if token == "NEED_RETRY":
        return
    
    if token is None:
        print()
        choice = input("  ⚠️ Connection test failed. Save credentials anyway? (y/N): ").strip().lower()
        if choice != "y":
            print("  ❌ Cancelled. Fix credentials and try again!")
            return
    
    # Step 5: Save
    step5_save_env(creds, token)
    
    # Step 6: Next steps
    step6_next_steps()


if __name__ == "__main__":
    main()
