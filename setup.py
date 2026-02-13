#!/usr/bin/env python3
"""
ViralOps Engine — One-Click Setup & Launch 🚀
Handles everything: install deps, validate .env, test connections, launch.

Usage:
    python setup.py           # Full setup + launch
    python setup.py --check   # Only validate .env + test connections
    python setup.py --launch  # Skip checks, just launch
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

# Fix Windows terminal encoding (cp1252 can't handle Unicode)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).parent
ENV_FILE = PROJECT_DIR / ".env"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"


def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     🚀 ViralOps Engine — One-Click Setup & Launch      ║")
    print("║     Multi-Channel Social Media Auto-Poster             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def step_install_deps():
    """Install Python dependencies."""
    print("📦 Step 1: Installing dependencies...")
    
    if not REQUIREMENTS.exists():
        print("   ⚠️ requirements.txt not found!")
        return False
    
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS), "-q"],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"   ❌ pip install failed: {result.stderr[:200]}")
        return False
    
    # Also install PyCryptodome for AES encryption (legacy)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "pycryptodome", "-q"],
        capture_output=True, text=True
    )
    
    print("   ✅ All dependencies installed!")
    return True


def step_validate_env():
    """Validate .env file exists and has critical values."""
    print("\n🔍 Step 2: Validating .env configuration...")
    
    if not ENV_FILE.exists():
        env_example = PROJECT_DIR / ".env.example"
        if env_example.exists():
            import shutil
            shutil.copy(env_example, ENV_FILE)
            print("   📝 Created .env from .env.example")
            print("   ⚠️ Please edit .env with your actual credentials!")
            return False
        print("   ❌ No .env or .env.example found!")
        return False
    
    # Load env
    env = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    
    # Check critical variables
    checks = {
        "Core": {
            "OPENAI_API_KEY": ("required", "LLM content generation"),
        },
        "Alerts": {
            "TELEGRAM_BOT_TOKEN": ("optional", "Error alerts via Telegram"),
            "TELEGRAM_CHAT_ID": ("optional", "Telegram chat for alerts"),
        },
        "Shopify Blog": {
            "SHOPIFY_SHOP": ("optional", "Shopify blog auto-posting"),
            "SHOPIFY_ACCESS_TOKEN": ("optional", "Shopify API token"),
        },
        "Publer Bridge": {
            "SENDIBLE_APPLICATION_ID": ("optional", "TikTok/IG/FB via Sendible"),
            "SENDIBLE_USERNAME": ("optional", "Sendible login email"),
            "SENDIBLE_API_KEY": ("optional", "Sendible API key"),
        },
        "Free Platforms": {
            "BLUESKY_MAIN_HANDLE": ("optional", "Bluesky posting"),
            "MASTODON_MAIN_ACCESS_TOKEN": ("optional", "Mastodon posting"),
            "REDDIT_MAIN_CLIENT_ID": ("optional", "Reddit posting"),
            "MEDIUM_MAIN_ACCESS_TOKEN": ("optional", "Medium publishing"),
            "TUMBLR_MAIN_ACCESS_TOKEN": ("optional", "Tumblr posting"),
        },
    }
    
    all_ok = True
    configured_platforms = []
    missing_platforms = []
    
    for group, items in checks.items():
        print(f"\n   [{group}]")
        for key, (level, desc) in items.items():
            value = env.get(key, "")
            has_value = bool(value) and not value.startswith("your-") and not value.startswith("sk-your")
            
            if has_value:
                masked = value[:8] + "..." if len(value) > 12 else value
                print(f"   ✅ {key} = {masked}")
                configured_platforms.append(desc)
            elif level == "required":
                print(f"   ❌ {key} = MISSING (required!)")
                all_ok = False
            else:
                print(f"   ⬜ {key} = not set ({desc})")
                missing_platforms.append(desc)
    
    print(f"\n   📊 Summary:")
    print(f"      Configured: {len(configured_platforms)} services")
    print(f"      Not configured: {len(missing_platforms)} services")
    
    if not all_ok:
        print("\n   ⚠️ Required variables missing! Edit .env before launching.")
    
    return all_ok


def step_test_sendible():
    """Test Sendible connection if configured."""
    print("\n🔗 Step 3: Testing Sendible Bridge connection...")
    
    env = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    
    app_id = env.get("SENDIBLE_APPLICATION_ID", "")
    username = env.get("SENDIBLE_USERNAME", "")
    api_key = env.get("SENDIBLE_API_KEY", "")
    shared_key = env.get("SENDIBLE_SHARED_KEY", "")
    shared_iv = env.get("SENDIBLE_SHARED_IV", "")
    direct_token = env.get("SENDIBLE_ACCESS_TOKEN", "")
    
    if not app_id and not direct_token:
        print("   ⬜ Sendible not configured. Skipping.")
        print("   💡 Run: python setup_sendible.py")
        return True  # Not a failure, just not configured
    
    try:
        import httpx
    except ImportError:
        print("   ❌ httpx not installed!")
        return False
    
    if direct_token:
        # Test with direct token
        print("   📡 Testing with direct access token...")
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(
                    f"https://api.sendible.com/api/v1/services.json",
                    params={"application_id": app_id or "viralops", "access_token": direct_token}
                )
            if resp.status_code == 200:
                print(f"   ✅ Sendible connected! Status: {resp.status_code}")
                return True
            else:
                print(f"   ❌ Sendible returned {resp.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Sendible test failed: {e}")
            return False
    
    if all([app_id, username, api_key, shared_key, shared_iv]):
        try:
            from Crypto.Cipher import AES
            from Crypto.Util.Padding import pad
            import base64
            import json
            import time
            
            # Build access_key
            timestamp = int(time.time())
            payload = json.dumps({
                "user_login": username,
                "user_api_key": api_key,
                "timestamp": timestamp,
            })
            
            key_bytes = base64.b64decode(shared_key)
            iv_bytes = base64.b64decode(shared_iv)
            
            cipher = AES.new(key_bytes, AES.MODE_CBC, iv_bytes)
            encrypted = cipher.encrypt(pad(payload.encode("utf-8"), AES.block_size))
            access_key = base64.b64encode(encrypted).decode("utf-8")
            
            print("   🔐 Encrypted access_key, calling /v1/auth...")
            
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(
                    f"https://api.sendible.com/api/v1/auth",
                    params={"app_id": app_id, "access_key": access_key}
                )
            
            token = resp.text.strip()
            if token.startswith("<error") or not token or len(token) < 5:
                print(f"   ❌ Auth failed: {token[:100]}")
                return False
            
            print(f"   ✅ Sendible connected! Token: {token[:15]}...")
            
            # Get services count
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(
                    f"https://api.sendible.com/api/v1/services.json",
                    params={"application_id": app_id, "access_token": token}
                )
            
            if resp.status_code == 200:
                data = resp.json()
                services = data if isinstance(data, list) else []
                print(f"   📱 Connected social profiles: {len(services)}")
                for svc in services[:5]:
                    name = svc.get("service_name", svc.get("name", "Unknown"))
                    print(f"      • {name}")
            
            return True
        
        except ImportError:
            print("   ⚠️ PyCryptodome not installed. Run: pip install pycryptodome")
            return False
        except Exception as e:
            print(f"   ❌ Sendible test failed: {e}")
            return False
    else:
        print("   ⚠️ Sendible partially configured. Missing some credentials.")
        print("   💡 Run: python setup_sendible.py")
        return True


def step_test_other_platforms():
    """Quick test of other configured platforms."""
    print("\n🌐 Step 4: Checking other platforms...")
    
    env = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    
    platforms = [
        ("Shopify Blog", "SHOPIFY_ACCESS_TOKEN", "shpat_"),
        ("Telegram", "TELEGRAM_BOT_TOKEN", ""),
        ("OpenAI", "OPENAI_API_KEY", "sk-"),
        ("Bluesky", "BLUESKY_MAIN_HANDLE", ""),
        ("Mastodon", "MASTODON_MAIN_ACCESS_TOKEN", ""),
        ("Reddit", "REDDIT_MAIN_CLIENT_ID", ""),
        ("Medium", "MEDIUM_MAIN_ACCESS_TOKEN", ""),
        ("Tumblr", "TUMBLR_MAIN_ACCESS_TOKEN", ""),
    ]
    
    for name, key, prefix in platforms:
        val = env.get(key, "")
        if val and (not prefix or val.startswith(prefix) or not val.startswith("your-")):
            print(f"   ✅ {name}: configured")
        else:
            print(f"   ⬜ {name}: not configured")


def step_launch():
    """Launch ViralOps Engine."""
    print("\n🚀 Step 5: Launching ViralOps Engine...")
    print("   Dashboard: http://localhost:8000")
    print("   Press Ctrl+C to stop")
    print()
    
    os.chdir(PROJECT_DIR)
    os.execv(sys.executable, [sys.executable, "main.py"])


def main():
    print_banner()
    
    args = sys.argv[1:]
    
    if "--launch" in args:
        step_launch()
        return
    
    # Full setup
    step_install_deps()
    ok = step_validate_env()
    step_test_sendible()
    step_test_other_platforms()
    
    if "--check" in args:
        print("\n✅ Check complete!")
        return
    
    print()
    print("=" * 60)
    
    if not ok:
        print("⚠️  Some required config missing. Fix .env first.")
        print()
        print("   Quick setup commands:")
        print("   • Sendible: python setup_sendible.py")
        print("   • Then: python setup.py")
    else:
        print("✅ All checks passed!")
        print()
        choice = input("🚀 Launch ViralOps Engine now? (Y/n): ").strip().lower()
        if choice != "n":
            step_launch()
        else:
            print("\n   To launch later: python main.py")
            print("   Dashboard: http://localhost:8000")


if __name__ == "__main__":
    main()
