#!/usr/bin/env python3
"""
ViralOps Engine — Publer Setup Wizard 🧙‍♂️
Tự động hướng dẫn + setup Publer API credentials.

Replaces setup_sendible.py — Publer is the new bridge publisher.
Publer: $10/mo per account (Business plan) vs Sendible $199/mo.

Usage:
    python setup_publer.py

Wizard sẽ:
1. Mở trình duyệt → Publer signup (Free trial, no credit card)
2. Hướng dẫn upgrade to Business plan → enable API
3. Hướng dẫn tạo API Key → lấy key + workspace ID
4. Test connection (GET /users/me → validate)
5. Tự động ghi vào .env
6. List connected social accounts (TikTok, IG, FB, etc.)
"""

from __future__ import annotations

import os
import sys
import io
import json
import webbrowser
from pathlib import Path
from datetime import datetime

# Fix Windows terminal encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    updated_keys = set()
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in env:
                new_lines.append(f"{key}={env[key]}")
                updated_keys.add(key)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Append new keys not already in file
    for key, value in env.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}")

    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def banner():
    print("\n" + "=" * 60)
    print("  🚀 ViralOps Engine — Publer Setup Wizard")
    print("  📦 Bridge Publisher: Publer REST API")
    print("  💰 ~$10/mo per account (vs Sendible $199/mo)")
    print("=" * 60)


def step_1_signup():
    """Step 1: Publer signup."""
    print("\n── Step 1/6: Publer Account ──")
    print()
    print("  Publer hỗ trợ 13+ mạng: TikTok, IG, FB, Twitter/X,")
    print("  LinkedIn, YouTube, Pinterest, Threads, Bluesky, v.v.")
    print()
    print("  Pricing: $10/mo per social account (Business plan)")
    print("  → 5 accounts (TikTok×3 + Pinterest + LinkedIn) = $50/mo")
    print()

    ans = input("  Bạn đã có tài khoản Publer chưa? (y/n): ").strip().lower()
    if ans != "y":
        print("\n  → Mở trình duyệt để đăng ký Publer (Free trial)...")
        webbrowser.open("https://app.publer.com/users/sign_up")
        input("\n  ⏳ Nhấn Enter sau khi đã đăng ký xong... ")
    else:
        print("  ✅ Tốt! Tiếp tục...")


def step_2_upgrade():
    """Step 2: Upgrade to Business plan."""
    print("\n── Step 2/6: Business Plan (API Access) ──")
    print()
    print("  API chỉ available cho Business & Enterprise plan.")
    print("  Business plan: $10/mo per social account")
    print()
    print("  Nếu bạn chưa upgrade:")
    print("  → Settings → Billing → Upgrade to Business")
    print()

    ans = input("  Bạn đã có Business plan chưa? (y/n): ").strip().lower()
    if ans != "y":
        print("\n  → Mở trang pricing...")
        webbrowser.open("https://publer.com/plans")
        input("\n  ⏳ Nhấn Enter sau khi đã upgrade... ")
    else:
        print("  ✅ Business plan active!")


def step_3_api_key():
    """Step 3: Generate API key."""
    print("\n── Step 3/6: Generate API Key ──")
    print()
    print("  1. Vào Publer → Settings → Access & Login → API Keys")
    print("  2. Click 'Create API Key'")
    print("  3. Name: 'ViralOps Engine'")
    print("  4. Scopes: ✅ workspaces, ✅ accounts, ✅ posts, ✅ media, ✅ analytics")
    print("  5. Click 'Create' → Copy key (sẽ không hiện lại!)")
    print()

    ans = input("  Mở trang API Keys? (y/n): ").strip().lower()
    if ans == "y":
        webbrowser.open("https://app.publer.com/#/settings")

    print()
    api_key = input("  Paste API Key: ").strip()
    if not api_key:
        print("  ❌ API Key không được để trống!")
        sys.exit(1)

    return api_key


def step_4_workspace():
    """Step 4: Get workspace ID."""
    print("\n── Step 4/6: Workspace ID ──")
    print()
    print("  Workspace ID sẽ được tự động detect khi test connection.")
    print("  Nếu bạn có nhiều workspaces, nhập ID cụ thể.")
    print("  Để trống = auto-detect workspace đầu tiên.")
    print()

    workspace_id = input("  Workspace ID (Enter để auto-detect): ").strip()
    return workspace_id


def step_5_test(api_key: str, workspace_id: str):
    """Step 5: Test connection."""
    print("\n── Step 5/6: Test Connection ──")
    print()
    print("  🔄 Đang test kết nối...")

    try:
        import httpx
    except ImportError:
        print("  ⚠️  httpx chưa cài. Chạy: pip install httpx")
        print("  → Skip test, ghi credentials trước.")
        return True, workspace_id, []

    try:
        headers = {
            "Authorization": f"Bearer-API {api_key}",
            "Content-Type": "application/json",
        }

        # Test auth
        with httpx.Client(timeout=15) as client:
            resp = client.get("https://app.publer.com/api/v1/users/me", headers=headers)
            if resp.status_code != 200:
                print(f"  ❌ Auth failed: {resp.status_code} — {resp.text[:200]}")
                return False, workspace_id, []

            user_data = resp.json()
            print(f"  ✅ Authenticated as: {user_data.get('data', user_data).get('email', 'OK')}")

            # Get workspaces
            resp = client.get("https://app.publer.com/api/v1/workspaces", headers=headers)
            if resp.status_code == 200:
                ws_data = resp.json()
                workspaces = ws_data.get("data", ws_data) if isinstance(ws_data, dict) else ws_data
                if isinstance(workspaces, list) and workspaces:
                    if not workspace_id:
                        workspace_id = str(workspaces[0].get("id", ""))
                        print(f"  ✅ Auto-detected workspace: {workspace_id}")
                    for ws in workspaces:
                        print(f"     📁 {ws.get('name', '?')} (ID: {ws.get('id', '?')})")

            # Get accounts
            if workspace_id:
                headers["Publer-Workspace-Id"] = workspace_id

            resp = client.get("https://app.publer.com/api/v1/accounts", headers=headers)
            accounts = []
            if resp.status_code == 200:
                acc_data = resp.json()
                accounts = acc_data.get("data", acc_data) if isinstance(acc_data, dict) else acc_data
                if isinstance(accounts, list):
                    print(f"\n  ✅ Found {len(accounts)} connected accounts:")
                    for acc in accounts:
                        name = acc.get("name", "?")
                        provider = acc.get("type", acc.get("provider", "?"))
                        print(f"     📱 {name} ({provider})")
                else:
                    accounts = []

            if not accounts:
                print("\n  ⚠️  Không có social account nào connected!")
                print("  → Vào Publer dashboard → Connect accounts (TikTok, IG, etc.)")
                webbrowser.open("https://app.publer.com/#/settings/accounts")

            return True, workspace_id, accounts

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False, workspace_id, []


def step_6_save(api_key: str, workspace_id: str, accounts: list):
    """Step 6: Save to .env."""
    print("\n── Step 6/6: Save to .env ──")

    env = load_env()
    env["PUBLER_API_KEY"] = api_key
    if workspace_id:
        env["PUBLER_WORKSPACE_ID"] = workspace_id

    save_env(env)
    print(f"  ✅ Saved to {ENV_FILE}")

    # Summary
    print("\n" + "=" * 60)
    print("  🎉 Publer Setup Complete!")
    print("=" * 60)
    print()
    print(f"  API Key:      {api_key[:12]}...{api_key[-4:]}")
    print(f"  Workspace:    {workspace_id or '(auto-detect)'}")
    print(f"  Accounts:     {len(accounts)}")
    print()
    print("  📋 Next Steps:")
    print("  1. Connect social accounts tại Publer dashboard")
    print("     (TikTok, Pinterest, LinkedIn, etc.)")
    print("  2. Test với ViralOps: python main.py --mode draft")
    print("  3. Schedule posts: python main.py --mode queue")
    print()
    print("  💰 Cost estimate:")
    if accounts:
        est = len(accounts) * 10
        print(f"     {len(accounts)} accounts × $10/mo = ~${est}/mo")
    else:
        print("     $10/mo per social account (Business plan)")
    print(f"     vs Sendible: ~$199/mo (saved ~${199 - len(accounts) * 10}/mo)")
    print()


def main():
    banner()
    step_1_signup()
    step_2_upgrade()
    api_key = step_3_api_key()
    workspace_id = step_4_workspace()
    success, workspace_id, accounts = step_5_test(api_key, workspace_id)

    if not success:
        retry = input("\n  Retry? (y/n): ").strip().lower()
        if retry == "y":
            api_key = step_3_api_key()
            workspace_id = step_4_workspace()
            success, workspace_id, accounts = step_5_test(api_key, workspace_id)

    if success or input("\n  Save anyway? (y/n): ").strip().lower() == "y":
        step_6_save(api_key, workspace_id, accounts)
    else:
        print("\n  ❌ Setup cancelled.")


if __name__ == "__main__":
    main()
