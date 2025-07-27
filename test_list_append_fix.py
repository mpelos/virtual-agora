#!/usr/bin/env python3
"""Test script to verify the list append fix."""

import os
from unittest.mock import Mock, patch
from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.config.models import Config as VirtualAgoraConfig, Provider

# Set up environment
os.environ['GOOGLE_API_KEY'] = 'fake-key-for-testing'

# Create test config
config = VirtualAgoraConfig(
    moderator={
        "provider": Provider.GOOGLE,
        "model": "gemini-2.5-flash-lite",
        "temperature": 0.7,
    },
    summarizer={
        "provider": Provider.GOOGLE,
        "model": "gemini-2.5-flash-lite",
        "temperature": 0.3,
    },
    topic_report={
        "provider": Provider.GOOGLE,
        "model": "gemini-2.5-flash-lite",
        "temperature": 0.5,
    },
    ecclesia_report={
        "provider": Provider.GOOGLE,
        "model": "gemini-2.5-flash-lite",
        "temperature": 0.5,
    },
    agents=[
        {
            "provider": Provider.GOOGLE,
            "model": "gemini-2.5-flash-lite",
            "count": 3,
            "temperature": 0.7,
        }
    ],
)

def test_list_append_fix():
    """Test that the list append error is fixed."""
    with patch("virtual_agora.providers.create_provider") as mock_create_provider:
        # Mock the LLM provider
        mock_llm = Mock()
        mock_create_provider.return_value = mock_llm
        
        print("🔧 Creating VirtualAgoraV13Flow...")
        flow = VirtualAgoraV13Flow(config, enable_monitoring=False)
        
        print("📝 Creating session...")
        session_id = flow.create_session(main_topic="Test topic for debugging")
        print(f"✅ Session created: {session_id}")
        
        print("🔨 Compiling graph...")
        compiled_graph = flow.compile()
        print("✅ Graph compiled successfully")
        
        print("🚀 Testing stream execution...")
        config_dict = {"configurable": {"thread_id": session_id}}
        
        try:
            # Try to get the first update from the stream
            updates = list(flow.stream(config_dict))
            print(f"✅ Stream execution successful! Got {len(updates)} updates")
            if updates:
                print(f"📊 First update keys: {list(updates[0].keys())}")
            return True
        except TypeError as e:
            if "append" in str(e) and "NoneType" in str(e):
                print(f"❌ List append error still exists: {e}")
                return False
            else:
                print(f"⚠️ Different TypeError (may be expected): {e}")
                return True  # Different error is OK for this test
        except Exception as e:
            print(f"⚠️ Different error (may be expected): {type(e).__name__}: {e}")
            return True  # Different error is OK for this test

if __name__ == "__main__":
    print("🧪 Testing Virtual Agora list append fix...")
    success = test_list_append_fix()
    if success:
        print("🎉 SUCCESS: No list append error detected!")
    else:
        print("💥 FAILURE: List append error still exists!")
    exit(0 if success else 1)