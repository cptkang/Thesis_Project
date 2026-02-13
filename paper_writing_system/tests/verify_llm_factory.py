import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from core.llm import get_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

def test_llm_factory():
    print("Testing LLM Factory...")

    # Test Case 1: Anthropic
    print("\n[Case 1] Testing Anthropic Provider (Default)")
    settings_anthropic = Settings(
        anthropic_api_key="test-key",
        llm_provider="anthropic",
        llm_model="claude-3-sonnet"
    )
    model = get_chat_model(settings_anthropic)
    print(f"Model Type: {type(model)}")
    if isinstance(model, ChatAnthropic):
        print("PASS: Correctly returned ChatAnthropic")
    else:
        print(f"FAIL: Expected ChatAnthropic, got {type(model)}")
    
    # Test Case 2: Google
    print("\n[Case 2] Testing Google Provider")
    settings_google = Settings(
        google_api_key="test-key",
        llm_provider="google",
        llm_model="gemini-1.5-pro"
    )
    model = get_chat_model(settings_google)
    print(f"Model Type: {type(model)}")
    
    if isinstance(model, ChatGoogleGenerativeAI):
        print("PASS: Correctly returned ChatGoogleGenerativeAI")
    else:
        print(f"FAIL: Expected ChatGoogleGenerativeAI, got {type(model)}")
    
    # Test Case 3: Interface Check
    print("\n[Case 3] Interface Check")
    if isinstance(model, BaseChatModel):
        print("PASS: Model implements BaseChatModel")
    else:
        print("FAIL: Model does not implement BaseChatModel")

if __name__ == "__main__":
    test_llm_factory()
