"""
LLM Factory Module
Provides a factory function to create LangChain chat models based on configuration.
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import Settings

logger = logging.getLogger(__name__)


def get_chat_model(settings: Settings) -> BaseChatModel:
    """
    설정에 따라 적절한 Chat Model을 반환합니다.

    Args:
        settings: 애플리케이션 설정

    Returns:
        BaseChatModel 인스턴스 (ChatAnthropic 또는 ChatGoogleGenerativeAI)
    """
    provider = settings.llm_provider.lower()
    
    if provider == "google" or provider == "gemini":
        if not settings.google_api_key:
            logger.warning("Google API Key가 설정되지 않았습니다.")
        
        logger.info(f"Google Gemini 모델 초기화: {settings.llm_model}")
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_tokens,
            convert_system_message_to_human=True, 
        )
    
    # Default to Anthropic
    if provider != "anthropic":
        logger.warning(f"알 수 없는 LLM Provider: {provider}. Anthropic으로 기본 설정합니다.")

    if not settings.anthropic_api_key:
        logger.warning("Anthropic API Key가 설정되지 않았습니다.")

    logger.info(f"Anthropic Claude 모델 초기화: {settings.llm_model}")
    return ChatAnthropic(
        model=settings.llm_model,
        anthropic_api_key=settings.anthropic_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
