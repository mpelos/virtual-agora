"""Provider factory for Virtual Agora.

This module provides a factory for creating LangChain chat model instances
based on provider configuration. It handles the mapping between Virtual Agora's
provider types and LangChain's implementation classes.
"""

import os
from typing import Optional, Dict, Any, Union, List, Type
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chat_models import init_chat_model

from virtual_agora.providers.config import (
    ProviderConfig,
    ProviderType,
    GoogleProviderConfig,
    OpenAIProviderConfig,
    AnthropicProviderConfig,
    GrokProviderConfig,
    create_provider_config,
)
from virtual_agora.providers.registry import registry
from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import ConfigurationError

# Import LangGraph retry policy for enhanced error handling
try:
    from langgraph.pregel import RetryPolicy
except ImportError:
    # Fallback if LangGraph is not available
    RetryPolicy = None


logger = get_logger(__name__)


class ProviderFactory:
    """Factory for creating LangChain chat model instances."""

    # Cache for provider instances
    _instance_cache: Dict[str, BaseChatModel] = {}

    @classmethod
    def create_provider(
        cls, config: Union[ProviderConfig, Dict[str, Any]], use_cache: bool = True
    ) -> BaseChatModel:
        """Create a LangChain chat model instance using the init_chat_model pattern.

        This method uses LangChain's recommended init_chat_model() pattern as the primary
        approach, with fallback to legacy provider-specific instantiation if needed.

        Args:
            config: Provider configuration (ProviderConfig or dict)
            use_cache: Whether to use cached instances

        Returns:
            LangChain chat model instance

        Raises:
            ConfigurationError: If provider is not supported or configuration is invalid
            ImportError: If required LangChain package is not installed
        """
        # Convert dict to ProviderConfig if needed
        if isinstance(config, dict):
            config = create_provider_config(**config)

        # Validate configuration
        is_valid, error_msg = registry.validate_model_config(
            config.provider, config.model
        )
        if not is_valid:
            raise ConfigurationError(error_msg)

        # Generate cache key
        cache_key = cls._generate_cache_key(config)

        # Check cache
        if use_cache and cache_key in cls._instance_cache:
            logger.debug(f"Using cached provider instance for {cache_key}")
            return cls._instance_cache[cache_key]

        # Create new instance using init_chat_model pattern (preferred) or fallback to legacy
        try:
            provider_instance = cls._create_provider_with_init_chat_model(config)
            logger.debug(
                f"Successfully created provider using init_chat_model: {config.provider}:{config.model}"
            )
        except (ImportError, ModuleNotFoundError) as e:
            # Missing dependencies - try legacy method
            logger.info(
                f"init_chat_model failed due to missing dependencies, falling back to legacy method: {e}"
            )
            provider_instance = cls._create_provider_instance_legacy(config)
        except ConfigurationError:
            # Configuration errors should not fallback - re-raise
            raise
        except Exception as e:
            # Other errors - log and try legacy method
            logger.warning(
                f"init_chat_model failed unexpectedly, falling back to legacy method: {e}"
            )
            provider_instance = cls._create_provider_instance_legacy(config)

        # Cache instance
        if use_cache:
            cls._instance_cache[cache_key] = provider_instance
            logger.debug(f"Cached provider instance for {cache_key}")

        return provider_instance

    @classmethod
    def create_provider_with_fallbacks(
        cls,
        primary_config: Union[ProviderConfig, Dict[str, Any]],
        fallback_configs: List[Union[ProviderConfig, Dict[str, Any]]] = None,
        use_cache: bool = True,
        max_retries: int = 3,
        retry_on_errors: Optional[List[Type[Exception]]] = None,
    ) -> BaseChatModel:
        """Create a LangChain chat model with fallback providers using .with_fallbacks().

        This method implements LangGraph's recommended fallback pattern by creating
        a primary provider and chaining fallback providers using the .with_fallbacks() method.
        It also supports retry configuration for transient errors.

        Args:
            primary_config: Primary provider configuration
            fallback_configs: List of fallback provider configurations
            use_cache: Whether to use cached instances
            max_retries: Maximum number of retries for each provider
            retry_on_errors: List of exception types to retry on

        Returns:
            LangChain chat model instance with fallbacks configured

        Raises:
            ConfigurationError: If no providers can be created
        """
        # Import here to avoid circular dependency
        from virtual_agora.utils.exceptions import (
            ProviderError,
            TimeoutError,
            NetworkTransientError,
        )

        # Default retry-able errors
        if retry_on_errors is None:
            retry_on_errors = [ProviderError, TimeoutError, NetworkTransientError]

        # Create primary provider with retry configuration
        primary_provider = cls.create_provider(primary_config, use_cache)

        # Apply retry configuration if available
        if hasattr(primary_provider, "with_retry"):
            primary_provider = primary_provider.with_retry(
                stop_after_attempt=max_retries,
                retry_on_exception=lambda e: any(
                    isinstance(e, err_type) for err_type in retry_on_errors
                ),
            )

        # If no fallbacks specified, return primary
        if not fallback_configs:
            return primary_provider

        # Create fallback providers with retry configuration
        fallback_providers = []
        for fallback_config in fallback_configs:
            try:
                fallback_provider = cls.create_provider(fallback_config, use_cache)

                # Apply retry configuration to fallback providers
                if hasattr(fallback_provider, "with_retry"):
                    fallback_provider = fallback_provider.with_retry(
                        stop_after_attempt=max_retries,
                        retry_on_exception=lambda e: any(
                            isinstance(e, err_type) for err_type in retry_on_errors
                        ),
                    )

                fallback_providers.append(fallback_provider)
            except Exception as e:
                logger.warning(f"Failed to create fallback provider: {e}")
                continue

        # Return primary with fallbacks
        if fallback_providers:
            # Use LangChain's .with_fallbacks() method for proper error handling
            return primary_provider.with_fallbacks(
                fallback_providers,
                exceptions_to_handle=(Exception,),  # Handle all exceptions by default
            )
        else:
            logger.warning("No fallback providers could be created")
            return primary_provider

    @classmethod
    def create_provider_with_retry_policy(
        cls,
        config: Union[ProviderConfig, Dict[str, Any]],
        retry_policy: Optional["RetryPolicy"] = None,
        use_cache: bool = True,
    ) -> BaseChatModel:
        """Create a provider with LangGraph RetryPolicy configuration.

        This method creates a provider and wraps it with LangGraph's RetryPolicy
        for better error handling and retry mechanisms.

        Args:
            config: Provider configuration
            retry_policy: LangGraph RetryPolicy instance (if None, creates default)
            use_cache: Whether to use cached instances

        Returns:
            LangChain chat model instance with retry policy

        Raises:
            ConfigurationError: If provider cannot be created
        """
        # Create the base provider
        provider = cls.create_provider(config, use_cache)

        # If RetryPolicy is not available, return provider as-is
        if RetryPolicy is None:
            logger.debug(
                "LangGraph RetryPolicy not available, returning provider without retry policy"
            )
            return provider

        # Create default retry policy if not provided
        if retry_policy is None:
            from virtual_agora.utils.exceptions import (
                ProviderError,
                TimeoutError,
                NetworkTransientError,
            )

            # Default retry policy for provider errors
            retry_policy = RetryPolicy(
                retry_on=lambda e: isinstance(
                    e, (ProviderError, TimeoutError, NetworkTransientError)
                ),
                max_attempts=3,
            )

        # Note: LangGraph RetryPolicy is typically applied at the node level in StateGraph,
        # not directly to the LLM. For direct LLM retry, we use LangChain's built-in retry.
        # This method is provided for future integration with LangGraph workflows.

        logger.info(
            f"Created provider with retry policy (max_attempts={retry_policy.max_attempts})"
        )
        return provider

    @classmethod
    def _create_provider_with_init_chat_model(
        cls, config: ProviderConfig
    ) -> BaseChatModel:
        """Create a provider instance using LangChain's init_chat_model pattern.

        Args:
            config: Provider configuration

        Returns:
            LangChain chat model instance

        Raises:
            ConfigurationError: If provider is not supported
        """
        # Map Virtual Agora provider types to LangChain provider strings
        provider_mapping = {
            ProviderType.GOOGLE: "google_genai",
            ProviderType.OPENAI: "openai",
            ProviderType.ANTHROPIC: "anthropic",
            ProviderType.GROK: "openai",  # Grok uses OpenAI-compatible API
        }

        provider_str = provider_mapping.get(config.provider)
        if not provider_str:
            raise ConfigurationError(f"Unsupported provider: {config.provider}")

        # Build model identifier
        model_identifier = f"{provider_str}:{config.model}"

        # Set up environment variables for API keys
        cls._setup_api_key_environment(config)

        # Build kwargs for init_chat_model
        init_kwargs = {
            "temperature": config.temperature,
            "timeout": float(config.timeout),
        }

        # Add streaming parameter based on provider compatibility
        if config.provider == ProviderType.GOOGLE:
            # Google Gemini uses 'disable_streaming' instead of 'streaming'
            if not config.streaming:
                init_kwargs["disable_streaming"] = True
        else:
            # Other providers use standard 'streaming' parameter
            init_kwargs["streaming"] = config.streaming

        # Add max_tokens if specified
        if config.max_tokens:
            init_kwargs["max_tokens"] = config.max_tokens

        # Add provider-specific parameters
        cls._add_provider_specific_params(config, init_kwargs)

        # Add any extra kwargs
        init_kwargs.update(config.extra_kwargs)

        logger.info(f"Creating provider with init_chat_model: {model_identifier}")

        # Create the model using init_chat_model
        return init_chat_model(model_identifier, **init_kwargs)

    @classmethod
    def _add_provider_specific_params(
        cls, config: ProviderConfig, init_kwargs: Dict[str, Any]
    ) -> None:
        """Add provider-specific parameters to init_kwargs.

        Args:
            config: Provider configuration
            init_kwargs: Dictionary to add parameters to
        """
        if config.provider == ProviderType.GOOGLE:
            # Google Gemini specific parameters
            if hasattr(config, "top_p") and config.top_p is not None:
                init_kwargs["top_p"] = config.top_p
            if hasattr(config, "top_k") and config.top_k is not None:
                init_kwargs["top_k"] = config.top_k
            if hasattr(config, "safety_settings") and config.safety_settings:
                init_kwargs["safety_settings"] = config.safety_settings

        elif config.provider == ProviderType.OPENAI:
            # OpenAI specific parameters
            if hasattr(config, "presence_penalty"):
                init_kwargs["presence_penalty"] = config.presence_penalty
            if hasattr(config, "frequency_penalty"):
                init_kwargs["frequency_penalty"] = config.frequency_penalty
            if hasattr(config, "top_p") and config.top_p is not None:
                init_kwargs["top_p"] = config.top_p
            if hasattr(config, "seed") and config.seed is not None:
                init_kwargs["seed"] = config.seed

        elif config.provider == ProviderType.ANTHROPIC:
            # Anthropic Claude specific parameters
            if hasattr(config, "top_p") and config.top_p is not None:
                init_kwargs["top_p"] = config.top_p
            if hasattr(config, "top_k") and config.top_k is not None:
                init_kwargs["top_k"] = config.top_k

        elif config.provider == ProviderType.GROK:
            # Grok specific configuration (OpenAI-compatible)
            base_url = config.extra_kwargs.get("base_url", "https://api.x.ai/v1")
            init_kwargs["base_url"] = base_url

        logger.debug(
            f"Added provider-specific parameters for {config.provider}: {list(init_kwargs.keys())}"
        )

    @classmethod
    def _setup_api_key_environment(cls, config: ProviderConfig) -> None:
        """Set up environment variables for API keys.

        Args:
            config: Provider configuration
        """
        if config.api_key:
            # Set provider-specific environment variable
            env_var = registry.get_api_key_env_var(config.provider)
            if env_var:
                os.environ[env_var] = config.api_key
        else:
            # Check if API key is available in environment
            env_var = registry.get_api_key_env_var(config.provider)
            if env_var and not os.getenv(env_var):
                raise ConfigurationError(
                    f"API key not found. Please set {env_var} environment variable."
                )

    @classmethod
    def _create_provider_instance_legacy(cls, config: ProviderConfig) -> BaseChatModel:
        """Create a provider instance using legacy direct instantiation.

        This method is kept as a fallback when init_chat_model fails.

        Args:
            config: Provider configuration

        Returns:
            LangChain chat model instance

        Raises:
            ImportError: If required LangChain package is not installed
            ConfigurationError: If provider is not supported
        """
        # Get API key from environment if not provided
        if not config.api_key:
            env_var = registry.get_api_key_env_var(config.provider)
            if env_var:
                config.api_key = os.getenv(env_var)
                if not config.api_key:
                    raise ConfigurationError(
                        f"API key not found. Please set {env_var} environment variable."
                    )

        # Create provider-specific instance
        if config.provider == ProviderType.GOOGLE:
            return cls._create_google_provider_legacy(config)
        elif config.provider == ProviderType.OPENAI:
            return cls._create_openai_provider_legacy(config)
        elif config.provider == ProviderType.ANTHROPIC:
            return cls._create_anthropic_provider_legacy(config)
        elif config.provider == ProviderType.GROK:
            return cls._create_grok_provider_legacy(config)
        else:
            raise ConfigurationError(f"Unsupported provider: {config.provider}")

    @classmethod
    def _create_google_provider_legacy(
        cls, config: GoogleProviderConfig
    ) -> BaseChatModel:
        """Create Google Gemini provider instance."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "langchain-google-genai package is required for Google provider. "
                "Install it with: pip install langchain-google-genai"
            ) from e

        # Build kwargs
        kwargs = {
            "model": config.model,
            "google_api_key": config.api_key,
            "temperature": config.temperature,
            "streaming": config.streaming,
            "timeout": config.timeout,
        }

        # Add optional parameters
        if config.max_tokens:
            kwargs["max_output_tokens"] = config.max_tokens
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.top_k is not None:
            kwargs["top_k"] = config.top_k
        if config.safety_settings:
            kwargs["safety_settings"] = config.safety_settings

        # Add extra kwargs
        kwargs.update(config.extra_kwargs)

        logger.info(f"Creating Google provider with model: {config.model}")
        return ChatGoogleGenerativeAI(**kwargs)

    @classmethod
    def _create_openai_provider_legacy(
        cls, config: OpenAIProviderConfig
    ) -> BaseChatModel:
        """Create OpenAI provider instance."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai package is required for OpenAI provider. "
                "Install it with: pip install langchain-openai"
            ) from e

        # Build kwargs
        kwargs = {
            "model": config.model,
            "api_key": config.api_key,
            "temperature": config.temperature,
            "streaming": config.streaming,
            "timeout": config.timeout,
            "presence_penalty": config.presence_penalty,
            "frequency_penalty": config.frequency_penalty,
        }

        # Add optional parameters
        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.seed is not None:
            kwargs["seed"] = config.seed

        # Add extra kwargs
        kwargs.update(config.extra_kwargs)

        logger.info(f"Creating OpenAI provider with model: {config.model}")
        return ChatOpenAI(**kwargs)

    @classmethod
    def _create_anthropic_provider_legacy(
        cls, config: AnthropicProviderConfig
    ) -> BaseChatModel:
        """Create Anthropic provider instance."""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "langchain-anthropic package is required for Anthropic provider. "
                "Install it with: pip install langchain-anthropic"
            ) from e

        # Build kwargs
        kwargs = {
            "model": config.model,
            "anthropic_api_key": config.api_key,
            "temperature": config.temperature,
            "streaming": config.streaming,
            "timeout": float(config.timeout),  # Anthropic expects float
        }

        # Add optional parameters
        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.top_k is not None:
            kwargs["top_k"] = config.top_k

        # Add extra kwargs
        kwargs.update(config.extra_kwargs)

        logger.info(f"Creating Anthropic provider with model: {config.model}")
        return ChatAnthropic(**kwargs)

    @classmethod
    def _create_grok_provider_legacy(cls, config: GrokProviderConfig) -> BaseChatModel:
        """Create Grok provider instance.

        Note: This is a placeholder. The actual implementation will depend
        on Grok's API and available LangChain integration.
        """
        # Check if Grok uses OpenAI-compatible API
        # If so, we might be able to use ChatOpenAI with a custom base_url
        try:
            from langchain_openai import ChatOpenAI

            # This is speculative - adjust based on actual Grok API
            kwargs = {
                "model": config.model,
                "api_key": config.api_key,
                "temperature": config.temperature,
                "streaming": config.streaming,
                "timeout": config.timeout,
                # Grok might use a custom base URL
                "base_url": config.extra_kwargs.get("base_url", "https://api.x.ai/v1"),
            }

            # Add optional parameters
            if config.max_tokens:
                kwargs["max_tokens"] = config.max_tokens

            # Add extra kwargs
            kwargs.update(config.extra_kwargs)

            logger.info(f"Creating Grok provider with model: {config.model}")
            logger.warning(
                "Grok provider implementation is experimental. "
                "Please verify it works with your Grok API access."
            )
            return ChatOpenAI(**kwargs)

        except ImportError as e:
            raise ImportError(
                "langchain-openai package is required for Grok provider. "
                "Install it with: pip install langchain-openai"
            ) from e

    @classmethod
    def _generate_cache_key(cls, config: ProviderConfig) -> str:
        """Generate a cache key for a provider configuration.

        Args:
            config: Provider configuration

        Returns:
            Cache key string
        """
        # Include key parameters in cache key
        key_parts = [
            config.provider.value,
            config.model,
            str(config.temperature),
            str(config.streaming),
        ]

        # Include max_tokens if set
        if config.max_tokens:
            key_parts.append(str(config.max_tokens))

        return "|".join(key_parts)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the provider instance cache."""
        cls._instance_cache.clear()
        logger.info("Cleared provider instance cache")

    @classmethod
    def get_cache_size(cls) -> int:
        """Get the number of cached provider instances."""
        return len(cls._instance_cache)


def create_provider(
    provider: Union[str, ProviderType], model: str, **kwargs
) -> BaseChatModel:
    """Convenience function to create a provider instance.

    Args:
        provider: Provider type (string or enum)
        model: Model name
        **kwargs: Additional configuration parameters

    Returns:
        LangChain chat model instance

    Raises:
        ConfigurationError: If provider is not supported or configuration is invalid
        ImportError: If required LangChain package is not installed
    """
    # Create configuration
    if isinstance(provider, str):
        config = create_provider_config(provider=provider, model=model, **kwargs)
    else:
        config = create_provider_config(provider=provider.value, model=model, **kwargs)

    # Create provider
    return ProviderFactory.create_provider(config)


def create_provider_with_fallbacks(
    primary_provider: Union[str, ProviderType],
    primary_model: str,
    fallback_configs: List[Dict[str, Any]] = None,
    max_retries: int = 3,
    retry_on_errors: Optional[List[Type[Exception]]] = None,
    **primary_kwargs,
) -> BaseChatModel:
    """Convenience function to create a provider instance with fallbacks.

    Args:
        primary_provider: Primary provider type (string or enum)
        primary_model: Primary model name
        fallback_configs: List of fallback configurations as dicts
        max_retries: Maximum number of retries for each provider
        retry_on_errors: List of exception types to retry on
        **primary_kwargs: Additional configuration parameters for primary provider

    Returns:
        LangChain chat model instance with fallbacks configured

    Raises:
        ConfigurationError: If no providers can be created
        ImportError: If required LangChain package is not installed
    """
    # Create primary configuration
    if isinstance(primary_provider, str):
        primary_config = create_provider_config(
            provider=primary_provider, model=primary_model, **primary_kwargs
        )
    else:
        primary_config = create_provider_config(
            provider=primary_provider.value, model=primary_model, **primary_kwargs
        )

    # Create provider with fallbacks
    return ProviderFactory.create_provider_with_fallbacks(
        primary_config,
        fallback_configs or [],
        max_retries=max_retries,
        retry_on_errors=retry_on_errors,
    )
