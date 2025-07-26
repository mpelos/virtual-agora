"""Agent response validation for Virtual Agora.

This module provides validation middleware for agent responses to ensure
they meet format and content requirements.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
import re
import json
from datetime import datetime
from dataclasses import dataclass

from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import ValidationError

logger = get_logger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    PERMISSIVE = "permissive"
    STANDARD = "standard"
    STRICT = "strict"


class ResponseType(Enum):
    """Types of agent responses to validate."""
    DISCUSSION = "discussion"
    TOPIC_PROPOSAL = "topic_proposal"
    AGENDA_VOTE = "agenda_vote"
    CONCLUSION_VOTE = "conclusion_vote"
    MINORITY_CONSIDERATION = "minority_consideration"


@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool
    response_type: ResponseType
    original_response: str
    cleaned_response: Optional[str] = None
    issues: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    validation_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}
        if self.validation_time is None:
            self.validation_time = datetime.now()


class ResponseValidator:
    """Validates agent responses for format and content compliance."""
    
    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        max_response_length: int = 2000,
        min_response_length: int = 10,
        max_retries: int = 2
    ):
        """Initialize response validator.
        
        Args:
            validation_level: Strictness of validation
            max_response_length: Maximum allowed response length
            min_response_length: Minimum required response length
            max_retries: Maximum retry attempts for failed validation
        """
        self.validation_level = validation_level
        self.max_response_length = max_response_length
        self.min_response_length = min_response_length
        self.max_retries = max_retries
        
        # Validation statistics
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "retries_attempted": 0,
            "validation_errors": {},
            "response_type_stats": {}
        }
        
        logger.info(
            f"Initialized ResponseValidator with level={validation_level.value}, "
            f"length_range=({min_response_length}, {max_response_length})"
        )
    
    def validate_response(
        self,
        response: str,
        response_type: ResponseType,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None
    ) -> ValidationResult:
        """Validate an agent response.
        
        Args:
            response: The response to validate
            response_type: Type of response being validated
            context: Optional context for validation
            agent_id: Optional agent ID for logging
            
        Returns:
            ValidationResult with validation outcome
        """
        self.validation_stats["total_validations"] += 1
        
        # Initialize validation result
        result = ValidationResult(
            is_valid=True,
            response_type=response_type,
            original_response=response,
            cleaned_response=response
        )
        
        try:
            # Basic validation
            self._validate_basic_format(response, result)
            
            # Type-specific validation
            if response_type == ResponseType.DISCUSSION:
                self._validate_discussion_response(response, result, context)
            elif response_type == ResponseType.TOPIC_PROPOSAL:
                self._validate_topic_proposal(response, result, context)
            elif response_type == ResponseType.AGENDA_VOTE:
                self._validate_agenda_vote(response, result, context)
            elif response_type == ResponseType.CONCLUSION_VOTE:
                self._validate_conclusion_vote(response, result, context)
            elif response_type == ResponseType.MINORITY_CONSIDERATION:
                self._validate_minority_consideration(response, result, context)
            
            # Apply cleaning and improvements
            if result.is_valid:
                result.cleaned_response = self._clean_response(response, response_type)
            
            # Update statistics
            if result.is_valid:
                self.validation_stats["successful_validations"] += 1
            else:
                self.validation_stats["failed_validations"] += 1
                error_key = f"{response_type.value}_errors"
                self.validation_stats["validation_errors"][error_key] = \
                    self.validation_stats["validation_errors"].get(error_key, 0) + 1
            
            # Update response type statistics
            type_key = response_type.value
            if type_key not in self.validation_stats["response_type_stats"]:
                self.validation_stats["response_type_stats"][type_key] = {
                    "total": 0, "valid": 0, "invalid": 0
                }
            
            self.validation_stats["response_type_stats"][type_key]["total"] += 1
            if result.is_valid:
                self.validation_stats["response_type_stats"][type_key]["valid"] += 1
            else:
                self.validation_stats["response_type_stats"][type_key]["invalid"] += 1
            
            logger.debug(
                f"Validated {response_type.value} response "
                f"{'successfully' if result.is_valid else 'with issues'} "
                f"for agent {agent_id or 'unknown'}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            result.is_valid = False
            result.issues.append(f"Validation error: {str(e)}")
            self.validation_stats["failed_validations"] += 1
            return result
    
    def _validate_basic_format(self, response: str, result: ValidationResult) -> None:
        """Validate basic response format.
        
        Args:
            response: Response to validate
            result: ValidationResult to update
        """
        # Check if response is empty or whitespace
        if not response or not response.strip():
            result.is_valid = False
            result.issues.append("Response is empty")
            return
        
        # Check length constraints
        response_length = len(response.strip())
        if response_length < self.min_response_length:
            result.is_valid = False
            result.issues.append(
                f"Response too short ({response_length} chars, minimum {self.min_response_length})"
            )
        
        if response_length > self.max_response_length:
            if self.validation_level == ValidationLevel.STRICT:
                result.is_valid = False
                result.issues.append(
                    f"Response too long ({response_length} chars, maximum {self.max_response_length})"
                )
            else:
                result.warnings.append(
                    f"Response is long ({response_length} chars), consider truncating"
                )
        
        # Check for basic formatting issues
        if response.count('\n') > 20:
            result.warnings.append("Response has many line breaks")
        
        if len(response.split()) < 3:
            result.warnings.append("Response seems very brief")
        
        # Check for potentially malformed content
        if response.count('```') % 2 != 0:
            result.warnings.append("Unmatched code block markers")
        
        if response.count('"') % 2 != 0:
            result.warnings.append("Unmatched quotation marks")
    
    def _validate_discussion_response(
        self,
        response: str,
        result: ValidationResult,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Validate discussion response.
        
        Args:
            response: Response to validate
            result: ValidationResult to update
            context: Optional context with topic information
        """
        # Check for substantive content
        word_count = len(response.split())
        if word_count < 5:
            result.issues.append("Discussion response lacks substance")
            result.is_valid = False
        
        # Check for topic relevance (basic keyword matching if topic provided)
        if context and "topic" in context:
            topic = context["topic"].lower()
            response_lower = response.lower()
            
            # Extract key terms from topic
            topic_words = set(word for word in topic.split() if len(word) > 3)
            response_words = set(word for word in response_lower.split() if len(word) > 3)
            
            # Check for some overlap
            if topic_words and not topic_words.intersection(response_words):
                result.warnings.append("Response may not be directly related to the topic")
        
        # Check for respectful tone (basic checks)
        inappropriate_patterns = [
            r'\b(stupid|dumb|idiot|moron)\b',
            r'\b(shut up|shut it)\b',
            r'\b(you\'re wrong|you are wrong)\b(?!\s+because|\s+that|\s+to)',
        ]
        
        for pattern in inappropriate_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                result.warnings.append("Response may contain inappropriate language")
                break
        
        # Check for constructive elements
        constructive_indicators = [
            r'\b(however|although|but|while|whereas)\b',
            r'\b(consider|perhaps|maybe|might|could)\b',
            r'\b(builds?\s+on|expands?\s+on|adds?\s+to)\b',
            r'\b(I\s+think|I\s+believe|in\s+my\s+opinion)\b'
        ]
        
        has_constructive_elements = any(
            re.search(pattern, response, re.IGNORECASE)
            for pattern in constructive_indicators
        )
        
        if not has_constructive_elements and word_count > 20:
            result.warnings.append("Response could be more constructive or nuanced")
    
    def _validate_topic_proposal(
        self,
        response: str,
        result: ValidationResult,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Validate topic proposal response.
        
        Args:
            response: Response to validate
            result: ValidationResult to update
            context: Optional context
        """
        # Look for numbered list format
        numbered_lines = re.findall(r'^\d+\.\s*(.+)$', response, re.MULTILINE)
        bullet_lines = re.findall(r'^[-*•]\s*(.+)$', response, re.MULTILINE)
        
        total_topics = len(numbered_lines) + len(bullet_lines)
        
        if total_topics < 3:
            result.issues.append("Fewer than 3 topics proposed")
            result.is_valid = False
        elif total_topics > 5:
            result.warnings.append("More than 5 topics proposed, some may be ignored")
        
        # Check topic quality
        all_topics = numbered_lines + bullet_lines
        for i, topic in enumerate(all_topics):
            topic = topic.strip()
            if len(topic) < 5:
                result.warnings.append(f"Topic {i+1} seems too brief")
            if len(topic) > 200:
                result.warnings.append(f"Topic {i+1} is very long")
        
        # Check for duplicates
        unique_topics = set(topic.lower().strip() for topic in all_topics)
        if len(unique_topics) < len(all_topics):
            result.warnings.append("Some proposed topics may be duplicates")
        
        # Store parsed topics in metadata
        result.metadata["parsed_topics"] = all_topics
        result.metadata["topic_count"] = len(all_topics)
    
    def _validate_agenda_vote(
        self,
        response: str,
        result: ValidationResult,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Validate agenda vote response.
        
        Args:
            response: Response to validate
            result: ValidationResult to update
            context: Optional context with proposed topics
        """
        # Check for presence of reasoning
        if len(response.split()) < 10:
            result.warnings.append("Vote response lacks detailed reasoning")
        
        # Look for preference indicators
        preference_patterns = [
            r'\b(prefer|choose|select|priority|order)\b',
            r'\b(first|second|third|1st|2nd|3rd)\b',
            r'\b(most\s+important|least\s+important)\b',
            r'\b(should\s+be\s+discussed|discuss\s+first)\b'
        ]
        
        has_preferences = any(
            re.search(pattern, response, re.IGNORECASE)
            for pattern in preference_patterns
        )
        
        if not has_preferences:
            result.warnings.append("Vote response should express clear preferences")
        
        # Check for reasoning indicators
        reasoning_patterns = [
            r'\b(because|since|as|due\s+to|reason)\b',
            r'\b(important|relevant|urgent|timely)\b',
            r'\b(should|would|could|might)\b'
        ]
        
        has_reasoning = any(
            re.search(pattern, response, re.IGNORECASE)
            for pattern in reasoning_patterns
        )
        
        if not has_reasoning:
            result.warnings.append("Vote should include reasoning for preferences")
    
    def _validate_conclusion_vote(
        self,
        response: str,
        result: ValidationResult,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Validate conclusion vote response.
        
        Args:
            response: Response to validate
            result: ValidationResult to update
            context: Optional context
        """
        response_lower = response.lower().strip()
        
        # Check for clear Yes/No
        has_yes = 'yes' in response_lower
        has_no = 'no' in response_lower
        
        if not has_yes and not has_no:
            result.issues.append("Vote must contain 'Yes' or 'No'")
            result.is_valid = False
        elif has_yes and has_no:
            # Check if it's clearly one or the other
            if not (response_lower.startswith('yes') or response_lower.startswith('no')):
                result.warnings.append("Vote contains both 'Yes' and 'No', may be ambiguous")
        
        # Check for justification
        justification_patterns = [
            r'\b(because|since|as|reason|due\s+to)\b',
            r'\b(think|feel|believe|consider)\b',
            r'\b(adequate|sufficient|complete|incomplete)\b',
            r'\b(more\s+discussion|need\s+more|further\s+discussion)\b'
        ]
        
        has_justification = any(
            re.search(pattern, response, re.IGNORECASE)
            for pattern in justification_patterns
        )
        
        if not has_justification and len(response.split()) < 8:
            result.warnings.append("Vote should include justification")
        
        # Parse vote and reasoning for metadata
        vote_result = None
        if response_lower.startswith('yes'):
            vote_result = True
        elif response_lower.startswith('no'):
            vote_result = False
        elif 'yes' in response_lower.split()[:3]:
            vote_result = True
        elif 'no' in response_lower.split()[:3]:
            vote_result = False
        
        result.metadata["parsed_vote"] = vote_result
        result.metadata["has_justification"] = has_justification
    
    def _validate_minority_consideration(
        self,
        response: str,
        result: ValidationResult,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Validate minority consideration response.
        
        Args:
            response: Response to validate
            result: ValidationResult to update
            context: Optional context
        """
        word_count = len(response.split())
        
        # Should be substantive but not too long
        if word_count < 15:
            result.warnings.append("Minority consideration seems brief")
        elif word_count > 150:
            result.warnings.append("Minority consideration is quite long")
        
        # Check for appropriate content indicators
        consideration_patterns = [
            r'\b(concern|worry|issue|problem)\b',
            r'\b(overlooked|missed|ignored|important)\b',
            r'\b(should\s+consider|need\s+to|ought\s+to)\b',
            r'\b(final|last|concluding|closing)\b'
        ]
        
        has_appropriate_content = any(
            re.search(pattern, response, re.IGNORECASE)
            for pattern in consideration_patterns
        )
        
        if not has_appropriate_content:
            result.warnings.append(
                "Minority consideration should highlight concerns or overlooked aspects"
            )
    
    def _clean_response(self, response: str, response_type: ResponseType) -> str:
        """Clean and improve response formatting.
        
        Args:
            response: Original response
            response_type: Type of response
            
        Returns:
            Cleaned response
        """
        cleaned = response.strip()
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = re.sub(r' +', ' ', cleaned)
        
        # Fix common formatting issues
        cleaned = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', cleaned)
        
        # Truncate if too long (for non-strict validation)
        if len(cleaned) > self.max_response_length and self.validation_level != ValidationLevel.STRICT:
            cleaned = cleaned[:self.max_response_length-3] + "..."
        
        return cleaned
    
    def validate_with_retry(
        self,
        response: str,
        response_type: ResponseType,
        retry_callback,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None
    ) -> ValidationResult:
        """Validate response with retry mechanism.
        
        Args:
            response: Response to validate
            response_type: Type of response
            retry_callback: Function to call for retry (should return new response)
            context: Optional context
            agent_id: Optional agent ID
            
        Returns:
            Final validation result
        """
        result = self.validate_response(response, response_type, context, agent_id)
        
        retry_count = 0
        while not result.is_valid and retry_count < self.max_retries:
            logger.info(f"Retrying validation for {agent_id}, attempt {retry_count + 1}")
            
            try:
                # Call retry callback to get new response
                new_response = retry_callback(result.issues, result.warnings)
                
                if new_response and new_response != response:
                    result = self.validate_response(
                        new_response, response_type, context, agent_id
                    )
                    self.validation_stats["retries_attempted"] += 1
                    retry_count += 1
                else:
                    break  # No new response or same response
                    
            except Exception as e:
                logger.error(f"Error during retry callback: {e}")
                break
        
        return result
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics.
        
        Returns:
            Dictionary with validation statistics
        """
        success_rate = 0.0
        if self.validation_stats["total_validations"] > 0:
            success_rate = (
                self.validation_stats["successful_validations"] /
                self.validation_stats["total_validations"]
            )
        
        return {
            "total_validations": self.validation_stats["total_validations"],
            "successful_validations": self.validation_stats["successful_validations"],
            "failed_validations": self.validation_stats["failed_validations"],
            "success_rate": success_rate,
            "retries_attempted": self.validation_stats["retries_attempted"],
            "validation_errors": self.validation_stats["validation_errors"],
            "response_type_stats": self.validation_stats["response_type_stats"],
            "validation_level": self.validation_level.value,
            "max_response_length": self.max_response_length,
            "min_response_length": self.min_response_length
        }
    
    def reset_statistics(self) -> None:
        """Reset validation statistics."""
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "retries_attempted": 0,
            "validation_errors": {},
            "response_type_stats": {}
        }
        logger.info("Validation statistics reset")


class ValidationMiddleware:
    """Middleware wrapper for automatic response validation."""
    
    def __init__(
        self,
        validator: ResponseValidator,
        enable_auto_retry: bool = True,
        log_validation_results: bool = True
    ):
        """Initialize validation middleware.
        
        Args:
            validator: ResponseValidator instance
            enable_auto_retry: Whether to enable automatic retries
            log_validation_results: Whether to log validation results
        """
        self.validator = validator
        self.enable_auto_retry = enable_auto_retry
        self.log_validation_results = log_validation_results
        
    def validate_agent_response(
        self,
        agent,
        response_method: str,
        response_type: ResponseType,
        *args,
        **kwargs
    ) -> Tuple[str, ValidationResult]:
        """Validate an agent response with middleware.
        
        Args:
            agent: Agent instance
            response_method: Name of the method to call
            response_type: Type of response expected
            *args: Arguments for the response method
            **kwargs: Keyword arguments for the response method
            
        Returns:
            Tuple of (response, validation_result)
        """
        # Get the response from the agent
        method = getattr(agent, response_method)
        response = method(*args, **kwargs)
        
        # Validate the response
        context = kwargs.get('context', {})
        context.update(kwargs)  # Include all kwargs as context
        
        if self.enable_auto_retry:
            # Create retry callback
            def retry_callback(issues, warnings):
                logger.info(f"Retrying {response_method} for agent {agent.agent_id}")
                return method(*args, **kwargs)
            
            result = self.validator.validate_with_retry(
                response, response_type, retry_callback, context, agent.agent_id
            )
        else:
            result = self.validator.validate_response(
                response, response_type, context, agent.agent_id
            )
        
        if self.log_validation_results:
            if result.is_valid:
                logger.debug(f"Validation passed for {agent.agent_id} {response_type.value}")
            else:
                logger.warning(
                    f"Validation failed for {agent.agent_id} {response_type.value}: "
                    f"{', '.join(result.issues)}"
                )
        
        # Return cleaned response if valid, original if not
        final_response = result.cleaned_response if result.is_valid else response
        return final_response, result