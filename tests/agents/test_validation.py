"""Tests for agent response validation system."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any, List

from virtual_agora.agents.validation import (
    ValidationLevel,
    ResponseType,
    ValidationResult,
    ResponseValidator,
    ValidationMiddleware,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_initialization(self):
        """Test ValidationResult initialization."""
        result = ValidationResult(
            is_valid=True,
            response_type=ResponseType.DISCUSSION,
            original_response="Test response",
        )

        assert result.is_valid is True
        assert result.response_type == ResponseType.DISCUSSION
        assert result.original_response == "Test response"
        assert result.cleaned_response is None
        assert result.issues == []
        assert result.warnings == []
        assert result.metadata == {}
        assert isinstance(result.validation_time, datetime)

    def test_post_init_defaults(self):
        """Test that post_init sets up default values."""
        result = ValidationResult(
            is_valid=False,
            response_type=ResponseType.TOPIC_PROPOSAL,
            original_response="Test",
        )

        # Should initialize empty lists and dict
        assert isinstance(result.issues, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.metadata, dict)
        assert isinstance(result.validation_time, datetime)

    def test_with_data(self):
        """Test ValidationResult with provided data."""
        issues = ["Issue 1", "Issue 2"]
        warnings = ["Warning 1"]
        metadata = {"key": "value"}

        result = ValidationResult(
            is_valid=False,
            response_type=ResponseType.AGENDA_VOTE,
            original_response="Test response",
            cleaned_response="Cleaned response",
            issues=issues,
            warnings=warnings,
            metadata=metadata,
        )

        assert result.issues == issues
        assert result.warnings == warnings
        assert result.metadata == metadata
        assert result.cleaned_response == "Cleaned response"


class TestResponseValidator:
    """Test ResponseValidator class."""

    def setup_method(self):
        """Set up test method."""
        self.validator = ResponseValidator(
            validation_level=ValidationLevel.STANDARD,
            max_response_length=1000,
            min_response_length=5,
            max_retries=2,
        )

    def test_initialization(self):
        """Test validator initialization."""
        assert self.validator.validation_level == ValidationLevel.STANDARD
        assert self.validator.max_response_length == 1000
        assert self.validator.min_response_length == 5
        assert self.validator.max_retries == 2

        # Check statistics initialization
        stats = self.validator.validation_stats
        assert stats["total_validations"] == 0
        assert stats["successful_validations"] == 0
        assert stats["failed_validations"] == 0
        assert stats["retries_attempted"] == 0
        assert isinstance(stats["validation_errors"], dict)
        assert isinstance(stats["response_type_stats"], dict)

    def test_validate_basic_format_success(self):
        """Test successful basic format validation."""
        response = "This is a valid response with good length."
        result = ValidationResult(True, ResponseType.DISCUSSION, response)

        self.validator._validate_basic_format(response, result)

        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_validate_basic_format_empty_response(self):
        """Test validation of empty response."""
        response = ""
        result = ValidationResult(True, ResponseType.DISCUSSION, response)

        self.validator._validate_basic_format(response, result)

        assert result.is_valid is False
        assert "Response is empty" in result.issues

    def test_validate_basic_format_too_short(self):
        """Test validation of too short response."""
        response = "Hi"  # Only 2 chars, minimum is 5
        result = ValidationResult(True, ResponseType.DISCUSSION, response)

        self.validator._validate_basic_format(response, result)

        assert result.is_valid is False
        assert any("too short" in issue for issue in result.issues)

    def test_validate_basic_format_too_long_strict(self):
        """Test validation of too long response in strict mode."""
        validator = ResponseValidator(
            validation_level=ValidationLevel.STRICT, max_response_length=10
        )

        response = "This response is definitely too long for the limit"
        result = ValidationResult(True, ResponseType.DISCUSSION, response)

        validator._validate_basic_format(response, result)

        assert result.is_valid is False
        assert any("too long" in issue for issue in result.issues)

    def test_validate_basic_format_too_long_standard(self):
        """Test validation of too long response in standard mode."""
        validator = ResponseValidator(
            validation_level=ValidationLevel.STANDARD, max_response_length=10
        )

        response = "This response is definitely too long for the limit"
        result = ValidationResult(True, ResponseType.DISCUSSION, response)

        validator._validate_basic_format(response, result)

        assert result.is_valid is True  # Should still be valid
        assert any("is long" in warning for warning in result.warnings)

    def test_validate_basic_format_warnings(self):
        """Test basic format validation warnings."""
        response = "Line1\nLine2\nLine3\nLine4\nLine5\nLine6\nLine7\nLine8\nLine9\nLine10\nLine11\nLine12\nLine13\nLine14\nLine15\nLine16\nLine17\nLine18\nLine19\nLine20\nLine21\nLine22"
        result = ValidationResult(True, ResponseType.DISCUSSION, response)

        self.validator._validate_basic_format(response, result)

        assert result.is_valid is True
        assert any("many line breaks" in warning for warning in result.warnings)

    def test_validate_basic_format_malformed_content(self):
        """Test detection of malformed content."""
        # Unmatched code blocks
        response = "Here is some code: ```python code here"
        result = ValidationResult(True, ResponseType.DISCUSSION, response)

        self.validator._validate_basic_format(response, result)

        assert any("Unmatched code block" in warning for warning in result.warnings)

        # Unmatched quotes
        response2 = 'He said "hello world and left'
        result2 = ValidationResult(True, ResponseType.DISCUSSION, response2)

        self.validator._validate_basic_format(response2, result2)

        assert any("Unmatched quotation" in warning for warning in result2.warnings)


class TestResponseValidatorDiscussion:
    """Test discussion response validation."""

    def setup_method(self):
        """Set up test method."""
        self.validator = ResponseValidator()

    def test_validate_discussion_response_success(self):
        """Test successful discussion response validation."""
        response = "This is a thoughtful discussion response with good substance."
        result = ValidationResult(True, ResponseType.DISCUSSION, response)

        self.validator._validate_discussion_response(response, result)

        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_validate_discussion_response_too_brief(self):
        """Test validation of too brief discussion response."""
        response = "Yes"  # Only 1 word, less than 5
        result = ValidationResult(True, ResponseType.DISCUSSION, response)

        self.validator._validate_discussion_response(response, result)

        assert result.is_valid is False
        assert any("lacks substance" in issue for issue in result.issues)

    def test_validate_discussion_response_topic_relevance(self):
        """Test topic relevance checking."""
        response = "I really enjoy talking about cars and sports today."
        context = {"topic": "artificial intelligence ethics"}
        result = ValidationResult(True, ResponseType.DISCUSSION, response)

        self.validator._validate_discussion_response(response, result, context)

        assert any("not be directly related" in warning for warning in result.warnings)

    def test_validate_discussion_response_topic_relevant(self):
        """Test topic relevance with relevant response."""
        response = "Artificial intelligence ethics requires careful consideration of bias and fairness."
        context = {"topic": "artificial intelligence ethics"}
        result = ValidationResult(True, ResponseType.DISCUSSION, response)

        self.validator._validate_discussion_response(response, result, context)

        # Should not have topic relevance warning
        assert not any(
            "not be directly related" in warning for warning in result.warnings
        )

    def test_validate_discussion_response_inappropriate_language(self):
        """Test detection of inappropriate language."""
        response = "That idea is really stupid and makes no sense at all."
        result = ValidationResult(True, ResponseType.DISCUSSION, response)

        self.validator._validate_discussion_response(response, result)

        assert any("inappropriate language" in warning for warning in result.warnings)

    def test_validate_discussion_response_constructive_elements(self):
        """Test detection of constructive elements."""
        # Response without constructive elements
        response = "The solution is wrong. The approach fails. The method breaks everything completely. This implementation is terrible and will not work in any scenario. The whole design is flawed from the beginning."
        result = ValidationResult(True, ResponseType.DISCUSSION, response)

        self.validator._validate_discussion_response(response, result)

        assert any("more constructive" in warning for warning in result.warnings)

        # Response with constructive elements
        response2 = "While I understand the approach, perhaps we could consider an alternative method."
        result2 = ValidationResult(True, ResponseType.DISCUSSION, response2)

        self.validator._validate_discussion_response(response2, result2)

        # Should not have constructive warning
        assert not any("more constructive" in warning for warning in result2.warnings)


class TestResponseValidatorTopicProposal:
    """Test topic proposal validation."""

    def setup_method(self):
        """Set up test method."""
        self.validator = ResponseValidator()

    def test_validate_topic_proposal_success(self):
        """Test successful topic proposal validation."""
        response = """1. Ethics in artificial intelligence development
2. Safety measures for autonomous systems  
3. Regulatory frameworks for AI governance
4. Bias and fairness in machine learning algorithms"""

        result = ValidationResult(True, ResponseType.TOPIC_PROPOSAL, response)

        self.validator._validate_topic_proposal(response, result)

        assert result.is_valid is True
        assert len(result.issues) == 0
        assert result.metadata["topic_count"] == 4
        assert len(result.metadata["parsed_topics"]) == 4

    def test_validate_topic_proposal_too_few(self):
        """Test topic proposal with too few topics."""
        response = """1. Ethics in AI
2. Safety measures"""

        result = ValidationResult(True, ResponseType.TOPIC_PROPOSAL, response)

        self.validator._validate_topic_proposal(response, result)

        assert result.is_valid is False
        assert any("Fewer than 3 topics" in issue for issue in result.issues)

    def test_validate_topic_proposal_too_many(self):
        """Test topic proposal with too many topics."""
        response = """1. Topic one
2. Topic two
3. Topic three
4. Topic four
5. Topic five
6. Topic six
7. Topic seven"""

        result = ValidationResult(True, ResponseType.TOPIC_PROPOSAL, response)

        self.validator._validate_topic_proposal(response, result)

        assert result.is_valid is True  # Still valid
        assert any("More than 5 topics" in warning for warning in result.warnings)

    def test_validate_topic_proposal_bullet_format(self):
        """Test topic proposal with bullet format."""
        response = """- Ethics in artificial intelligence
* Safety in autonomous systems
• Governance and regulation"""

        result = ValidationResult(True, ResponseType.TOPIC_PROPOSAL, response)

        self.validator._validate_topic_proposal(response, result)

        assert result.is_valid is True
        assert result.metadata["topic_count"] == 3

    def test_validate_topic_proposal_quality_checks(self):
        """Test topic proposal quality checks."""
        response = """1. AI
2. This is a very long topic that goes on and on and provides way too much detail that could be considered excessive for a simple topic proposal and might overwhelm participants with its verbosity and unnecessary elaboration on every minor detail and consideration
3. Machine learning ethics and fairness"""

        result = ValidationResult(True, ResponseType.TOPIC_PROPOSAL, response)

        self.validator._validate_topic_proposal(response, result)

        assert any("seems too brief" in warning for warning in result.warnings)
        assert any("is very long" in warning for warning in result.warnings)

    def test_validate_topic_proposal_duplicates(self):
        """Test detection of duplicate topics."""
        response = """1. AI Ethics
2. Machine Learning Ethics  
3. AI Ethics
4. Governance frameworks"""

        result = ValidationResult(True, ResponseType.TOPIC_PROPOSAL, response)

        self.validator._validate_topic_proposal(response, result)

        assert any("duplicates" in warning for warning in result.warnings)


class TestResponseValidatorAgendaVote:
    """Test agenda vote validation."""

    def setup_method(self):
        """Set up test method."""
        self.validator = ResponseValidator()

    def test_validate_agenda_vote_success(self):
        """Test successful agenda vote validation."""
        response = "I prefer topics in the order 1, 3, 2 because the first topic is most important for our discussion."
        result = ValidationResult(True, ResponseType.AGENDA_VOTE, response)

        self.validator._validate_agenda_vote(response, result)

        assert result.is_valid is True
        assert len(result.issues) == 0
        assert len(result.warnings) == 0

    def test_validate_agenda_vote_lacks_reasoning(self):
        """Test agenda vote that lacks detailed reasoning."""
        response = "Topic 1 first."
        result = ValidationResult(True, ResponseType.AGENDA_VOTE, response)

        self.validator._validate_agenda_vote(response, result)

        assert any("lacks detailed reasoning" in warning for warning in result.warnings)

    def test_validate_agenda_vote_no_preferences(self):
        """Test agenda vote without clear preferences."""
        response = "All topics seem fine and good for discussion today."
        result = ValidationResult(True, ResponseType.AGENDA_VOTE, response)

        self.validator._validate_agenda_vote(response, result)

        assert any(
            "express clear preferences" in warning for warning in result.warnings
        )

    def test_validate_agenda_vote_no_reasoning(self):
        """Test agenda vote without reasoning."""
        response = "I prefer topic 1 first, then topic 2."
        result = ValidationResult(True, ResponseType.AGENDA_VOTE, response)

        self.validator._validate_agenda_vote(response, result)

        assert any("include reasoning" in warning for warning in result.warnings)


class TestResponseValidatorConclusionVote:
    """Test conclusion vote validation."""

    def setup_method(self):
        """Set up test method."""
        self.validator = ResponseValidator()

    def test_validate_conclusion_vote_clear_yes(self):
        """Test clear yes conclusion vote."""
        response = (
            "Yes, I think we have adequately covered the main aspects of this topic."
        )
        result = ValidationResult(True, ResponseType.CONCLUSION_VOTE, response)

        self.validator._validate_conclusion_vote(response, result)

        assert result.is_valid is True
        assert result.metadata["parsed_vote"] is True
        assert result.metadata["has_justification"] is True

    def test_validate_conclusion_vote_clear_no(self):
        """Test clear no conclusion vote."""
        response = "No, we need more discussion on the ethical implications."
        result = ValidationResult(True, ResponseType.CONCLUSION_VOTE, response)

        self.validator._validate_conclusion_vote(response, result)

        assert result.is_valid is True
        assert result.metadata["parsed_vote"] is False

    def test_validate_conclusion_vote_missing_vote(self):
        """Test conclusion vote missing yes/no."""
        response = "I think the discussion has been quite comprehensive overall."
        result = ValidationResult(True, ResponseType.CONCLUSION_VOTE, response)

        self.validator._validate_conclusion_vote(response, result)

        assert result.is_valid is False
        assert any("must contain 'Yes' or 'No'" in issue for issue in result.issues)

    def test_validate_conclusion_vote_ambiguous(self):
        """Test ambiguous conclusion vote."""
        response = "It's been good (yes), but no, we could discuss more."
        result = ValidationResult(True, ResponseType.CONCLUSION_VOTE, response)

        self.validator._validate_conclusion_vote(response, result)

        assert any("may be ambiguous" in warning for warning in result.warnings)

    def test_validate_conclusion_vote_lacks_justification(self):
        """Test conclusion vote lacking justification."""
        response = "Yes."
        result = ValidationResult(True, ResponseType.CONCLUSION_VOTE, response)

        self.validator._validate_conclusion_vote(response, result)

        assert any(
            "should include justification" in warning for warning in result.warnings
        )


class TestResponseValidatorMinorityConsideration:
    """Test minority consideration validation."""

    def setup_method(self):
        """Set up test method."""
        self.validator = ResponseValidator()

    def test_validate_minority_consideration_success(self):
        """Test successful minority consideration validation."""
        response = "I believe there's an important concern about privacy that we haven't fully addressed in our discussion."
        result = ValidationResult(True, ResponseType.MINORITY_CONSIDERATION, response)

        self.validator._validate_minority_consideration(response, result)

        assert result.is_valid is True
        assert len(result.warnings) == 0

    def test_validate_minority_consideration_too_brief(self):
        """Test minority consideration that's too brief."""
        response = "Privacy concerns remain."
        result = ValidationResult(True, ResponseType.MINORITY_CONSIDERATION, response)

        self.validator._validate_minority_consideration(response, result)

        assert any("seems brief" in warning for warning in result.warnings)

    def test_validate_minority_consideration_too_long(self):
        """Test minority consideration that's too long."""
        response = " ".join(["Long consideration"] * 80)  # Very long response
        result = ValidationResult(True, ResponseType.MINORITY_CONSIDERATION, response)

        self.validator._validate_minority_consideration(response, result)

        assert any("quite long" in warning for warning in result.warnings)

    def test_validate_minority_consideration_lacks_appropriate_content(self):
        """Test minority consideration lacking appropriate content."""
        response = "Thank you for the good discussion today everyone."
        result = ValidationResult(True, ResponseType.MINORITY_CONSIDERATION, response)

        self.validator._validate_minority_consideration(response, result)

        assert any("highlight concerns" in warning for warning in result.warnings)


class TestResponseValidatorIntegration:
    """Test ResponseValidator integration and full validation."""

    def setup_method(self):
        """Set up test method."""
        self.validator = ResponseValidator(validation_level=ValidationLevel.STANDARD)

    def test_validate_response_complete_flow(self):
        """Test complete validation flow."""
        response = "This is a well-reasoned discussion response that addresses the topic thoughtfully."

        result = self.validator.validate_response(
            response,
            ResponseType.DISCUSSION,
            context={"topic": "AI ethics"},
            agent_id="test-agent",
        )

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.response_type == ResponseType.DISCUSSION
        assert result.original_response == response
        assert result.cleaned_response is not None
        assert isinstance(result.validation_time, datetime)

    def test_validate_response_with_cleaning(self):
        """Test response validation with cleaning."""
        response = "  This has   extra    spaces  and \n\n\n multiple lines.  "

        result = self.validator.validate_response(response, ResponseType.DISCUSSION)

        assert result.is_valid is True
        assert result.cleaned_response != response
        assert "extra    spaces" not in result.cleaned_response
        assert "\n\n\n" not in result.cleaned_response

    def test_validate_response_error_handling(self):
        """Test error handling in validation."""
        with patch.object(
            self.validator,
            "_validate_basic_format",
            side_effect=Exception("Test error"),
        ):
            result = self.validator.validate_response("Test", ResponseType.DISCUSSION)

            assert result.is_valid is False
            assert any("Validation error" in issue for issue in result.issues)

    def test_validate_with_retry_success_first_try(self):
        """Test validation with retry when first attempt succeeds."""
        response = "Good response that passes validation"

        def mock_retry_callback(issues, warnings):
            return "Retry response"

        result = self.validator.validate_with_retry(
            response,
            ResponseType.DISCUSSION,
            mock_retry_callback,
            agent_id="test-agent",
        )

        assert result.is_valid is True
        assert self.validator.validation_stats["retries_attempted"] == 0

    def test_validate_with_retry_needs_retry(self):
        """Test validation with retry when retry is needed."""
        bad_response = "Bad"  # Too short
        good_response = "This is a good response that passes validation"

        def mock_retry_callback(issues, warnings):
            return good_response

        result = self.validator.validate_with_retry(
            bad_response,
            ResponseType.DISCUSSION,
            mock_retry_callback,
            agent_id="test-agent",
        )

        assert result.is_valid is True
        assert result.original_response == good_response  # Should be the retry response
        assert self.validator.validation_stats["retries_attempted"] == 1

    def test_validate_with_retry_callback_error(self):
        """Test validation with retry when callback raises error."""
        bad_response = "Bad"

        def failing_callback(issues, warnings):
            raise Exception("Callback failed")

        result = self.validator.validate_with_retry(
            bad_response,
            ResponseType.DISCUSSION,
            failing_callback,
            agent_id="test-agent",
        )

        assert result.is_valid is False
        assert self.validator.validation_stats["retries_attempted"] == 0

    def test_get_validation_statistics(self):
        """Test getting validation statistics."""
        # Perform some validations
        self.validator.validate_response(
            "This is a good and substantive response", ResponseType.DISCUSSION
        )
        self.validator.validate_response(
            "Bad", ResponseType.TOPIC_PROPOSAL
        )  # Should fail

        stats = self.validator.get_validation_statistics()

        assert stats["total_validations"] == 2
        assert stats["successful_validations"] == 1
        assert stats["failed_validations"] == 1
        assert stats["success_rate"] == 0.5
        assert "response_type_stats" in stats
        assert "validation_level" in stats

    def test_reset_statistics(self):
        """Test resetting validation statistics."""
        # Perform validation to change stats
        self.validator.validate_response("Test", ResponseType.DISCUSSION)

        # Reset
        self.validator.reset_statistics()

        stats = self.validator.get_validation_statistics()
        assert stats["total_validations"] == 0
        assert stats["successful_validations"] == 0
        assert stats["failed_validations"] == 0


class TestValidationMiddleware:
    """Test ValidationMiddleware class."""

    def setup_method(self):
        """Set up test method."""
        self.validator = ResponseValidator()
        self.middleware = ValidationMiddleware(
            self.validator, enable_auto_retry=True, log_validation_results=True
        )

    def test_middleware_initialization(self):
        """Test middleware initialization."""
        assert self.middleware.validator == self.validator
        assert self.middleware.enable_auto_retry is True
        assert self.middleware.log_validation_results is True

    def test_validate_agent_response_success(self):
        """Test successful agent response validation through middleware."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "test-agent"
        mock_agent.generate_discussion_response.return_value = (
            "This is a good and thoughtful discussion response"
        )

        response, result = self.middleware.validate_agent_response(
            mock_agent,
            "generate_discussion_response",
            ResponseType.DISCUSSION,
            "Test topic",
        )

        assert response == "This is a good and thoughtful discussion response"
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        mock_agent.generate_discussion_response.assert_called_once_with("Test topic")

    def test_validate_agent_response_with_retry(self):
        """Test agent response validation with retry."""
        # Mock agent that fails first time, succeeds second time
        mock_agent = Mock()
        mock_agent.agent_id = "test-agent"
        mock_agent.generate_discussion_response.side_effect = [
            "Bad",  # Too short, will fail validation
            "This is a good response that passes validation",
        ]

        response, result = self.middleware.validate_agent_response(
            mock_agent,
            "generate_discussion_response",
            ResponseType.DISCUSSION,
            "Test topic",
        )

        assert result.is_valid is True
        assert mock_agent.generate_discussion_response.call_count == 2

    def test_validate_agent_response_no_retry(self):
        """Test agent response validation without retry."""
        middleware = ValidationMiddleware(self.validator, enable_auto_retry=False)

        mock_agent = Mock()
        mock_agent.agent_id = "test-agent"
        mock_agent.generate_discussion_response.return_value = "Bad"  # Too short

        response, result = middleware.validate_agent_response(
            mock_agent,
            "generate_discussion_response",
            ResponseType.DISCUSSION,
            "Test topic",
        )

        assert result.is_valid is False
        assert mock_agent.generate_discussion_response.call_count == 1

    def test_validate_agent_response_with_context(self):
        """Test agent response validation with context."""
        mock_agent = Mock()
        mock_agent.agent_id = "test-agent"
        mock_agent.vote_on_agenda.return_value = "I prefer topics 1, 2, 3"

        response, result = self.middleware.validate_agent_response(
            mock_agent,
            "vote_on_agenda",
            ResponseType.AGENDA_VOTE,
            ["Topic A", "Topic B"],
            context={"discussion_phase": "agenda"},
        )

        assert response == "I prefer topics 1, 2, 3"
        assert result.is_valid is True


class TestValidationIntegration:
    """Integration tests for validation system."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        validator = ResponseValidator(
            validation_level=ValidationLevel.STANDARD,
            max_response_length=500,
            min_response_length=10,
        )

        # Test different response types
        test_cases = [
            (
                ResponseType.DISCUSSION,
                "This is a thoughtful discussion response about the important topic.",
                True,
            ),
            (
                ResponseType.TOPIC_PROPOSAL,
                "1. First topic\n2. Second topic\n3. Third topic",
                True,
            ),
            (
                ResponseType.AGENDA_VOTE,
                "I prefer topics 1, 3, 2 because the first is most urgent.",
                True,
            ),
            (
                ResponseType.CONCLUSION_VOTE,
                "Yes, we have covered the main points adequately.",
                True,
            ),
            (
                ResponseType.MINORITY_CONSIDERATION,
                "I believe we should consider the privacy implications more carefully.",
                True,
            ),
        ]

        for response_type, response_text, expected_valid in test_cases:
            result = validator.validate_response(response_text, response_type)

            assert result.is_valid == expected_valid
            assert result.response_type == response_type
            assert result.cleaned_response is not None
            assert isinstance(result.validation_time, datetime)

        # Check final statistics
        stats = validator.get_validation_statistics()
        assert stats["total_validations"] == len(test_cases)
        assert stats["successful_validations"] == sum(
            1 for _, _, valid in test_cases if valid
        )

    def test_validation_with_different_levels(self):
        """Test validation with different strictness levels."""
        response = (
            "This is a moderately long response" + " that goes on" * 50
        )  # Long response

        # Permissive level
        permissive_validator = ResponseValidator(
            validation_level=ValidationLevel.PERMISSIVE, max_response_length=100
        )
        result_permissive = permissive_validator.validate_response(
            response, ResponseType.DISCUSSION
        )

        # Standard level
        standard_validator = ResponseValidator(
            validation_level=ValidationLevel.STANDARD, max_response_length=100
        )
        result_standard = standard_validator.validate_response(
            response, ResponseType.DISCUSSION
        )

        # Strict level
        strict_validator = ResponseValidator(
            validation_level=ValidationLevel.STRICT, max_response_length=100
        )
        result_strict = strict_validator.validate_response(
            response, ResponseType.DISCUSSION
        )

        # All should handle the long response differently
        assert result_permissive.is_valid is True
        assert result_standard.is_valid is True
        assert result_strict.is_valid is False

        # Check that strict mode generates an error for length
        assert any("too long" in issue for issue in result_strict.issues)

    def test_response_cleaning_functionality(self):
        """Test response cleaning across different response types."""
        validator = ResponseValidator()

        # Test various cleaning scenarios
        messy_response = (
            "  This   has    extra spaces   \n\n\n\n and multiple newlines.  "
        )

        result = validator.validate_response(messy_response, ResponseType.DISCUSSION)

        assert result.is_valid is True
        assert result.cleaned_response != messy_response
        assert "   " not in result.cleaned_response  # Multiple spaces should be cleaned
        assert (
            result.cleaned_response.strip() == result.cleaned_response
        )  # Should be trimmed
