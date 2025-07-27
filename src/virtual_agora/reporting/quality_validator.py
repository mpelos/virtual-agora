"""Report quality validation for Virtual Agora.

This module provides functionality to validate report quality,
including Markdown syntax, completeness, and readability.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import statistics

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ReportQualityValidator:
    """Validate quality of generated reports."""

    # Readability thresholds
    MAX_SENTENCE_LENGTH = 30  # words
    MAX_PARAGRAPH_LENGTH = 150  # words
    MIN_PARAGRAPH_LENGTH = 20  # words
    MAX_HEADING_LEVEL = 4

    # Common Markdown issues to check
    MARKDOWN_PATTERNS = {
        "unclosed_bold": r"\*\*[^*]+$",
        "unclosed_italic": r"\*[^*]+$",
        "unclosed_code": r"`[^`]+$",
        "malformed_link": r"\[[^\]]+\](?!\()",
        "empty_heading": r"^#+\s*$",
        "invalid_list": r"^[0-9]+[^.]",  # Number without period
        "excessive_blank_lines": r"\n{4,}",
    }

    def __init__(self):
        """Initialize ReportQualityValidator."""
        self.validation_results = {}

    def validate_report(
        self,
        report_dir: Path,
        check_completeness: bool = True,
        check_readability: bool = True,
        check_markdown: bool = True,
    ) -> Dict[str, Any]:
        """Validate a complete report directory.

        Args:
            report_dir: Path to report directory.
            check_completeness: Whether to check completeness.
            check_readability: Whether to check readability.
            check_markdown: Whether to validate Markdown syntax.

        Returns:
            Validation results dictionary.
        """
        results = {
            "valid": True,
            "report_dir": str(report_dir),
            "issues": [],
            "warnings": [],
            "file_validations": {},
            "overall_score": 100.0,
        }

        try:
            # Check directory exists
            if not report_dir.exists():
                results["valid"] = False
                results["issues"].append("Report directory does not exist")
                results["overall_score"] = 0.0
                return results

            # Get all Markdown files
            md_files = list(report_dir.glob("*.md"))

            if not md_files:
                results["valid"] = False
                results["issues"].append("No Markdown files found in report directory")
                results["overall_score"] = 0.0
                return results

            # Validate each file
            file_scores = []
            for md_file in md_files:
                file_results = self.validate_file(
                    md_file,
                    check_readability=check_readability,
                    check_markdown=check_markdown,
                )
                results["file_validations"][md_file.name] = file_results
                file_scores.append(file_results["score"])

                # Aggregate issues
                if not file_results["valid"]:
                    results["valid"] = False
                results["issues"].extend(
                    [f"{md_file.name}: {issue}" for issue in file_results["issues"]]
                )
                results["warnings"].extend(
                    [
                        f"{md_file.name}: {warning}"
                        for warning in file_results["warnings"]
                    ]
                )

            # Check completeness
            if check_completeness:
                completeness_results = self._check_report_completeness(report_dir)
                results["completeness"] = completeness_results
                if not completeness_results["complete"]:
                    results["valid"] = False
                    results["issues"].extend(completeness_results["missing"])

            # Calculate overall score
            if file_scores:
                results["overall_score"] = round(statistics.mean(file_scores), 2)

            # Add summary
            results["summary"] = self._generate_validation_summary(results)

            self.validation_results = results
            return results

        except Exception as e:
            logger.error(f"Error validating report: {e}")
            results["valid"] = False
            results["issues"].append(f"Validation error: {str(e)}")
            results["overall_score"] = 0.0
            return results

    def validate_file(
        self,
        file_path: Path,
        check_readability: bool = True,
        check_markdown: bool = True,
    ) -> Dict[str, Any]:
        """Validate a single Markdown file.

        Args:
            file_path: Path to Markdown file.
            check_readability: Whether to check readability.
            check_markdown: Whether to validate Markdown syntax.

        Returns:
            File validation results.
        """
        results = {
            "valid": True,
            "file": file_path.name,
            "issues": [],
            "warnings": [],
            "score": 100.0,
            "metrics": {},
        }

        try:
            content = file_path.read_text(encoding="utf-8")

            # Check for empty file
            if not content.strip():
                results["valid"] = False
                results["issues"].append("File is empty")
                results["score"] = 0.0
                return results

            # Validate Markdown syntax
            if check_markdown:
                markdown_results = self._validate_markdown_syntax(content)
                results["markdown_validation"] = markdown_results
                if markdown_results["issues"]:
                    results["issues"].extend(markdown_results["issues"])
                    results["score"] -= len(markdown_results["issues"]) * 5
                if markdown_results["warnings"]:
                    results["warnings"].extend(markdown_results["warnings"])
                    results["score"] -= len(markdown_results["warnings"]) * 2

            # Check readability
            if check_readability:
                readability_results = self._check_readability(content)
                results["readability"] = readability_results
                if readability_results["issues"]:
                    results["warnings"].extend(readability_results["issues"])
                    results["score"] -= len(readability_results["issues"]) * 3
                results["metrics"].update(readability_results["metrics"])

            # Check structure
            structure_results = self._validate_structure(content)
            results["structure"] = structure_results
            if structure_results["issues"]:
                results["warnings"].extend(structure_results["issues"])
                results["score"] -= len(structure_results["issues"]) * 2

            # Ensure score doesn't go below 0
            results["score"] = max(0, results["score"])

            # Mark as invalid if score is too low
            if results["score"] < 50:
                results["valid"] = False

            return results

        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            results["valid"] = False
            results["issues"].append(f"Validation error: {str(e)}")
            results["score"] = 0.0
            return results

    def _validate_markdown_syntax(self, content: str) -> Dict[str, Any]:
        """Validate Markdown syntax.

        Args:
            content: Markdown content.

        Returns:
            Markdown validation results.
        """
        results = {
            "issues": [],
            "warnings": [],
            "syntax_errors": {},
        }

        lines = content.split("\n")

        # Check for common Markdown issues
        for pattern_name, pattern in self.MARKDOWN_PATTERNS.items():
            matches = []
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    matches.append(i)

            if matches:
                results["syntax_errors"][pattern_name] = matches
                if pattern_name in [
                    "unclosed_bold",
                    "unclosed_italic",
                    "unclosed_code",
                ]:
                    results["issues"].append(
                        f"{pattern_name.replace('_', ' ').title()} on lines: {matches[:5]}"
                    )
                else:
                    results["warnings"].append(
                        f"{pattern_name.replace('_', ' ').title()} on lines: {matches[:5]}"
                    )

        # Check heading hierarchy
        heading_issues = self._check_heading_hierarchy(lines)
        if heading_issues:
            results["warnings"].extend(heading_issues)

        # Check list formatting
        list_issues = self._check_list_formatting(lines)
        if list_issues:
            results["warnings"].extend(list_issues)

        return results

    def _check_heading_hierarchy(self, lines: List[str]) -> List[str]:
        """Check heading hierarchy for issues.

        Args:
            lines: List of document lines.

        Returns:
            List of heading issues.
        """
        issues = []
        heading_levels = []

        for i, line in enumerate(lines):
            heading_match = re.match(r"^(#+)\s+(.+)$", line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2)

                # Check heading level
                if level > self.MAX_HEADING_LEVEL:
                    issues.append(f"Heading level {level} too deep on line {i+1}")

                # Check for skipped levels
                if heading_levels and level > heading_levels[-1] + 1:
                    issues.append(
                        f"Skipped heading level on line {i+1} "
                        f"(jumped from {heading_levels[-1]} to {level})"
                    )

                heading_levels.append(level)

        return issues

    def _check_list_formatting(self, lines: List[str]) -> List[str]:
        """Check list formatting for issues.

        Args:
            lines: List of document lines.

        Returns:
            List of list formatting issues.
        """
        issues = []
        in_list = False
        list_indent = 0

        for i, line in enumerate(lines):
            # Check for unordered list
            ul_match = re.match(r"^(\s*)[-*+]\s+(.+)$", line)
            # Check for ordered list
            ol_match = re.match(r"^(\s*)(\d+)\.\s+(.+)$", line)

            if ul_match or ol_match:
                if not in_list:
                    in_list = True
                    list_indent = len(
                        ul_match.group(1) if ul_match else ol_match.group(1)
                    )
                else:
                    # Check consistent indentation
                    current_indent = len(
                        ul_match.group(1) if ul_match else ol_match.group(1)
                    )
                    if (
                        current_indent != list_indent
                        and current_indent != list_indent + 2
                    ):
                        issues.append(f"Inconsistent list indentation on line {i+1}")
            else:
                in_list = False

        return issues

    def _check_readability(self, content: str) -> Dict[str, Any]:
        """Check content readability.

        Args:
            content: Document content.

        Returns:
            Readability analysis results.
        """
        results = {
            "issues": [],
            "metrics": {},
        }

        # Split into sentences (simple approach)
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Calculate sentence lengths
        sentence_lengths = [len(s.split()) for s in sentences]

        if sentence_lengths:
            results["metrics"]["avg_sentence_length"] = round(
                statistics.mean(sentence_lengths), 1
            )
            results["metrics"]["max_sentence_length"] = max(sentence_lengths)

            # Check for overly long sentences
            long_sentences = sum(
                1 for length in sentence_lengths if length > self.MAX_SENTENCE_LENGTH
            )
            if long_sentences > len(sentences) * 0.2:  # More than 20% are long
                results["issues"].append(
                    f"Too many long sentences ({long_sentences}/{len(sentences)})"
                )

        # Split into paragraphs
        paragraphs = re.split(r"\n\n+", content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Calculate paragraph lengths
        paragraph_lengths = [len(p.split()) for p in paragraphs]

        if paragraph_lengths:
            results["metrics"]["avg_paragraph_length"] = round(
                statistics.mean(paragraph_lengths), 1
            )

            # Check for overly long or short paragraphs
            long_paragraphs = sum(
                1 for length in paragraph_lengths if length > self.MAX_PARAGRAPH_LENGTH
            )
            short_paragraphs = sum(
                1 for length in paragraph_lengths if length < self.MIN_PARAGRAPH_LENGTH
            )

            if long_paragraphs > len(paragraphs) * 0.3:
                results["issues"].append(
                    f"Too many long paragraphs ({long_paragraphs}/{len(paragraphs)})"
                )

            if short_paragraphs > len(paragraphs) * 0.5:
                results["issues"].append(
                    f"Too many short paragraphs ({short_paragraphs}/{len(paragraphs)})"
                )

        # Calculate word count
        words = content.split()
        results["metrics"]["word_count"] = len(words)
        results["metrics"]["character_count"] = len(content)

        return results

    def _validate_structure(self, content: str) -> Dict[str, Any]:
        """Validate document structure.

        Args:
            content: Document content.

        Returns:
            Structure validation results.
        """
        results = {
            "issues": [],
            "sections": [],
        }

        lines = content.split("\n")

        # Extract sections
        current_section = None
        for line in lines:
            heading_match = re.match(r"^(#+)\s+(.+)$", line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2)
                results["sections"].append(
                    {
                        "level": level,
                        "title": title,
                    }
                )

        # Check for missing main heading
        if not results["sections"] or results["sections"][0]["level"] != 1:
            results["issues"].append("Document missing main heading (level 1)")

        # Check for duplicate section titles
        titles = [s["title"] for s in results["sections"]]
        duplicates = [title for title, count in Counter(titles).items() if count > 1]
        if duplicates:
            results["issues"].append(f"Duplicate section titles: {duplicates}")

        return results

    def _check_report_completeness(self, report_dir: Path) -> Dict[str, Any]:
        """Check report completeness.

        Args:
            report_dir: Report directory path.

        Returns:
            Completeness check results.
        """
        results = {
            "complete": True,
            "missing": [],
            "found": [],
        }

        # Required files
        required_files = [
            "00_Table_of_Contents.md",
            "01_Executive_Summary.md",
            "02_Introduction.md",
            "README.md",
            "report_metadata.json",
            "manifest.json",
        ]

        for req_file in required_files:
            file_path = report_dir / req_file
            if file_path.exists():
                results["found"].append(req_file)
            else:
                # Check for variations (e.g., different numbering)
                pattern = req_file.replace("01_", "*_").replace("02_", "*_")
                matches = list(report_dir.glob(pattern))
                if matches:
                    results["found"].append(req_file)
                else:
                    results["complete"] = False
                    results["missing"].append(req_file)

        return results

    def _generate_validation_summary(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable validation summary.

        Args:
            results: Validation results.

        Returns:
            Summary string.
        """
        lines = []

        if results["valid"]:
            lines.append("✅ Report validation PASSED")
        else:
            lines.append("❌ Report validation FAILED")

        lines.append(f"Overall Score: {results['overall_score']}/100")

        if results["issues"]:
            lines.append(f"Issues: {len(results['issues'])}")

        if results["warnings"]:
            lines.append(f"Warnings: {len(results['warnings'])}")

        if "completeness" in results:
            if results["completeness"]["complete"]:
                lines.append("✅ All required files present")
            else:
                lines.append(
                    f"❌ Missing files: {len(results['completeness']['missing'])}"
                )

        return " | ".join(lines)

    def generate_quality_report(self) -> str:
        """Generate a detailed quality report.

        Returns:
            Formatted quality report.
        """
        if not self.validation_results:
            return "No validation results available"

        r = self.validation_results
        lines = [
            "# Report Quality Validation Results",
            "",
            f"**Report Directory**: {r['report_dir']}",
            f"**Overall Score**: {r['overall_score']}/100",
            f"**Status**: {'✅ PASSED' if r['valid'] else '❌ FAILED'}",
            "",
        ]

        # Issues section
        if r["issues"]:
            lines.extend(
                [
                    "## Issues (Must Fix)",
                    "",
                ]
            )
            for issue in r["issues"]:
                lines.append(f"- {issue}")
            lines.append("")

        # Warnings section
        if r["warnings"]:
            lines.extend(
                [
                    "## Warnings (Should Fix)",
                    "",
                ]
            )
            for warning in r["warnings"]:
                lines.append(f"- {warning}")
            lines.append("")

        # File-by-file results
        if r["file_validations"]:
            lines.extend(
                [
                    "## File Validation Details",
                    "",
                ]
            )
            for filename, file_results in r["file_validations"].items():
                status = "✅" if file_results["valid"] else "❌"
                lines.append(
                    f"### {filename} {status} (Score: {file_results['score']}/100)"
                )

                if "metrics" in file_results and file_results["metrics"]:
                    lines.append("**Metrics:**")
                    for metric, value in file_results["metrics"].items():
                        lines.append(f"- {metric.replace('_', ' ').title()}: {value}")

                lines.append("")

        return "\n".join(lines)
