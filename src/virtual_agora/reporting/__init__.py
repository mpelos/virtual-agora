"""Reporting and documentation system for Virtual Agora.

This module provides comprehensive reporting capabilities including:
- Per-topic summary generation
- Multi-file final report generation
- Session logging and analytics
- Report quality validation
- Export and distribution options
- Customizable templates
"""

from .topic_summary import TopicSummaryGenerator
from .report_structure import ReportStructureManager
from .report_writer import ReportSectionWriter
from .file_manager import ReportFileManager
from .session_logger import EnhancedSessionLogger
from .metadata import ReportMetadataGenerator
from .quality_validator import ReportQualityValidator
from .exporter import ReportExporter
from .templates import ReportTemplateManager

__all__ = [
    "TopicSummaryGenerator",
    "ReportStructureManager",
    "ReportSectionWriter",
    "ReportFileManager",
    "EnhancedSessionLogger",
    "ReportMetadataGenerator",
    "ReportQualityValidator",
    "ReportExporter",
    "ReportTemplateManager",
]
