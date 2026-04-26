"""Advanced validation system for Robodata.

Provides comprehensive data validation, schema checking, and
quality assurance for robotics datasets.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from robodata.core.logging_system import get_logger
from robodata.core.models import DatasetInfo, Episode, Frame, FieldSchema, Dtype


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation checks."""
    SCHEMA = "schema"
    DATA_INTEGRITY = "data_integrity"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    QUALITY = "quality"


@dataclass
class ValidationIssue:
    """Single validation issue."""
    level: ValidationLevel
    category: ValidationCategory
    message: str
    field: Optional[str] = None
    episode_id: Optional[str] = None
    frame_index: Optional[int] = None
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validation process."""
    success: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def issues_by_level(self) -> Dict[ValidationLevel, List[ValidationIssue]]:
        """Group issues by validation level."""
        result = {level: [] for level in ValidationLevel}
        for issue in self.issues:
            result[issue.level].append(issue)
        return result
    
    @property
    def issues_by_category(self) -> Dict[ValidationCategory, List[ValidationIssue]]:
        """Group issues by validation category."""
        result = {category: [] for category in ValidationCategory}
        for issue in self.issues:
            result[issue.category].append(issue)
        return result
    
    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """Get critical issues."""
        return self.issues_by_level[ValidationLevel.CRITICAL]
    
    @property
    def error_issues(self) -> List[ValidationIssue]:
        """Get error issues."""
        return self.issues_by_level[ValidationLevel.ERROR]


class Validator(ABC):
    """Abstract base class for validators."""
    
    @abstractmethod
    def validate(self, data: Any, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate data and return issues."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Validator name."""
        pass


class SchemaValidator(Validator):
    """Validate dataset schema compliance."""
    
    def __init__(self, required_fields: List[str], optional_fields: List[str] = None):
        self.required_fields = required_fields
        self.optional_fields = optional_fields or []
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "schema_validator"
    
    def validate(self, dataset_info: DatasetInfo, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate dataset schema."""
        issues = []
        
        # Check required fields
        for field_name in self.required_fields:
            if field_name not in dataset_info.schema:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.SCHEMA,
                    message=f"Missing required field: {field_name}",
                    field=field_name,
                    suggestion=f"Add {field_name} to your dataset"
                ))
        
        # Validate field schemas
        for field_name, field_schema in dataset_info.schema.items():
            field_issues = self._validate_field_schema(field_name, field_schema)
            issues.extend(field_issues)
        
        return issues
    
    def _validate_field_schema(self, field_name: str, field_schema: FieldSchema) -> List[ValidationIssue]:
        """Validate individual field schema."""
        issues = []
        
        # Check shape validity
        if not field_schema.shape or any(dim <= 0 for dim in field_schema.shape):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SCHEMA,
                message=f"Invalid shape for field {field_name}: {field_schema.shape}",
                field=field_name,
                suggestion="Ensure all dimensions are positive integers"
            ))
        
        # Check dtype compatibility
        if field_schema.dtype == Dtype.STRING and field_schema.shape:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.SCHEMA,
                message=f"String field {field_name} should not have shape",
                field=field_name,
                suggestion="Remove shape for string fields"
            ))
        
        return issues


class DataIntegrityValidator(Validator):
    """Validate data integrity and consistency."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "data_integrity_validator"
    
    def validate(self, episodes: List[Episode], context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate data integrity across episodes."""
        issues = []
        
        if not episodes:
            issues.append(ValidationIssue(
                level=ValidationLevel.CRITICAL,
                category=ValidationCategory.DATA_INTEGRITY,
                message="No episodes found in dataset"
            ))
            return issues
        
        # Check episode consistency
        issues.extend(self._validate_episode_consistency(episodes))
        
        # Check frame sequences
        issues.extend(self._validate_frame_sequences(episodes))
        
        # Check data ranges
        issues.extend(self._validate_data_ranges(episodes))
        
        return issues
    
    def _validate_episode_consistency(self, episodes: List[Episode]) -> List[ValidationIssue]:
        """Validate episode consistency."""
        issues = []
        
        # Check for duplicate episode IDs
        episode_ids = [ep.episode_id for ep in episodes]
        duplicates = [ep_id for ep_id in set(episode_ids) if episode_ids.count(ep_id) > 1]
        
        for dup_id in duplicates:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.DATA_INTEGRITY,
                message=f"Duplicate episode ID: {dup_id}",
                episode_id=dup_id,
                suggestion="Ensure all episode IDs are unique"
            ))
        
        # Check episode lengths distribution
        episode_lengths = [len(list(ep.load_frames())) for ep in episodes]
        if episode_lengths:
            avg_length = np.mean(episode_lengths)
            std_length = np.std(episode_lengths)
            
            # Flag outliers (more than 3 std from mean)
            for i, length in enumerate(episode_lengths):
                if abs(length - avg_length) > 3 * std_length:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        category=ValidationCategory.DATA_INTEGRITY,
                        message=f"Episode {episodes[i].episode_id} has unusual length: {length} (avg: {avg_length:.1f})",
                        episode_id=episodes[i].episode_id,
                        suggestion="Check if this episode is complete or corrupted"
                    ))
        
        return issues
    
    def _validate_frame_sequences(self, episodes: List[Episode]) -> List[ValidationIssue]:
        """Validate frame sequence consistency."""
        issues = []
        
        for episode in episodes:
            frames = list(episode.load_frames())
            
            if not frames:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.DATA_INTEGRITY,
                    message=f"Episode {episode.episode_id} has no frames",
                    episode_id=episode.episode_id,
                    suggestion="Ensure episodes contain at least one frame"
                ))
                continue
            
            # Check timestamp monotonicity
            timestamps = [frame.timestamp for frame in frames if frame.timestamp is not None]
            if timestamps:
                for i in range(1, len(timestamps)):
                    if timestamps[i] <= timestamps[i-1]:
                        issues.append(ValidationIssue(
                            level=ValidationLevel.WARNING,
                            category=ValidationCategory.DATA_INTEGRITY,
                            message=f"Non-monotonic timestamp in episode {episode.episode_id} at frame {i}",
                            episode_id=episode.episode_id,
                            frame_index=i,
                            suggestion="Ensure timestamps are monotonically increasing"
                        ))
        
        return issues
    
    def _validate_data_ranges(self, episodes: List[Episode]) -> List[ValidationIssue]:
        """Validate data value ranges."""
        issues = []
        
        # Collect statistics across all episodes
        all_actions = []
        all_states = []
        
        for episode in episodes:
            for frame in episode.load_frames():
                if frame.action is not None:
                    all_actions.append(frame.action)
                if frame.state is not None:
                    all_states.append(frame.state)
        
        # Check action ranges
        if all_actions:
            actions_array = np.array(all_actions)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(actions_array)):
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.DATA_INTEGRITY,
                    message="NaN values found in action data",
                    suggestion="Replace NaN values with appropriate defaults"
                ))
            
            if np.any(np.isinf(actions_array)):
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.DATA_INTEGRITY,
                    message="Infinite values found in action data",
                    suggestion="Check for numerical overflow in action computation"
                ))
            
            # Check for suspicious ranges
            action_std = np.std(actions_array)
            if action_std == 0:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.QUALITY,
                    message="Action data has zero variance (constant values)",
                    suggestion="Check if action data is properly computed"
                ))
        
        # Similar checks for states
        if all_states:
            states_array = np.array(all_states)
            
            if np.any(np.isnan(states_array)):
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.DATA_INTEGRITY,
                    message="NaN values found in state data",
                    suggestion="Replace NaN values with appropriate defaults"
                ))
        
        return issues


class PerformanceValidator(Validator):
    """Validate performance characteristics."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "performance_validator"
    
    def validate(self, dataset_info: DatasetInfo, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate performance-related aspects."""
        issues = []
        
        # Check dataset size
        if dataset_info.total_frames == 0:
            issues.append(ValidationIssue(
                level=ValidationLevel.CRITICAL,
                category=ValidationCategory.PERFORMANCE,
                message="Dataset has no frames",
                suggestion="Check if dataset is properly loaded"
            ))
        
        # Check frame rate consistency
        if dataset_info.inferred_fps:
            if dataset_info.inferred_fps < 1:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.PERFORMANCE,
                    message=f"Very low frame rate: {dataset_info.inferred_fps} Hz",
                    suggestion="Consider increasing frame rate for better temporal resolution"
                ))
            elif dataset_info.inferred_fps > 100:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.PERFORMANCE,
                    message=f"Very high frame rate: {dataset_info.inferred_fps} Hz",
                    suggestion="Consider downsampling to reduce storage requirements"
                ))
        
        # Check image sizes
        for camera_name, camera_info in dataset_info.cameras.items():
            total_pixels = camera_info.height * camera_info.width
            
            if total_pixels > 1920 * 1080:  # Full HD
                issues.append(ValidationIssue(
                    level=ValidationLevel.INFO,
                    category=ValidationCategory.PERFORMANCE,
                    message=f"High resolution camera {camera_name}: {camera_info.width}x{camera_info.height}",
                    field=camera_name,
                    suggestion="Consider if this resolution is necessary for your application"
                ))
            
            if total_pixels < 320 * 240:  # Low resolution
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.PERFORMANCE,
                    message=f"Low resolution camera {camera_name}: {camera_info.width}x{camera_info.height}",
                    field=camera_name,
                    suggestion="Consider increasing resolution for better visual quality"
                ))
        
        return issues


class ComplianceValidator(Validator):
    """Validate compliance with standards and best practices."""
    
    def __init__(self, standard: str = "robodata_v2"):
        self.standard = standard
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "compliance_validator"
    
    def validate(self, dataset_info: DatasetInfo, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate compliance with standards."""
        issues = []
        
        # Check required metadata
        required_metadata = ["language_instruction", "episode_id"]
        for field in required_metadata:
            if field not in dataset_info.metadata:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.COMPLIANCE,
                    message=f"Missing recommended metadata: {field}",
                    suggestion=f"Add {field} to improve dataset usability"
                ))
        
        # Check field naming conventions
        for field_name in dataset_info.schema.keys():
            if not field_name.replace('_', '').isalnum():
                issues.append(ValidationIssue(
                    level=ValidationLevel.INFO,
                    category=ValidationCategory.COMPLIANCE,
                    message=f"Field name '{field_name}' contains special characters",
                    field=field_name,
                    suggestion="Use alphanumeric names with underscores only"
                ))
        
        # Check camera naming
        for camera_name in dataset_info.cameras.keys():
            if not camera_name.replace('_', '').isalnum():
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.COMPLIANCE,
                    message=f"Camera name '{camera_name}' contains special characters",
                    field=camera_name,
                    suggestion="Use alphanumeric camera names with underscores only"
                ))
        
        return issues


class ValidationEngine:
    """Main validation engine that orchestrates all validators."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.validators: List[Validator] = []
        self._setup_default_validators()
    
    def _setup_default_validators(self):
        """Setup default validators."""
        # These would be configured based on dataset type
        self.validators = [
            SchemaValidator(
                required_fields=["action", "observation"],
                optional_fields=["reward", "done"]
            ),
            DataIntegrityValidator(),
            PerformanceValidator(),
            ComplianceValidator()
        ]
    
    def add_validator(self, validator: Validator):
        """Add a custom validator."""
        self.validators.append(validator)
        self.logger.info(f"Added validator: {validator.name}")
    
    def remove_validator(self, validator_name: str):
        """Remove a validator by name."""
        self.validators = [v for v in self.validators if v.name != validator_name]
        self.logger.info(f"Removed validator: {validator_name}")
    
    def validate_dataset(
        self,
        dataset_info: DatasetInfo,
        episodes: Optional[List[Episode]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate entire dataset."""
        context = context or {}
        all_issues = []
        
        self.logger.info(f"Starting validation with {len(self.validators)} validators")
        
        for validator in self.validators:
            try:
                self.logger.debug(f"Running validator: {validator.name}")
                
                # Different validators need different data
                if isinstance(validator, SchemaValidator):
                    issues = validator.validate(dataset_info, context)
                elif isinstance(validator, DataIntegrityValidator):
                    if episodes is None:
                        self.logger.warning(f"Skipping {validator.name}: no episodes provided")
                        continue
                    issues = validator.validate(episodes, context)
                else:
                    issues = validator.validate(dataset_info, context)
                
                all_issues.extend(issues)
                self.logger.debug(f"{validator.name} found {len(issues)} issues")
                
            except Exception as e:
                self.logger.error(f"Validator {validator.name} failed: {e}")
                all_issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.COMPLIANCE,
                    message=f"Validation failed: {validator.name} - {e}"
                ))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues)
        
        # Create result
        result = ValidationResult(
            success=len([i for i in all_issues if i.level in [ValidationLevel.CRITICAL, ValidationLevel.ERROR]]) == 0,
            issues=all_issues,
            stats=self._generate_stats(all_issues),
            recommendations=recommendations
        )
        
        self.logger.info(f"Validation complete: {len(all_issues)} issues found")
        return result
    
    def _generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate recommendations based on issues."""
        recommendations = []
        
        # Count issues by category
        category_counts = {}
        for issue in issues:
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
        
        # Generate category-specific recommendations
        if ValidationCategory.SCHEMA in category_counts:
            recommendations.append("Review dataset schema and ensure all required fields are present")
        
        if ValidationCategory.DATA_INTEGRITY in category_counts:
            recommendations.append("Check data integrity and fix corrupted or missing values")
        
        if ValidationCategory.PERFORMANCE in category_counts:
            recommendations.append("Optimize dataset performance by adjusting frame rates and resolutions")
        
        if ValidationCategory.COMPLIANCE in category_counts:
            recommendations.append("Ensure compliance with naming conventions and metadata standards")
        
        if ValidationCategory.QUALITY in category_counts:
            recommendations.append("Improve data quality by addressing outliers and anomalies")
        
        return recommendations
    
    def _generate_stats(self, issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Generate validation statistics."""
        stats = {
            "total_issues": len(issues),
            "by_level": {},
            "by_category": {},
            "affected_episodes": len(set(issue.episode_id for issue in issues if issue.episode_id)),
            "affected_fields": len(set(issue.field for issue in issues if issue.field))
        }
        
        # Count by level
        for level in ValidationLevel:
            stats["by_level"][level.value] = len([i for i in issues if i.level == level])
        
        # Count by category
        for category in ValidationCategory:
            stats["by_category"][category.value] = len([i for i in issues if i.category == category])
        
        return stats


# Convenience functions
def validate_dataset(
    dataset_info: DatasetInfo,
    episodes: Optional[List[Episode]] = None,
    custom_validators: Optional[List[Validator]] = None
) -> ValidationResult:
    """Validate a dataset with default validators."""
    engine = ValidationEngine()
    
    if custom_validators:
        for validator in custom_validators:
            engine.add_validator(validator)
    
    return engine.validate_dataset(dataset_info, episodes)


def create_schema_validator(required_fields: List[str], optional_fields: List[str] = None) -> SchemaValidator:
    """Create a schema validator with custom field requirements."""
    return SchemaValidator(required_fields, optional_fields)


def create_compliance_validator(standard: str = "robodata_v2") -> ComplianceValidator:
    """Create a compliance validator for specific standard."""
    return ComplianceValidator(standard)
