"""Advanced visualization and reporting system for Robodata.

Provides comprehensive data visualization, dashboard creation,
and automated report generation capabilities.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jinja2 import Template

from robodata.core.logging_system import get_logger
from robodata.core.models import DatasetInfo, Episode, Frame


class VisualizationType(Enum):
    """Types of visualizations."""
    HISTOGRAM = "histogram"
    SCATTER_PLOT = "scatter_plot"
    LINE_PLOT = "line_plot"
    HEATMAP = "heatmap"
    BOX_PLOT = "box_plot"
    VIOLIN_PLOT = "violin_plot"
    CORRELATION_MATRIX = "correlation_matrix"
    DISTRIBUTION = "distribution"
    TIME_SERIES = "time_series"
    PLOT_3D = "3d_plot"


class ReportFormat(Enum):
    """Report output formats."""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    MARKDOWN = "markdown"


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    figure_size: tuple[int, int] = (10, 6)
    dpi: int = 100
    style: str = "seaborn-v0_8"
    color_palette: str = "viridis"
    save_format: str = "png"
    interactive: bool = False
    show_grid: bool = True
    show_legend: bool = True


@dataclass
class VisualizationResult:
    """Result of visualization generation."""
    visualization_type: VisualizationType
    title: str
    image_data: Optional[str] = None  # Base64 encoded image
    image_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    interactive_data: Optional[Dict[str, Any]] = None


@dataclass
class ReportSection:
    """Section of a report."""
    title: str
    content: str
    visualizations: List[VisualizationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    """Complete report."""
    title: str
    sections: List[ReportSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: "")
    
    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)


class DataVisualizer:
    """Data visualization utilities."""
    
    def __init__(self, config: VisualizationConfig | None = None):
        self.config = config or VisualizationConfig()
        self.logger = get_logger(__name__)
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style."""
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette)
    
    def create_histogram(
        self, 
        data: np.ndarray, 
        title: str = "Histogram",
        bins: int = 50,
        xlabel: str = "Value",
        ylabel: str = "Frequency"
    ) -> VisualizationResult:
        """Create histogram visualization."""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        
        # Convert to base64 for embedding
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return VisualizationResult(
            visualization_type=VisualizationType.HISTOGRAM,
            title=title,
            image_data=img_data,
            metadata={
                "bins": bins,
                "data_points": len(data),
                "mean": np.mean(data),
                "std": np.std(data)
            }
        )
    
    def create_scatter_plot(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        title: str = "Scatter Plot",
        xlabel: str = "X",
        ylabel: str = "Y",
        alpha: float = 0.6
    ) -> VisualizationResult:
        """Create scatter plot visualization."""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        ax.scatter(x_data, y_data, alpha=alpha)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return VisualizationResult(
            visualization_type=VisualizationType.SCATTER_PLOT,
            title=title,
            image_data=img_data,
            metadata={
                "data_points": len(x_data),
                "correlation": np.corrcoef(x_data, y_data)[0, 1]
            }
        )
    
    def create_line_plot(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        title: str = "Line Plot",
        xlabel: str = "X",
        ylabel: str = "Y"
    ) -> VisualizationResult:
        """Create line plot visualization."""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        ax.plot(x_data, y_data, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return VisualizationResult(
            visualization_type=VisualizationType.LINE_PLOT,
            title=title,
            image_data=img_data,
            metadata={
                "data_points": len(x_data),
                "trend": self._calculate_trend(y_data)
            }
        )
    
    def create_heatmap(
        self,
        data: np.ndarray,
        title: str = "Heatmap",
        xlabel: str = "X",
        ylabel: str = "Y",
        cmap: str = "viridis"
    ) -> VisualizationResult:
        """Create heatmap visualization."""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value')
        
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return VisualizationResult(
            visualization_type=VisualizationType.HEATMAP,
            title=title,
            image_data=img_data,
            metadata={
                "shape": data.shape,
                "min": np.min(data),
                "max": np.max(data),
                "mean": np.mean(data)
            }
        )
    
    def create_correlation_matrix(
        self,
        data: np.ndarray,
        feature_names: List[str],
        title: str = "Correlation Matrix"
    ) -> VisualizationResult:
        """Create correlation matrix visualization."""
        corr_matrix = np.corrcoef(data.T)
        
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title(title)
        
        # Set ticks and labels
        ax.set_xticks(range(len(feature_names)))
        ax.set_yticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticklabels(feature_names)
        
        # Add correlation values
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation')
        
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return VisualizationResult(
            visualization_type=VisualizationType.CORRELATION_MATRIX,
            title=title,
            image_data=img_data,
            metadata={
                "features": feature_names,
                "shape": corr_matrix.shape
            }
        )
    
    def create_box_plot(
        self,
        data: List[np.ndarray],
        labels: List[str],
        title: str = "Box Plot",
        ylabel: str = "Value"
    ) -> VisualizationResult:
        """Create box plot visualization."""
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        ax.boxplot(data, labels=labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return VisualizationResult(
            visualization_type=VisualizationType.BOX_PLOT,
            title=title,
            image_data=img_data,
            metadata={
                "groups": len(data),
                "labels": labels
            }
        )
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return img_base64
    
    def _calculate_trend(self, y_data: np.ndarray) -> str:
        """Calculate trend direction."""
        if len(y_data) < 2:
            return "insufficient_data"
        
        # Simple linear regression
        x = np.arange(len(y_data))
        slope = np.polyfit(x, y_data, 1)[0]
        
        if abs(slope) < 1e-6:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"


class DatasetVisualizer:
    """Specialized visualizer for robotics datasets."""
    
    def __init__(self, config: VisualizationConfig | None = None):
        self.visualizer = DataVisualizer(config)
        self.logger = get_logger(__name__)
    
    def visualize_dataset_overview(
        self, 
        dataset_info: DatasetInfo,
        episodes: List[Episode]
    ) -> List[VisualizationResult]:
        """Create overview visualizations for dataset."""
        visualizations = []
        
        # Episode length distribution
        episode_lengths = [len(list(ep.load_frames())) for ep in episodes]
        viz = self.visualizer.create_histogram(
            np.array(episode_lengths),
            title="Episode Length Distribution",
            bins=30,
            xlabel="Episode Length (frames)",
            ylabel="Frequency"
        )
        visualizations.append(viz)
        
        # Frame rate analysis
        frame_rates = []
        for episode in episodes[:100]:  # Sample for performance
            frames = list(episode.load_frames())
            timestamps = [f.timestamp for f in frames if f.timestamp is not None]
            if len(timestamps) > 1:
                duration = timestamps[-1] - timestamps[0]
                fps = (len(frames) - 1) / duration
                frame_rates.append(fps)
        
        if frame_rates:
            viz = self.visualizer.create_histogram(
                np.array(frame_rates),
                title="Frame Rate Distribution",
                bins=20,
                xlabel="Frames per Second",
                ylabel="Frequency"
            )
            visualizations.append(viz)
        
        # Camera information
        if dataset_info.cameras:
            camera_resolutions = [cam.height * cam.width for cam in dataset_info.cameras.values()]
            camera_names = list(dataset_info.cameras.keys())
            
            viz = self.visualizer.create_box_plot(
                [np.array([res]) for res in camera_resolutions],
                camera_names,
                title="Camera Resolutions",
                ylabel="Total Pixels"
            )
            visualizations.append(viz)
        
        return visualizations
    
    def visualize_action_analysis(
        self, 
        episodes: List[Episode],
        max_episodes: int = 100
    ) -> List[VisualizationResult]:
        """Create action analysis visualizations."""
        visualizations = []
        
        # Collect action data
        all_actions = []
        for episode in episodes[:max_episodes]:
            for frame in episode.load_frames():
                if frame.action is not None:
                    all_actions.append(np.array(frame.action).flatten())
        
        if not all_actions:
            return visualizations
        
        actions_matrix = np.array(all_actions)
        
        # Action magnitude distribution
        action_magnitudes = np.linalg.norm(actions_matrix, axis=1)
        viz = self.visualizer.create_histogram(
            action_magnitudes,
            title="Action Magnitude Distribution",
            bins=50,
            xlabel="Action Magnitude",
            ylabel="Frequency"
        )
        visualizations.append(viz)
        
        # Action correlation matrix
        if actions_matrix.shape[1] > 1:
            feature_names = [f"Action Dim {i}" for i in range(actions_matrix.shape[1])]
            viz = self.visualizer.create_correlation_matrix(
                actions_matrix,
                feature_names,
                title="Action Correlation Matrix"
            )
            visualizations.append(viz)
        
        # Action time series (sample)
        if len(all_actions) > 10:
            sample_actions = actions_matrix[:min(100, len(all_actions))]
            
            for i in range(min(3, actions_matrix.shape[1])):  # First 3 dimensions
                viz = self.visualizer.create_line_plot(
                    np.arange(len(sample_actions)),
                    sample_actions[:, i],
                    title=f"Action Dimension {i} Over Time",
                    xlabel="Frame Index",
                    ylabel="Action Value"
                )
                visualizations.append(viz)
        
        return visualizations
    
    def visualize_state_analysis(
        self, 
        episodes: List[Episode],
        max_episodes: int = 100
    ) -> List[VisualizationResult]:
        """Create state analysis visualizations."""
        visualizations = []
        
        # Collect state data
        all_states = []
        for episode in episodes[:max_episodes]:
            for frame in episode.load_frames():
                if frame.state is not None:
                    all_states.append(np.array(frame.state).flatten())
        
        if not all_states:
            return visualizations
        
        states_matrix = np.array(all_states)
        
        # State distribution
        for i in range(min(4, states_matrix.shape[1])):  # First 4 dimensions
            viz = self.visualizer.create_histogram(
                states_matrix[:, i],
                title=f"State Dimension {i} Distribution",
                bins=30,
                xlabel="State Value",
                ylabel="Frequency"
            )
            visualizations.append(viz)
        
        # State trajectory (2D projection)
        if states_matrix.shape[1] >= 2:
            viz = self.visualizer.create_scatter_plot(
                states_matrix[:, 0],
                states_matrix[:, 1],
                title="State Trajectory (First 2 Dimensions)",
                xlabel="State Dim 0",
                ylabel="State Dim 1",
                alpha=0.5
            )
            visualizations.append(viz)
        
        return visualizations


class ReportGenerator:
    """Generate comprehensive reports."""
    
    def __init__(self, output_dir: Path = Path("./reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.logger = get_logger(__name__)
        self.visualizer = DatasetVisualizer()
    
    def generate_dataset_report(
        self,
        dataset_info: DatasetInfo,
        episodes: List[Episode],
        format: ReportFormat = ReportFormat.HTML
    ) -> Report:
        """Generate comprehensive dataset report."""
        self.logger.info(f"Generating {format.value} dataset report")
        
        report = Report(
            title=f"Dataset Analysis Report - {dataset_info.path.name}",
            metadata={
                "dataset_format": dataset_info.format,
                "total_episodes": dataset_info.num_episodes,
                "total_frames": dataset_info.total_frames,
                "cameras": list(dataset_info.cameras.keys())
            }
        )
        
        # Overview section
        overview_section = self._create_overview_section(dataset_info, episodes)
        report.add_section(overview_section)
        
        # Action analysis section
        action_section = self._create_action_analysis_section(episodes)
        report.add_section(action_section)
        
        # State analysis section
        state_section = self._create_state_analysis_section(episodes)
        report.add_section(state_section)
        
        # Quality metrics section
        quality_section = self._create_quality_section(dataset_info, episodes)
        report.add_section(quality_section)
        
        # Save report
        if format == ReportFormat.HTML:
            self._save_html_report(report)
        elif format == ReportFormat.JSON:
            self._save_json_report(report)
        elif format == ReportFormat.MARKDOWN:
            self._save_markdown_report(report)
        
        return report
    
    def _create_overview_section(self, dataset_info: DatasetInfo, episodes: List[Episode]) -> ReportSection:
        """Create overview section."""
        visualizations = self.visualizer.visualize_dataset_overview(dataset_info, episodes)
        
        content = f"""
        ## Dataset Overview
        
        **Format**: {dataset_info.format}
        **Total Episodes**: {dataset_info.num_episodes}
        **Total Frames**: {dataset_info.total_frames}
        **Cameras**: {', '.join(dataset_info.cameras.keys())}
        
        ### Key Statistics
        - Average Episode Length: {np.mean([len(list(ep.load_frames())) for ep in episodes]):.1f} frames
        - Dataset Size: {dataset_info.total_frames} frames
        """
        
        return ReportSection(
            title="Dataset Overview",
            content=content,
            visualizations=visualizations
        )
    
    def _create_action_analysis_section(self, episodes: List[Episode]) -> ReportSection:
        """Create action analysis section."""
        visualizations = self.visualizer.visualize_action_analysis(episodes)
        
        content = """
        ## Action Analysis
        
        This section provides detailed analysis of the action sequences in the dataset.
        
        ### Key Insights
        - Action magnitude distribution
        - Correlation between action dimensions
        - Temporal patterns in actions
        """
        
        return ReportSection(
            title="Action Analysis",
            content=content,
            visualizations=visualizations
        )
    
    def _create_state_analysis_section(self, episodes: List[Episode]) -> ReportSection:
        """Create state analysis section."""
        visualizations = self.visualizer.visualize_state_analysis(episodes)
        
        content = """
        ## State Analysis
        
        This section analyzes the state space representation in the dataset.
        
        ### Key Insights
        - State dimension distributions
        - State trajectories
        - State space coverage
        """
        
        return ReportSection(
            title="State Analysis",
            content=content,
            visualizations=visualizations
        )
    
    def _create_quality_section(self, dataset_info: DatasetInfo, episodes: List[Episode]) -> ReportSection:
        """Create quality metrics section."""
        # Calculate basic quality metrics
        total_frames = sum(len(list(ep.load_frames())) for ep in episodes)
        complete_episodes = sum(1 for ep in episodes if self._is_episode_complete(ep))
        
        content = f"""
        ## Quality Metrics
        
        ### Data Completeness
        - Complete Episodes: {complete_episodes}/{len(episodes)} ({complete_episodes/len(episodes)*100:.1f}%)
        - Total Frames: {total_frames}
        
        ### Data Consistency
        - Frame Rate Consistency: {'Good' if self._check_frame_rate_consistency(episodes) else 'Needs Attention'}
        - Camera Consistency: {'Good' if len(dataset_info.cameras) > 0 else 'No Cameras'}
        
        ### Recommendations
        - {'✓' if complete_episodes/len(episodes) > 0.9 else '⚠'} Consider removing incomplete episodes
        - {'✓' if self._check_frame_rate_consistency(episodes) else '⚠'} Standardize frame rates
        """
        
        return ReportSection(
            title="Quality Metrics",
            content=content
        )
    
    def _is_episode_complete(self, episode: Episode) -> bool:
        """Check if episode appears complete."""
        frames = list(episode.load_frames())
        return len(frames) > 10  # Simple heuristic
    
    def _check_frame_rate_consistency(self, episodes: List[Episode]) -> bool:
        """Check frame rate consistency across episodes."""
        frame_rates = []
        for episode in episodes[:10]:  # Sample
            frames = list(episode.load_frames())
            timestamps = [f.timestamp for f in frames if f.timestamp is not None]
            if len(timestamps) > 1:
                duration = timestamps[-1] - timestamps[0]
                fps = (len(frames) - 1) / duration
                frame_rates.append(fps)
        
        if len(frame_rates) < 2:
            return True
        
        mean_fps = np.mean(frame_rates)
        return all(abs(fps - mean_fps) / mean_fps < 0.1 for fps in frame_rates)
    
    def _save_html_report(self, report: Report) -> Path:
        """Save report as HTML."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ report.title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin-bottom: 40px; }
                .visualization { text-align: center; margin: 20px 0; }
                .visualization img { max-width: 100%; height: auto; }
                .metadata { background: #f5f5f5; padding: 15px; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>{{ report.title }}</h1>
            <div class="metadata">
                <h2>Metadata</h2>
                <table>
                    {% for key, value in report.metadata.items() %}
                    <tr><td><strong>{{ key }}:</strong></td><td>{{ value }}</td></tr>
                    {% endfor %}
                </table>
            </div>
            
            {% for section in report.sections %}
            <div class="section">
                <h2>{{ section.title }}</h2>
                <div>{{ section.content | safe }}</div>
                
                {% for viz in section.visualizations %}
                <div class="visualization">
                    <h3>{{ viz.title }}</h3>
                    <img src="data:image/png;base64,{{ viz.image_data }}" alt="{{ viz.title }}">
                    {% if viz.metadata %}
                    <div class="metadata">
                        <strong>Metadata:</strong>
                        <table>
                            {% for key, value in viz.metadata.items() %}
                            <tr><td>{{ key }}:</td><td>{{ value }}</td></tr>
                            {% endfor %}
                        </table>
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            {% endfor %}
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(report=report)
        
        output_file = self.output_dir / f"{report.title.replace(' ', '_').lower()}.html"
        output_file.write_text(html_content)
        
        self.logger.info(f"HTML report saved to {output_file}")
        return output_file
    
    def _save_json_report(self, report: Report) -> Path:
        """Save report as JSON."""
        # Convert report to JSON-serializable format
        report_dict = {
            "title": report.title,
            "metadata": report.metadata,
            "sections": []
        }
        
        for section in report.sections:
            section_dict = {
                "title": section.title,
                "content": section.content,
                "visualizations": []
            }
            
            for viz in section.visualizations:
                viz_dict = {
                    "title": viz.title,
                    "visualization_type": viz.visualization_type.value,
                    "metadata": viz.metadata
                }
                if viz.image_data:
                    viz_dict["image_data"] = viz.image_data
                section_dict["visualizations"].append(viz_dict)
            
            report_dict["sections"].append(section_dict)
        
        output_file = self.output_dir / f"{report.title.replace(' ', '_').lower()}.json"
        output_file.write_text(json.dumps(report_dict, indent=2))
        
        self.logger.info(f"JSON report saved to {output_file}")
        return output_file
    
    def _save_markdown_report(self, report: Report) -> Path:
        """Save report as Markdown."""
        md_content = f"# {report.title}\n\n"
        
        # Add metadata
        if report.metadata:
            md_content += "## Metadata\n\n"
            for key, value in report.metadata.items():
                md_content += f"- **{key}**: {value}\n"
            md_content += "\n"
        
        # Add sections
        for section in report.sections:
            md_content += f"## {section.title}\n\n"
            md_content += f"{section.content}\n\n"
            
            # Add image references for visualizations
            for viz in section.visualizations:
                if viz.image_path:
                    md_content += f"### {viz.title}\n\n"
                    md_content += f"![{viz.title}]({viz.image_path.name})\n\n"
        
        output_file = self.output_dir / f"{report.title.replace(' ', '_').lower()}.md"
        output_file.write_text(md_content)
        
        self.logger.info(f"Markdown report saved to {output_file}")
        return output_file


# Convenience functions
def generate_dataset_report(
    dataset_info: DatasetInfo,
    episodes: List[Episode],
    output_dir: Path = Path("./reports"),
    format: ReportFormat = ReportFormat.HTML
) -> Report:
    """Generate a dataset report."""
    generator = ReportGenerator(output_dir)
    return generator.generate_dataset_report(dataset_info, episodes, format)


def create_visualization(
    viz_type: VisualizationType,
    data: Any,
    title: str = "Visualization",
    **kwargs
) -> VisualizationResult:
    """Create a single visualization."""
    visualizer = DataVisualizer()
    
    if viz_type == VisualizationType.HISTOGRAM:
        return visualizer.create_histogram(data, title, **kwargs)
    elif viz_type == VisualizationType.SCATTER_PLOT:
        return visualizer.create_scatter_plot(data[0], data[1], title, **kwargs)
    elif viz_type == VisualizationType.LINE_PLOT:
        return visualizer.create_line_plot(data[0], data[1], title, **kwargs)
    elif viz_type == VisualizationType.HEATMAP:
        return visualizer.create_heatmap(data, title, **kwargs)
    else:
        raise ValueError(f"Unsupported visualization type: {viz_type}")
