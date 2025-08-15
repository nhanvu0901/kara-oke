import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EducationalReporter:
    """Generate educational reports from processing results."""

    def __init__(self, config: Dict):
        self.config = config

    def generate_report(self, results: Dict) -> Dict:
        """Generate comprehensive educational report."""
        html = self._generate_html_report(results)

        return {
            "html": html,
            "timestamp": datetime.now().isoformat()
        }

    def _generate_html_report(self, results: Dict) -> str:
        """Generate HTML report."""
        insights_html = ""
        if results.get("insights"):
            for insight in results["insights"]:
                insights_html += f"""
                <div class="insight">
                    <h3>{insight['stage'].capitalize()}</h3>
                    <p>{insight['content']}</p>
                </div>
                """

        stages_html = ""
        for stage, data in results.get("stages", {}).items():
            stages_html += f"""
            <div class="stage">
                <h3>{stage.capitalize()}</h3>
                <pre>{json.dumps(data, indent=2)}</pre>
            </div>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Educational Audio Processing Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{
                    background: white;
                    border-radius: 10px;
                    padding: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                }}
                h1 {{
                    color: #667eea;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #764ba2;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #4a5568;
                }}
                .metadata {{
                    background: #f7fafc;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .insight {{
                    background: #edf2f7;
                    padding: 15px;
                    border-left: 4px solid #667eea;
                    margin: 15px 0;
                    border-radius: 5px;
                }}
                .stage {{
                    background: #f7fafc;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 5px;
                }}
                pre {{
                    background: #2d3748;
                    color: #e2e8f0;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .metric {{
                    background: #edf2f7;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #667eea;
                }}
                .metric-label {{
                    color: #718096;
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎵 Educational Audio Processing Report</h1>

                <div class="metadata">
                    <strong>Input:</strong> {results.get('input', 'N/A')}<br>
                    <strong>Mode:</strong> {results.get('mode', 'N/A')}<br>
                    <strong>Timestamp:</strong> {results.get('timestamp', 'N/A')}<br>
                    <strong>Processing Time:</strong> {results.get('total_time', 0):.2f} seconds
                </div>

                <h2>Educational Insights</h2>
                {insights_html}

                <h2>Processing Stages</h2>
                {stages_html}

                <h2>Key Metrics</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">{results.get('total_time', 0):.1f}s</div>
                        <div class="metric-label">Total Time</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{len(results.get('stages', {}))} </div>
                        <div class="metric-label">Stages Completed</div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def generate_comparison_report(self, results_list: List[Dict]) -> Dict:
        """Generate comparison report for multiple processing runs."""
        # Implementation for comparing different model outputs
        comparison = {
            "num_files": len(results_list),
            "models_used": set(),
            "average_processing_time": 0,
            "comparisons": []
        }

        total_time = 0
        for result in results_list:
            total_time += result.get("total_time", 0)

            # Collect models used
            if "stages" in result:
                if "separation" in result["stages"]:
                    comparison["models_used"].add(
                        result["stages"]["separation"].get("model", "unknown")
                    )

        comparison["average_processing_time"] = total_time / len(results_list)

        return comparison