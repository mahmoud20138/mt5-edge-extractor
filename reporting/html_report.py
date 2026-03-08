"""
HTML Report Generator Module.

Generates comprehensive HTML reports for edge analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os

from .visualizer import EdgeVisualizer


class HTMLReportGenerator:
    """
    Generate HTML reports for edge analysis.
    
    Creates comprehensive, styled HTML reports with:
    - Executive summary
    - Edge analysis tables
    - Visualizations
    - Detailed statistics
    """
    
    def __init__(self):
        """Initialize HTML report generator."""
        self.visualizer = EdgeVisualizer()
    
    def generate_report(self, 
                       edge_results: Dict,
                       df: pd.DataFrame = None,
                       returns: pd.Series = None,
                       title: str = "Edge Analysis Report") -> str:
        """
        Generate complete HTML report.
        
        Args:
            edge_results: Dictionary of edge results
            df: Full DataFrame with features
            returns: Strategy returns
            title: Report title
            
        Returns:
            HTML string
        """
        # Generate visualizations
        dashboard = {}
        if returns is not None:
            dashboard = self.visualizer.create_dashboard(returns, edge_results, df)
        
        # Build HTML
        html = self._build_html(edge_results, df, dashboard, title)
        
        return html
    
    def _build_html(self, edge_results: Dict, df: pd.DataFrame,
                   dashboard: Dict, title: str) -> str:
        """Build HTML content."""
        # Filter significant edges
        significant = {k: v for k, v in edge_results.items() if v.is_significant}
        
        # Statistics
        total_edges = len(edge_results)
        significant_edges = len(significant)
        
        # Build sections
        sections = []
        
        # Header
        sections.append(self._build_header(title))
        
        # Executive Summary
        sections.append(self._build_summary(total_edges, significant_edges, significant))
        
        # Top Edges Table
        sections.append(self._build_edges_table(significant))
        
        # Visualizations
        if dashboard:
            sections.append(self._build_visualizations(dashboard))
        
        # Edge Categories
        sections.append(self._build_categories(edge_results))
        
        # Data Summary
        if df is not None:
            sections.append(self._build_data_summary(df))
        
        # Footer
        sections.append(self._build_footer())
        
        return "\n".join(sections)
    
    def _build_header(self, title: str) -> str:
        """Build HTML header."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 40px 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .header .subtitle {{
            color: #888;
            font-size: 1.1rem;
        }}
        .header .warning {{
            margin-top: 20px;
            padding: 15px;
            background: rgba(231, 76, 60, 0.2);
            border-radius: 8px;
            border-left: 4px solid #e74c3c;
            color: #e74c3c;
        }}
        .section {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .section h2 {{
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #00d2ff;
            border-bottom: 2px solid rgba(0,210,255,0.3);
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-card .value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-card .label {{
            color: #888;
            font-size: 0.9rem;
        }}
        .stat-card.positive .value {{ color: #2ecc71; }}
        .stat-card.negative .value {{ color: #e74c3c; }}
        .stat-card.neutral .value {{ color: #3498db; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{
            background: rgba(0,210,255,0.1);
            color: #00d2ff;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
        }}
        tr:hover {{
            background: rgba(255,255,255,0.05);
        }}
        .positive {{ color: #2ecc71; }}
        .negative {{ color: #e74c3c; }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.9rem;
        }}
        .checklist {{
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            padding: 20px;
        }}
        .checklist-item {{
            display: flex;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .checklist-item:last-child {{
            border-bottom: none;
        }}
        .checklist-item .icon {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            margin-right: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
        }}
        .checklist-item .icon.pending {{ background: rgba(243, 156, 18, 0.3); color: #f39c12; }}
        .checklist-item .icon.check {{ background: rgba(46, 204, 113, 0.3); color: #2ecc71; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {title}</h1>
            <p class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <div class="warning">
                ⚠️ <strong>FOR BACKTESTING ONLY</strong> - No live trades are executed. 
                Past performance does not guarantee future results.
            </div>
        </div>
"""
    
    def _build_summary(self, total: int, significant: int, 
                       significant_edges: Dict) -> str:
        """Build executive summary section."""
        # Calculate top Sharpe
        if significant_edges:
            top_sharpe = max(e.sharpe_ratio for e in significant_edges.values())
            avg_win_rate = np.mean([e.win_rate for e in significant_edges.values()])
        else:
            top_sharpe = 0
            avg_win_rate = 0
        
        return f"""
        <div class="section">
            <h2>📈 Executive Summary</h2>
            <div class="stats-grid">
                <div class="stat-card neutral">
                    <div class="value">{total}</div>
                    <div class="label">Edges Tested</div>
                </div>
                <div class="stat-card positive">
                    <div class="value">{significant}</div>
                    <div class="label">Significant (p&lt;0.05)</div>
                </div>
                <div class="stat-card {'positive' if top_sharpe > 0 else 'neutral'}">
                    <div class="value">{top_sharpe:.2f}</div>
                    <div class="label">Best Sharpe Ratio</div>
                </div>
                <div class="stat-card {'positive' if avg_win_rate > 0.5 else 'negative'}">
                    <div class="value">{avg_win_rate:.1%}</div>
                    <div class="label">Avg Win Rate</div>
                </div>
            </div>
            
            <h3 style="margin: 20px 0 15px; color: #f39c12;">✅ Edge Validation Checklist</h3>
            <div class="checklist">
                <div class="checklist-item">
                    <div class="icon check">✓</div>
                    <span>Statistical significance (p-value < 0.05)</span>
                </div>
                <div class="checklist-item">
                    <div class="icon pending">?</div>
                    <span>Sample size > 100 trades</span>
                </div>
                <div class="checklist-item">
                    <div class="icon pending">?</div>
                    <span>Survives walk-forward validation</span>
                </div>
                <div class="checklist-item">
                    <div class="icon pending">?</div>
                    <span>Net positive after transaction costs</span>
                </div>
                <div class="checklist-item">
                    <div class="icon pending">?</div>
                    <span>Works on multiple symbols</span>
                </div>
                <div class="checklist-item">
                    <div class="icon pending">?</div>
                    <span>Stable across different time periods</span>
                </div>
            </div>
        </div>
"""
    
    def _build_edges_table(self, significant: Dict) -> str:
        """Build edges table."""
        if not significant:
            return """
        <div class="section">
            <h2>🎯 Significant Edges</h2>
            <p>No significant edges found in this analysis.</p>
        </div>
"""
        
        # Sort by Sharpe
        sorted_edges = sorted(significant.values(), 
                             key=lambda x: x.sharpe_ratio, reverse=True)
        
        rows = []
        for i, edge in enumerate(sorted_edges[:20], 1):
            sharpe_class = 'positive' if edge.sharpe_ratio > 0 else 'negative'
            win_class = 'positive' if edge.win_rate > 0.5 else 'negative'
            
            rows.append(f"""
                <tr>
                    <td>{i}</td>
                    <td>{edge.edge_name}</td>
                    <td>{edge.edge_type}</td>
                    <td>{edge.sample_size:,}</td>
                    <td class="{sharpe_class}">{edge.sharpe_ratio:.2f}</td>
                    <td class="{win_class}">{edge.win_rate:.1%}</td>
                    <td>{edge.mean_return:.4f}</td>
                    <td>{edge.p_value:.4f}</td>
                </tr>
""")
        
        return f"""
        <div class="section">
            <h2>🎯 Top Significant Edges</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Edge Name</th>
                        <th>Type</th>
                        <th>Sample Size</th>
                        <th>Sharpe</th>
                        <th>Win Rate</th>
                        <th>Mean Return</th>
                        <th>P-Value</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
"""
    
    def _build_visualizations(self, dashboard: Dict) -> str:
        """Build visualizations section."""
        charts = []
        
        chart_configs = [
            ('equity_curve', 'Equity Curve', 'Shows cumulative returns over time'),
            ('drawdown', 'Drawdown Analysis', 'Visualizes periods of decline from peak'),
            ('monthly_returns', 'Monthly Returns Heatmap', 'Returns broken down by month and year'),
            ('rolling_sharpe', 'Rolling Sharpe Ratio', '252-day rolling Sharpe ratio over time'),
            ('win_loss_dist', 'Win/Loss Distribution', 'Distribution of winning and losing trades'),
            ('edge_comparison', 'Edge Comparison', 'Top edges ranked by Sharpe ratio'),
            ('session_analysis', 'Session Analysis', 'Performance by trading session'),
            ('regime_analysis', 'Regime Analysis', 'Market regime identification')
        ]
        
        for key, title, desc in chart_configs:
            if key in dashboard:
                charts.append(f"""
            <div class="chart-container">
                <h3 style="margin-bottom: 10px; color: #00d2ff;">{title}</h3>
                <p style="color: #888; font-size: 0.9rem; margin-bottom: 15px;">{desc}</p>
                <img src="data:image/png;base64,{dashboard[key]}" alt="{title}">
            </div>
""")
        
        if not charts:
            return ""
        
        return f"""
        <div class="section">
            <h2>📊 Visualizations</h2>
            <div class="grid-2">
                {''.join(charts)}
            </div>
        </div>
"""
    
    def _build_categories(self, edge_results: Dict) -> str:
        """Build edge categories summary."""
        # Group by type
        by_type = {}
        for name, result in edge_results.items():
            edge_type = result.edge_type
            if edge_type not in by_type:
                by_type[edge_type] = {'total': 0, 'significant': 0}
            by_type[edge_type]['total'] += 1
            if result.is_significant:
                by_type[edge_type]['significant'] += 1
        
        rows = []
        for edge_type, counts in sorted(by_type.items(), 
                                        key=lambda x: x[1]['significant'], 
                                        reverse=True):
            pct = counts['significant'] / counts['total'] * 100 if counts['total'] > 0 else 0
            rows.append(f"""
                <tr>
                    <td>{edge_type.replace('_', ' ').title()}</td>
                    <td>{counts['total']}</td>
                    <td class="{'positive' if counts['significant'] > 0 else ''}">{counts['significant']}</td>
                    <td>{pct:.1f}%</td>
                </tr>
""")
        
        return f"""
        <div class="section">
            <h2>📁 Edge Categories</h2>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Total Tested</th>
                        <th>Significant</th>
                        <th>Success Rate</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
"""
    
    def _build_data_summary(self, df: pd.DataFrame) -> str:
        """Build data summary section."""
        date_range = f"{df.index[0]} to {df.index[-1]}"
        
        return f"""
        <div class="section">
            <h2>📋 Data Summary</h2>
            <div class="stats-grid">
                <div class="stat-card neutral">
                    <div class="value">{len(df):,}</div>
                    <div class="label">Total Bars</div>
                </div>
                <div class="stat-card neutral">
                    <div class="value">{len(df.columns)}</div>
                    <div class="label">Features</div>
                </div>
                <div class="stat-card neutral">
                    <div class="value">{date_range.split()[0]}</div>
                    <div class="label">Start Date</div>
                </div>
                <div class="stat-card neutral">
                    <div class="value">{date_range.split()[-1]}</div>
                    <div class="label">End Date</div>
                </div>
            </div>
        </div>
"""
    
    def _build_footer(self) -> str:
        """Build HTML footer."""
        return """
        <div class="footer">
            <p>Generated by MT5 Edge Extraction System v1.0.0</p>
            <p style="margin-top: 10px; color: #e74c3c;">
                ⚠️ FOR RESEARCH PURPOSES ONLY - NOT FINANCIAL ADICE
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    def save_report(self, html: str, filepath: str):
        """
        Save HTML report to file.
        
        Args:
            html: HTML content
            filepath: Output file path
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
