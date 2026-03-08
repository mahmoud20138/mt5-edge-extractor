"""
Edge Visualization Module.

Provides visualization for:
- Equity curves
- Drawdown charts
- Edge performance comparison
- Monthly return heatmaps
- Rolling metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
from datetime import datetime
import io
import base64


class EdgeVisualizer:
    """
    Create visualizations for edge analysis.
    
    Provides methods for:
    - Equity curve plots
    - Drawdown charts
    - Performance heatmaps
    - Rolling metrics charts
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid',
                 figsize: Tuple[int, int] = (12, 6),
                 dpi: int = 150):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
            dpi: Resolution
        """
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        # Try to set style, fall back to default if not available
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid')
        
        # Color palette
        self.colors = {
            'profit': '#2ecc71',
            'loss': '#e74c3c',
            'neutral': '#3498db',
            'highlight': '#f39c12',
            'background': '#2c3e50'
        }
    
    def plot_equity_curve(self, returns: pd.Series,
                          title: str = "Equity Curve",
                          benchmark: pd.Series = None) -> Figure:
        """
        Plot equity curve from returns.
        
        Args:
            returns: Series of returns
            title: Chart title
            benchmark: Optional benchmark returns
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate equity curve
        equity = (1 + returns).cumprod()
        
        # Plot equity
        ax.plot(equity.index, equity.values, 
                color=self.colors['neutral'], linewidth=1.5,
                label='Strategy')
        
        # Plot benchmark if provided
        if benchmark is not None:
            bench_equity = (1 + benchmark).cumprod()
            ax.plot(bench_equity.index, bench_equity.values,
                   color=self.colors['neutral'], linewidth=1, alpha=0.5,
                   linestyle='--', label='Benchmark')
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Equity', fontsize=11)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # Add performance annotation
        total_return = equity.iloc[-1] - 1
        color = self.colors['profit'] if total_return > 0 else self.colors['loss']
        ax.annotate(f'Total Return: {total_return:.1%}',
                   xy=(0.02, 0.95), xycoords='axes fraction',
                   fontsize=11, color=color, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown(self, returns: pd.Series,
                      title: str = "Drawdown Analysis") -> Figure:
        """
        Plot drawdown chart.
        
        Args:
            returns: Series of returns
            title: Chart title
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate equity and drawdown
        equity = (1 + returns).cumprod()
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        
        # Plot drawdown
        ax.fill_between(drawdown.index, 0, drawdown.values,
                       color=self.colors['loss'], alpha=0.3)
        ax.plot(drawdown.index, drawdown.values,
               color=self.colors['loss'], linewidth=1)
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Drawdown', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # Add max drawdown annotation
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax.annotate(f'Max DD: {max_dd:.1%}',
                   xy=(max_dd_date, max_dd),
                   xytext=(10, 20), textcoords='offset points',
                   fontsize=10, color=self.colors['loss'],
                   arrowprops=dict(arrowstyle='->', color=self.colors['loss']))
        
        plt.tight_layout()
        return fig
    
    def plot_monthly_returns(self, returns: pd.Series,
                            title: str = "Monthly Returns Heatmap") -> Figure:
        """
        Plot monthly returns heatmap.
        
        Args:
            returns: Series of returns
            title: Chart title
            
        Returns:
            Matplotlib Figure
        """
        # Resample to monthly
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table
        df = pd.DataFrame({'return': monthly})
        df['year'] = df.index.year
        df['month'] = df.index.month
        
        pivot = df.pivot(index='year', columns='month', values='return')
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = month_names[:len(pivot.columns)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.8)), dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn',
                   center=0, linewidths=0.5, ax=ax,
                   cbar_kws={'label': 'Return'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=11)
        ax.set_ylabel('Year', fontsize=11)
        
        plt.tight_layout()
        return fig
    
    def plot_rolling_sharpe(self, returns: pd.Series,
                           window: int = 252,
                           title: str = "Rolling Sharpe Ratio") -> Figure:
        """
        Plot rolling Sharpe ratio.
        
        Args:
            returns: Series of returns
            window: Rolling window
            title: Chart title
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate rolling Sharpe
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        
        # Plot
        ax.plot(rolling_sharpe.index, rolling_sharpe.values,
               color=self.colors['neutral'], linewidth=1.5)
        
        # Add reference lines
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.axhline(y=1, color=self.colors['profit'], linestyle='--', 
                  linewidth=1, alpha=0.7, label='Sharpe = 1')
        ax.axhline(y=-1, color=self.colors['loss'], linestyle='--',
                  linewidth=1, alpha=0.7, label='Sharpe = -1')
        
        # Fill areas
        ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                       where=(rolling_sharpe > 0),
                       color=self.colors['profit'], alpha=0.2)
        ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                       where=(rolling_sharpe < 0),
                       color=self.colors['loss'], alpha=0.2)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Sharpe Ratio', fontsize=11)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_win_loss_distribution(self, returns: pd.Series,
                                   title: str = "Win/Loss Distribution") -> Figure:
        """
        Plot distribution of winning and losing trades.
        
        Args:
            returns: Series of returns
            title: Chart title
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        # Plot histograms
        bins = np.linspace(returns.min(), returns.max(), 50)
        
        ax.hist(wins, bins=bins, color=self.colors['profit'], alpha=0.7,
               label=f'Wins ({len(wins)})', edgecolor='white', linewidth=0.5)
        ax.hist(losses, bins=bins, color=self.colors['loss'], alpha=0.7,
               label=f'Losses ({len(losses)})', edgecolor='white', linewidth=0.5)
        
        # Add statistics
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        stats_text = f'Win Rate: {win_rate:.1%}\nAvg Win: {avg_win:.2%}\nAvg Loss: {avg_loss:.2%}'
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Return', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        plt.tight_layout()
        return fig
    
    def plot_edge_comparison(self, edge_results: Dict,
                            metric: str = 'sharpe_ratio',
                            top_n: int = 15,
                            title: str = "Top Edges Comparison") -> Figure:
        """
        Plot comparison of top edges.
        
        Args:
            edge_results: Dictionary of edge results
            metric: Metric to compare
            top_n: Number of top edges to show
            title: Chart title
            
        Returns:
            Matplotlib Figure
        """
        # Filter and sort
        significant = {k: v for k, v in edge_results.items() if v.is_significant}
        
        if not significant:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No significant edges found',
                   ha='center', va='center', fontsize=14)
            return fig
        
        sorted_edges = sorted(significant.values(), 
                             key=lambda x: getattr(x, metric, 0),
                             reverse=True)[:top_n]
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_edges) * 0.4)), dpi=self.dpi)
        
        names = [e.edge_name[:30] for e in sorted_edges]
        values = [getattr(e, metric, 0) for e in sorted_edges]
        
        # Color based on value
        colors = [self.colors['profit'] if v > 0 else self.colors['loss'] for v in values]
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(names)), values, color=colors, edgecolor='white')
        
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_session_analysis(self, df: pd.DataFrame,
                              title: str = "Performance by Trading Session") -> Figure:
        """
        Plot performance by trading session.
        
        Args:
            df: DataFrame with session features and returns
            title: Chart title
            
        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)
        
        # Session returns
        if 'session' in df.columns and 'return_1' in df.columns:
            session_returns = df.groupby('session')['return_1'].mean()
            session_counts = df.groupby('session').size()
            
            sessions = ['asian', 'london', 'ny', 'other']
            session_labels = ['Asian', 'London', 'New York', 'Other']
            
            colors = [self.colors['profit'] if session_returns.get(s, 0) > 0 
                     else self.colors['loss'] for s in sessions]
            
            ax1 = axes[0]
            bars = ax1.bar(session_labels, 
                          [session_returns.get(s, 0) * 100 for s in sessions],
                          color=colors, edgecolor='white')
            
            ax1.set_title('Average Return by Session', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Session', fontsize=11)
            ax1.set_ylabel('Average Return (%)', fontsize=11)
            ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add count labels
            for bar, session in zip(bars, sessions):
                count = session_counts.get(session, 0)
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # Hourly returns
        if 'hour' in df.columns and 'return_1' in df.columns:
            hourly_returns = df.groupby('hour')['return_1'].mean()
            
            ax2 = axes[1]
            colors = [self.colors['profit'] if v > 0 else self.colors['loss'] 
                     for v in hourly_returns.values]
            
            ax2.bar(hourly_returns.index, hourly_returns.values * 100,
                   color=colors, edgecolor='white', width=0.8)
            
            ax2.set_title('Average Return by Hour', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Hour (UTC)', fontsize=11)
            ax2.set_ylabel('Average Return (%)', fontsize=11)
            ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Highlight sessions
            ax2.axvspan(0, 8, alpha=0.1, color='blue', label='Asian')
            ax2.axvspan(7, 16, alpha=0.1, color='green', label='London')
            ax2.axvspan(12, 21, alpha=0.1, color='red', label='NY')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_regime_analysis(self, df: pd.DataFrame,
                             title: str = "Market Regime Analysis") -> Figure:
        """
        Plot market regime analysis.
        
        Args:
            df: DataFrame with regime features
            title: Chart title
            
        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi, 
                                sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Price and regime
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], color=self.colors['neutral'], 
                linewidth=1, label='Price')
        
        # Add ADX if available
        if 'adx' in df.columns:
            ax1_twin = ax1.twinx()
            ax1_twin.plot(df.index, df['adx'], color=self.colors['highlight'],
                         linewidth=1, alpha=0.7, label='ADX')
            ax1_twin.axhline(y=25, color='gray', linestyle='--', linewidth=0.5)
            ax1_twin.set_ylabel('ADX', fontsize=11)
            ax1_twin.legend(loc='upper right')
        
        ax1.set_title('Price with Trend Indicator', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volatility regime
        ax2 = axes[1]
        if 'vol_percentile' in df.columns:
            ax2.fill_between(df.index, 0, df['vol_percentile'],
                           where=df['vol_percentile'] < 25,
                           color=self.colors['profit'], alpha=0.5, label='Low Vol')
            ax2.fill_between(df.index, 0, df['vol_percentile'],
                           where=df['vol_percentile'] > 75,
                           color=self.colors['loss'], alpha=0.5, label='High Vol')
            ax2.fill_between(df.index, 0, df['vol_percentile'],
                           where=(df['vol_percentile'] >= 25) & (df['vol_percentile'] <= 75),
                           color=self.colors['neutral'], alpha=0.5, label='Normal Vol')
            
            ax2.set_ylabel('Vol Percentile', fontsize=11)
            ax2.legend(loc='upper left', fontsize=8)
        
        ax2.set_title('Volatility Regime', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def fig_to_base64(self, fig: Figure) -> str:
        """
        Convert figure to base64 string.
        
        Args:
            fig: Matplotlib Figure
            
        Returns:
            Base64 encoded string
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def save_fig(self, fig: Figure, filepath: str):
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib Figure
            filepath: Output file path
        """
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
    
    def create_dashboard(self, returns: pd.Series,
                        edge_results: Dict,
                        df: pd.DataFrame = None) -> Dict[str, str]:
        """
        Create a complete visualization dashboard.
        
        Args:
            returns: Strategy returns
            edge_results: Edge analysis results
            df: Full DataFrame with features
            
        Returns:
            Dictionary of base64-encoded images
        """
        dashboard = {}
        
        # Equity curve
        fig = self.plot_equity_curve(returns)
        dashboard['equity_curve'] = self.fig_to_base64(fig)
        
        # Drawdown
        fig = self.plot_drawdown(returns)
        dashboard['drawdown'] = self.fig_to_base64(fig)
        
        # Monthly returns
        fig = self.plot_monthly_returns(returns)
        dashboard['monthly_returns'] = self.fig_to_base64(fig)
        
        # Rolling Sharpe
        fig = self.plot_rolling_sharpe(returns)
        dashboard['rolling_sharpe'] = self.fig_to_base64(fig)
        
        # Win/Loss distribution
        fig = self.plot_win_loss_distribution(returns)
        dashboard['win_loss_dist'] = self.fig_to_base64(fig)
        
        # Edge comparison
        fig = self.plot_edge_comparison(edge_results)
        dashboard['edge_comparison'] = self.fig_to_base64(fig)
        
        # Session analysis
        if df is not None and 'session' in df.columns:
            fig = self.plot_session_analysis(df)
            dashboard['session_analysis'] = self.fig_to_base64(fig)
        
        # Regime analysis
        if df is not None and 'adx' in df.columns:
            fig = self.plot_regime_analysis(df)
            dashboard['regime_analysis'] = self.fig_to_base64(fig)
        
        return dashboard
