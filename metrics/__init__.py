"""
Edge Metrics Module.

Provides calculations for:
- Per-trade metrics
- Portfolio metrics
- Stability metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TradeMetrics:
    """Container for trade-level metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    expectancy: float
    profit_factor: float
    max_consecutive_wins: int
    max_consecutive_losses: int


@dataclass
class PortfolioMetrics:
    """Container for portfolio-level metrics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    recovery_factor: float
    ulcer_index: float


class EdgeMetrics:
    """
    Calculate edge performance metrics.
    
    Provides methods for:
    - Trade-level metrics
    - Portfolio-level metrics
    - Stability analysis
    """
    
    @staticmethod
    def calculate_trade_metrics(returns: pd.Series) -> TradeMetrics:
        """
        Calculate trade-level metrics.
        
        Args:
            returns: Series of trade returns
            
        Returns:
            TradeMetrics object
        """
        returns = returns.dropna()
        
        if len(returns) == 0:
            return TradeMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Consecutive wins/losses
        is_win = (returns > 0).astype(int)
        
        def max_consecutive(arr):
            groups = (arr != arr.shift()).cumsum()
            return arr.groupby(groups).sum().max() if len(arr) > 0 else 0
        
        max_cons_wins = int(max_consecutive(is_win))
        max_cons_losses = int(max_consecutive(1 - is_win))
        
        return TradeMetrics(
            total_trades=len(returns),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=win_loss_ratio,
            expectancy=expectancy,
            profit_factor=profit_factor,
            max_consecutive_wins=max_cons_wins,
            max_consecutive_losses=max_cons_losses
        )
    
    @staticmethod
    def calculate_portfolio_metrics(returns: pd.Series,
                                    periods_per_year: int = 252) -> PortfolioMetrics:
        """
        Calculate portfolio-level metrics.
        
        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year
            
        Returns:
            PortfolioMetrics object
        """
        returns = returns.dropna()
        
        if len(returns) == 0:
            return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Annualized return
        n_periods = len(returns)
        years = n_periods / periods_per_year
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe ratio
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = (mean_ret * periods_per_year) / (std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino = (mean_ret * periods_per_year) / (downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0
        
        # Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = abs(drawdowns.min())
        
        # Max drawdown duration
        dd_days = 0
        max_dd_duration = 0
        in_drawdown = False
        
        for dd in drawdowns:
            if dd < 0:
                in_drawdown = True
                dd_days += 1
                max_dd_duration = max(max_dd_duration, dd_days)
            else:
                in_drawdown = False
                dd_days = 0
        
        # Recovery factor
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Calmar ratio
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Ulcer index
        ulcer_index = np.sqrt((drawdowns ** 2).mean())
        
        return PortfolioMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index
        )
    
    @staticmethod
    def calculate_stability_metrics(returns: pd.Series,
                                    window: int = 252) -> Dict[str, float]:
        """
        Calculate stability metrics.
        
        Args:
            returns: Series of returns
            window: Rolling window for stability
            
        Returns:
            Dictionary with stability metrics
        """
        returns = returns.dropna()
        
        if len(returns) < window:
            return {}
        
        # Rolling Sharpe
        rolling_sharpe = returns.rolling(window).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        
        # Rolling win rate
        rolling_win_rate = returns.rolling(window).apply(lambda x: (x > 0).mean())
        
        # Year by year breakdown
        returns_df = pd.DataFrame({'return': returns})
        returns_df['year'] = returns_df.index.year
        yearly_returns = returns_df.groupby('year')['return'].sum()
        
        return {
            'rolling_sharpe_mean': rolling_sharpe.mean(),
            'rolling_sharpe_std': rolling_sharpe.std(),
            'rolling_sharpe_min': rolling_sharpe.min(),
            'rolling_win_rate_mean': rolling_win_rate.mean(),
            'yearly_return_std': yearly_returns.std(),
            'positive_years': (yearly_returns > 0).sum(),
            'negative_years': (yearly_returns < 0).sum(),
            'best_year': yearly_returns.max(),
            'worst_year': yearly_returns.min()
        }
    
    @staticmethod
    def transaction_cost_adjustment(gross_return: float,
                                   spread: float,
                                   commission: float = 0,
                                   slippage: float = 0) -> float:
        """
        Calculate net return after transaction costs.
        
        Args:
            gross_return: Gross return per trade
            spread: Spread in price units
            commission: Commission per trade
            slippage: Expected slippage
            
        Returns:
            Net return after costs
        """
        total_cost = spread + commission + slippage
        return gross_return - total_cost
    
    @staticmethod
    def breakeven_win_rate(avg_win: float, avg_loss: float) -> float:
        """
        Calculate breakeven win rate.
        
        Args:
            avg_win: Average winning trade
            avg_loss: Average losing trade (positive value)
            
        Returns:
            Win rate needed to break even
        """
        if avg_win + avg_loss == 0:
            return 0.5
        
        return avg_loss / (avg_win + avg_loss)
    
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly criterion for position sizing.
        
        Args:
            win_rate: Probability of winning
            avg_win: Average winning trade
            avg_loss: Average losing trade (positive value)
            
        Returns:
            Optimal position size as fraction
        """
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss  # Win/loss ratio
        q = 1 - win_rate
        
        kelly = (win_rate * b - q) / b
        
        return max(0, kelly)  # Never negative position
    
    @staticmethod
    def risk_adjusted_metrics(returns: pd.Series,
                             risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate risk-adjusted metrics.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary with risk-adjusted metrics
        """
        returns = returns.dropna()
        
        if len(returns) == 0:
            return {}
        
        mean_ret = returns.mean() * 252
        std_ret = returns.std() * np.sqrt(252)
        
        # Excess return
        excess_return = mean_ret - risk_free_rate
        
        # Sharpe
        sharpe = excess_return / std_ret if std_ret > 0 else 0
        
        # Treynor (would need beta)
        # Information ratio (would need benchmark)
        
        # Modigliani-Modigliani (M2)
        benchmark_std = 0.15  # Assume 15% benchmark vol
        m2 = sharpe * benchmark_std + risk_free_rate
        
        return {
            'excess_return': excess_return,
            'sharpe_ratio': sharpe,
            'm2_measure': m2,
            'volatility': std_ret
        }
