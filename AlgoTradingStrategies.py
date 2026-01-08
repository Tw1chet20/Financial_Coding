# StrategyPractice.py

from backtesting import Strategy
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

TARGET_COL = "Target Up"
DROP_COLS = [
    "Open", "High", "Low", "Close", "Adj Close",
    "EMA_2", "EMA_5", "EMA_10", "Change Tomorrow"
    ]

class MLBaseStrategy(Strategy):
    buy_prob = 0.55
    sell_prob = 0.45

    n_train = 150
    retrain_every = 50

    disable_retrain = False

    def init(self):
        df = self.data.df
        if len(df) <= self.n_train:
            raise ValueError(f"Not enough rows to train: have {len(df)}, need > {self.n_train}")

        self._feature_cols = sorted([c for c in df.columns if (c != TARGET_COL and c not in DROP_COLS)])

        self.model = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=5,
            max_features="sqrt"
        )

        self._fit_model(end_index=self.n_train, start_index=0, do_select=True)

    def log_feature_importance(self):
        imp = pd.Series(
            self.model.feature_importances_,
            index=self._feature_cols
        ).sort_values(ascending=False)

        important = imp.head(15).index.tolist()
        self._feature_cols = sorted(important)

    def _fit_model(self, end_index: int, start_index: int = 0, do_select: bool = False):
        df = self.data.df
        X = df[self._feature_cols].iloc[start_index:end_index]
        y = df[TARGET_COL].iloc[start_index:end_index]
        self.model.fit(X, y)
        if do_select:
            self.log_feature_importance()

    def _predict_proba_up(self, idx: int) -> float:
        row = self.data.df[self._feature_cols].iloc[[idx]]
        return float(self.model.predict_proba(row)[0][1])

    def _current_index(self) -> int:
        return len(self.data) - 1

    def next(self):
        i = self._current_index()
        if i < self.n_train:
            return

        p_up = self._predict_proba_up(i)

        if not self.position and p_up > self.buy_prob:
            self.buy()
        elif self.position and p_up < self.sell_prob:
            self.position.close()


class WalkForwardAnchored(MLBaseStrategy):
    def next(self):
        i = self._current_index()
        if i < self.n_train:
            return

        if not self.disable_retrain:
            if (i - self.n_train) % self.retrain_every == 0:
                self._fit_model(end_index=i, start_index=0, do_select=True)

        super().next()


class WalkForwardUnanchored(MLBaseStrategy):
    def next(self):
        i = self._current_index()
        if i < self.n_train:
            return

        if not self.disable_retrain:
            if (i - self.n_train) % self.retrain_every == 0:
                start = max(0, i - self.n_train)
                self._fit_model(end_index=i, start_index=start, do_select=True)

        super().next()