"""Development/holdout split and a single-use holdout vault."""
from datetime import datetime

from sklearn.model_selection import train_test_split

from credit_risk import settings


def split_holdout(frame, holdout_size=settings.HOLDOUT_SIZE, seed=settings.SEED):
    """Stratified split into a development part and a final holdout part."""
    dev, holdout = train_test_split(
        frame, test_size=holdout_size, stratify=frame[settings.TARGET],
        random_state=seed,
    )
    return dev.reset_index(drop=True), holdout.reset_index(drop=True)


class HoldoutVault:
    """Keeps the holdout frame and allows exactly one look at it."""

    def __init__(self, holdout, log_path):
        self._holdout = holdout
        self._log_path = log_path
        self._opened = False

    def open(self, reason):
        """Return the holdout once; log the access; refuse a second call."""
        if self._opened:
            raise RuntimeError("Holdout was already opened once")
        self._opened = True
        stamp = datetime.now().isoformat(timespec="seconds")
        with open(self._log_path, "a", encoding="utf-8") as handle:
            handle.write(f"{stamp} rows={len(self._holdout)} reason={reason}\n")
        return self._holdout.copy()
