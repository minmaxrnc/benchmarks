
from ..utils import config

class EmaStopper:
    def __init__(self):
        enabled   = config.get('stopper__enabled')
        mode      = config.get('stopper__mode')
        window    = config.get('stopper__window')
        patience  = config.get('stopper__patience')
        min_delta = config.get('stopper__min_delta')
        debias    = config.get('stopper__debias')

        self.enabled = enabled
        self.sign = -1 if mode == "min" else 1          # unify improvement check
        self.alpha = 2.0 / (float(window) + 1.0)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.debias = debias
        self.reset()

    def reset(self):
        self.s = None
        self.t = 0
        self.best = None
        self.no_improve = 0

    def update(self, metric):
        # metric is raw val metric for this epoch
        self.t += 1
        if self.s is None:
            self.s = metric
        else:
            self.s = self.alpha * metric + (1 - self.alpha) * self.s
        s_hat = self.s / (1 - (1 - self.alpha)**self.t) if self.debias else self.s

        improved = (self.sign * (s_hat - (self.best if self.best is not None else s_hat - 1e9)) >= self.min_delta)

        if improved or self.best is None:
            self.best = s_hat
            self.no_improve = 0
        else:
            self.no_improve += 1

        print(f"  Stopper: best={self.best}  noimprove={self.no_improve}/{self.patience}")

        return self.enabled and (self.no_improve >= self.patience)


    def get_stop(self):
        return self.enabled and (self.no_improve >= self.patience)



