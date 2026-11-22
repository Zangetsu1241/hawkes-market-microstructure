"""Exponential-kernel multivariate Hawkes processes.

Intensity lambda_i(t) = mu_i + sum_j alpha[i,j] sum_{t_k^j < t} exp(-beta*(t-t_k^j)).
The branching matrix is alpha/beta, with spectral radius <1 for stationarity.
Timestamps must be strictly increasing (ties are not silently reordered).
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def validate_events(times, marks, horizon=None, n_types=2):
    t=np.asarray(times,dtype=float)
    m=np.asarray(marks)
    if t.ndim!=1 or m.shape!=t.shape or not np.isfinite(t).all():
        raise ValueError('times and marks must be finite 1D aligned arrays')
    if (t<0).any() or (np.diff(t)<=0).any():
        raise ValueError('timestamps must be nonnegative and strictly increasing')
    if not np.issubdtype(m.dtype, np.integer) or (m<0).any() or (m>=n_types).any():
        raise ValueError('marks must be integer indices 0..n_types-1')
    if horizon is not None and (not np.isfinite(horizon) or horizon<=0 or (len(t) and t[-1]>=horizon)):
        raise ValueError('horizon must exceed last timestamp and be positive')
    return t,m.astype(int)


def history_states(times,marks,beta,n_types=2,start=0.,initial=None):
    """Just-before-event exponential sums, preserving past history across windows."""
    times,marks=validate_events(times,marks,n_types=n_types)
    if not np.isfinite(beta) or beta<=0 or start<0:
        raise ValueError('invalid beta or start')
    state=np.zeros(n_types) if initial is None else np.array(initial,dtype=float).copy()
    if state.shape!=(n_types,) or not np.isfinite(state).all() or (state<0).any():
        raise ValueError('invalid initial excitation')
    out=np.zeros((len(times),n_types))
    last=start
    for i,(t,j) in enumerate(zip(times,marks)):
        if t<last:raise ValueError('times precede window start')
        state*=np.exp(-beta*(t-last))
        out[i]=state
        state[j]+=1
        last=t
    return out,state,last


@dataclass(frozen=True)
class Hawkes:
    mu: np.ndarray
    alpha: np.ndarray
    beta: float

    def __post_init__(self):
        mu=np.asarray(self.mu,float)
        a=np.asarray(self.alpha,float)
        if (mu.ndim!=1 or len(mu)<1 or a.shape!=(len(mu),len(mu)) or
            not np.isfinite(mu).all() or not np.isfinite(a).all() or
            (mu<=0).any() or (a<0).any() or not np.isfinite(self.beta) or self.beta<=0):
            raise ValueError('mu must be positive, alpha nonnegative and beta positive')
        object.__setattr__(self,'mu',mu.copy())
        object.__setattr__(self,'alpha',a.copy())

    @property
    def branching_ratio(self):
        return float(max(abs(np.linalg.eigvals(self.alpha/self.beta))))

    def loglik(self,times,marks,end,start=0.,initial=None):
        """Point process conditional log-likelihood, including compensator."""
        times,marks=validate_events(times,marks,end,len(self.mu))
        if not 0<=start<end or (len(times) and times[0]<start):raise ValueError('invalid likelihood window')
        states,_,_=history_states(times,marks,self.beta,len(self.mu),start,initial)
        rates=self.mu[marks]+np.einsum('ij,ij->i',self.alpha[marks],states)
        if np.any(rates<=0):return -np.inf
        duration=end-start
        compensation=self.mu.sum()*duration
        if initial is not None:
            compensation+=np.dot(self.alpha@np.asarray(initial,float),np.ones(len(self.mu)))*(1-np.exp(-self.beta*duration))/self.beta
        if len(times):
            # Each event from mark j contributes to all target intensities i after its arrival.
            tails=(1-np.exp(-self.beta*(end-times)))/self.beta
            compensation+=float(np.dot(self.alpha.sum(axis=0)[marks],tails))
        return float(np.log(rates).sum()-compensation)

    def transform_intervals(self,times,marks,start=0.,initial=None):
        """Compensator increments for pooled event stream, with all mark intensities summed."""
        times,marks=validate_events(times,marks,n_types=len(self.mu))
        state=np.zeros(len(self.mu)) if initial is None else np.array(initial,dtype=float).copy()
        last=start
        intervals=[]
        for t,j in zip(times,marks):
            if t<last:raise ValueError('event before start')
            dt=t-last
            z=self.mu.sum()*dt+(self.alpha@state).sum()*(-np.expm1(-self.beta*dt))/self.beta
            intervals.append(z)
            state*=np.exp(-self.beta*dt)
            state[j]+=1
            last=t
        return np.asarray(intervals)


def simulate(params, horizon, seed=42, max_events=200000):
    """Exact Ogata thinning for exponential kernels with shared beta."""
    if not isinstance(params,Hawkes) or params.branching_ratio>=1:
        raise ValueError('simulation requires stationary Hawkes parameters')
    if not np.isfinite(horizon) or horizon<=0 or max_events<1:
        raise ValueError('invalid simulation arguments')
    rng=np.random.default_rng(seed)
    state=np.zeros(len(params.mu))
    times=[]; marks=[];t=0.
    while t<horizon:
        upper=float(params.mu.sum()+(params.alpha@state).sum())
        candidate=t+rng.exponential(1/upper)
        if candidate>=horizon:break
        state*=np.exp(-params.beta*(candidate-t))
        rates=params.mu+params.alpha@state
        total=float(rates.sum())
        if rng.random()*upper<=total:
            j=int(rng.choice(len(rates),p=rates/total))
            times.append(candidate);marks.append(j)
            state[j]+=1
            if len(times)>=max_events:raise RuntimeError('Exceeded max events; check parameters')
        t=candidate
    return np.asarray(times,float),np.asarray(marks,int)


def fit_hawkes(times,marks,horizon,beta=1.,n_types=2,max_iter=160):
    """Constrained maximum likelihood; alpha column sums < beta enforces stability.

    Parameterizing alpha = beta * p, with 0 <= p_ij and sum_i p_ij <= .98,
    makes each column sum < beta, ensuring a bound on spectral radius.
    """
    times,marks=validate_events(times,marks,horizon,n_types)
    if len(times)<max(10,n_types*3) or beta<=0 or max_iter<1:raise ValueError('insufficient events or invalid fit parameters')
    n=n_types
    observation=times
    state,_,_=history_states(times,marks,beta,n)
    tails=(-np.expm1(-beta*(horizon-times)))/beta
    counts=np.bincount(marks,minlength=n)
    rate=np.maximum(counts/horizon,.001)
    def unpack(theta):
        mu=np.exp(np.clip(theta[:n],-25,15))
        raw=theta[n:].reshape(n,n)
        # positive entries + slack via softmax over n+1 categories per source
        raw=np.vstack([raw,np.zeros(n)])
        ex=np.exp(raw-raw.max(axis=0,keepdims=True))
        p=.98*ex[:-1]/ex.sum(axis=0,keepdims=True)
        return mu,beta*p
    def objective(theta):
        mu,alpha=unpack(theta)
        vals=mu[marks]+np.einsum('ij,ij->i',alpha[marks],state)
        integral=(mu.sum()*horizon+np.dot(alpha.sum(axis=0)[marks],tails))
        return float(integral-np.log(vals).sum())
    # Multiple deterministic starts improve convergence on small datasets.
    best=None
    for initial_alpha in [-3.,-1.]:
        x0=np.r_[np.log(rate),np.full(n*n,initial_alpha)]
        result=minimize(objective,x0,method='L-BFGS-B',options={'maxiter':max_iter,'ftol':1e-10})
        if best is None or result.fun<best.fun:best=result
    mu,alpha=unpack(best.x)
    return Hawkes(mu,alpha,beta),{'converged':bool(best.success),'iterations':int(best.nit),
                                 'train_log_likelihood':float(-best.fun)}


def poisson_loglik(times,marks,horizon,rates,start=0.):
    t,m=validate_events(times,marks,n_types=len(rates))
    rate=np.asarray(rates,float)
    if len(t) and (t[0]<start or t[-1]>=horizon):raise ValueError('outside evaluation interval')
    if np.any(rate<=0):raise ValueError('rates must be positive')
    return float(np.log(rate[m]).sum()-rate.sum()*(horizon-start))
