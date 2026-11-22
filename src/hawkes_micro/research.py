"""Strictly historical estimation and continuous-time forward evaluation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import kstest
from .core import Hawkes, simulate, fit_hawkes, history_states, poisson_loglik, validate_events

@dataclass(frozen=True)
class Config:
    beta:float=1.
    train_fraction:float=.7
    max_iter:int=160
    n_types:int=2


def evaluate(times,marks,horizon,cfg=Config()):
    times,marks=validate_events(times,marks,horizon,cfg.n_types)
    if not .4<=cfg.train_fraction<=.9 or cfg.beta<=0 or cfg.n_types<1:
        raise ValueError('invalid configuration')
    split=horizon*cfg.train_fraction
    train=times<split
    t_train,m_train=times[train],marks[train]
    t_test,m_test=times[~train],marks[~train]
    if len(t_train)<max(10,3*cfg.n_types) or len(t_test)<5:
        raise ValueError('Need enough train and test events')
    model,fit=fit_hawkes(t_train,m_train,split,beta=cfg.beta,n_types=cfg.n_types,max_iter=cfg.max_iter)
    counts=np.bincount(m_train,minlength=cfg.n_types)
    poisson_rates=(counts+.5)/(split+1.)  # small smoothing prevents log(0)
    _,state,last=history_states(t_train,m_train,cfg.beta,cfg.n_types)
    initial=state*np.exp(-cfg.beta*(split-last)) if len(t_train) else np.zeros(cfg.n_types)
    hawkes_test=model.loglik(t_test,m_test,horizon,start=split,initial=initial)
    poisson_test=poisson_loglik(t_test,m_test,horizon,poisson_rates,start=split)
    states,_,_=history_states(t_test,m_test,cfg.beta,cfg.n_types,start=split,initial=initial)
    observed=model.mu[m_test]+np.einsum('ij,ij->i',model.alpha[m_test],states)
    total=model.mu.sum()+(states@model.alpha.T).sum(axis=1)
    mark_logloss=float(-np.log(observed/total).mean())
    pmark=-np.log(poisson_rates[m_test]/poisson_rates.sum()).mean()
    transforms=model.transform_intervals(t_test,m_test,start=split,initial=initial)
    ks=kstest(transforms,'expon')
    report={'sample':{'train_events':int(len(t_train)),'test_events':int(len(t_test)),
                      'train_horizon':float(split),'test_horizon':float(horizon-split)},
            'hawkes':{'mu':model.mu.tolist(),'alpha':model.alpha.tolist(),'beta':float(model.beta),
                      'branching_ratio':model.branching_ratio,'train_fit':fit,
                      'test_loglik':float(hawkes_test),'test_loglik_per_event':float(hawkes_test/len(t_test)),
                      'test_mark_nll':mark_logloss,'time_rescaling_ks_stat':float(ks.statistic),
                      'time_rescaling_ks_pvalue':float(ks.pvalue)},
            'poisson':{'mu':poisson_rates.tolist(),'test_loglik':float(poisson_test),
                       'test_loglik_per_event':float(poisson_test/len(t_test)),
                       'test_mark_nll':float(pmark)},
            'delta_heldout_loglik':float(hawkes_test-poisson_test),
            'interpretation':'One chronological split; parameters estimated before test. Time-rescaling KS is a descriptive diagnostic; parameters are fitted and marks are pooled.'}
    detail=pd.DataFrame({'time':t_test,'mark':m_test,'hawkes_intensity_observed':observed,
                         'hawkes_intensity_total':total,'poisson_intensity_observed':poisson_rates[m_test],
                         'rescaled_interarrival':transforms})
    return report,detail,model
