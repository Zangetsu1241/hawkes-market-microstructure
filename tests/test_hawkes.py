import json
import numpy as np
import pandas as pd
import pytest
from hawkes_micro.core import Hawkes,simulate,fit_hawkes,poisson_loglik,history_states
from hawkes_micro.research import Config,evaluate
from hawkes_micro.data import read_events
from hawkes_micro.cli import main


def params():
    return Hawkes(np.array([.5,.35]),np.array([[.45,.15],[.13,.4]]),1.)


def test_branching_and_simulation_reproducibility():
    p=params()
    assert 0<p.branching_ratio<1
    a,b=simulate(p,200,seed=3)
    c,d=simulate(p,200,seed=3)
    assert np.array_equal(a,c) and np.array_equal(b,d)
    assert len(a)>80 and len(a)<1000 and np.all(np.diff(a)>0)


def test_poisson_recovery_zero_excitation():
    mu=np.array([.2,.3]);p=Hawkes(mu,np.zeros((2,2)),1.)
    t=np.array([1.,2.,3.]);m=np.array([0,1,0])
    assert np.isclose(p.loglik(t,m,5),poisson_loglik(t,m,5,mu))


def test_likelihood_hand_calculation_single_type():
    p=Hawkes(np.array([.5]),np.array([[.3]]),2.)
    t=np.array([1.,2.]);m=np.array([0,0]);T=3.
    expected=(np.log(.5)+np.log(.5+.3*np.exp(-2))-.5*T-
              .3/2*((1-np.exp(-2*(T-1)))+(1-np.exp(-2*(T-2)))))
    assert np.isclose(p.loglik(t,m,T),expected)


def test_likelihood_continuation_matches_complete_window():
    p=params()
    t,m=simulate(p,40,seed=21)
    cut=20.
    a=p.loglik(t,m,40.)
    pre=t<cut
    _,state,last=history_states(t[pre],m[pre],p.beta,2)
    initial=state*np.exp(-p.beta*(cut-last))
    b=p.loglik(t[pre],m[pre],cut)+p.loglik(t[~pre],m[~pre],40,start=cut,initial=initial)
    assert np.isclose(a,b,atol=1e-9)


def test_time_rescaling_poisson_mean():
    p=Hawkes(np.array([.7]),np.zeros((1,1)),1.)
    t,m=simulate(p,3000,seed=3)
    z=p.transform_intervals(t,m)
    assert .9<z.mean()<1.1


def test_fit_is_stationary_and_finite():
    t,m=simulate(params(),110,seed=7)
    fit,summary=fit_hawkes(t,m,110,max_iter=55)
    assert np.all(fit.mu>0) and np.all(fit.alpha>=0)
    assert fit.branching_ratio<.981
    assert np.isfinite(summary['train_log_likelihood'])


def test_hawkes_better_on_simulated_large_sample():
    t,m=simulate(params(),600,seed=42)
    result,detail,model=evaluate(t,m,600,Config(max_iter=95))
    assert result['delta_heldout_loglik']>0
    assert len(detail)==result['sample']['test_events']


def test_future_events_do_not_change_fit():
    t,m=simulate(params(),220,seed=45)
    a,_,_=evaluate(t,m,220,Config(max_iter=50))
    split=220*.7
    t_new=np.r_[t[t<split], t[t>=split]+.01]
    b,_,_=evaluate(t_new,m,220,Config(max_iter=50))
    assert a['hawkes']['mu']==b['hawkes']['mu']
    assert a['hawkes']['alpha']==b['hawkes']['alpha']
    assert a['poisson']['mu']==b['poisson']['mu']


def test_input_validation_and_nonstationarity():
    with pytest.raises(ValueError):Hawkes([.2,.3],[[1.]],1.)
    with pytest.raises(ValueError):simulate(Hawkes([.5],[[1.1]],1.),10)
    with pytest.raises(ValueError):params().loglik([2.,1.],[0,1],5)
    with pytest.raises(ValueError):params().loglik([1.,1.],[0,1],5)
    with pytest.raises(ValueError):params().loglik([1.],[2],5)


def test_csv_ingestion(tmp_path):
    file=tmp_path/'events.csv'
    file.write_text('time,side\n0.2,buy\n0.7,sell\n')
    t,m=read_events(file)
    assert m.tolist()==[0,1]
    file.write_text('time,side\n0.7,buy\n0.2,sell\n')
    with pytest.raises(ValueError):read_events(file)


def test_predictive_mark_probabilities_sum_to_one():
    p=params()
    t,m=simulate(p,100,seed=11)
    states,_,_=history_states(t,m,p.beta,2)
    lamb=p.mu[None,:]+states@p.alpha.T
    assert np.allclose((lamb/lamb.sum(axis=1)[:,None]).sum(axis=1),1)


def test_cli_outputs(tmp_path):
    out=tmp_path/'results'
    summary=main(['--horizon','100','--max-iter','50','--output',str(out)])
    assert summary['sample']['test_events']>5
    for f in ['metrics.json','heldout_event_intensities.csv','diagnostics.png']:
        assert (out/f).exists()
    assert json.loads((out/'metrics.json').read_text())['dataset'].startswith('synthetic')
