"""Reproducible Hawkes microstructure research command line interface."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .core import Hawkes,simulate
from .data import read_events
from .research import Config,evaluate


def main(argv=None):
    p=argparse.ArgumentParser(description='Event clustering: bivariate Hawkes versus Poisson')
    p.add_argument('--csv',help='Time-in-seconds, side (buy or sell) CSV')
    p.add_argument('--horizon',type=float,default=1000.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--beta',type=float,default=1.)
    p.add_argument('--max-iter',type=int,default=160)
    p.add_argument('--output',default='results')
    a=p.parse_args(argv)
    if a.csv:
        t,m=read_events(a.csv)
    else:
        truth=Hawkes(np.array([.5,.35]),np.array([[.45,.15],[.13,.40]]),a.beta)
        t,m=simulate(truth,a.horizon,seed=a.seed)
    cfg=Config(beta=a.beta,max_iter=a.max_iter)
    report,preds,model=evaluate(t,m,a.horizon,cfg)
    report['dataset']='CSV events' if a.csv else 'synthetic exponential Hawkes'
    report['configuration']=vars(a)
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    preds.to_csv(out/'heldout_event_intensities.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    fig,axs=plt.subplots(2,1,figsize=(11,7))
    axs[0].plot(preds.time,preds.hawkes_intensity_total,label='Hawkes conditional intensity')
    axs[0].axhline(sum(report['poisson']['mu']),color='gray',linestyle='--',label='Poisson constant intensity')
    axs[0].set(ylabel='Events / second',title='Test period intensity (observed-event instants)')
    axs[0].legend()
    axs[1].hist(preds.rescaled_interarrival,bins=25,density=True,alpha=.6,label='Hawkes time-rescaled intervals')
    x=np.linspace(0,max(5.,float(np.quantile(preds.rescaled_interarrival,.99))),200)
    axs[1].plot(x,np.exp(-x),label='Unit exponential reference')
    axs[1].set(xlabel='Rescaled interarrival time',ylabel='Density')
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(out/'diagnostics.png',dpi=150)
    plt.close(fig)
    print(json.dumps({'hawkes_test_loglik':report['hawkes']['test_loglik'],
                      'poisson_test_loglik':report['poisson']['test_loglik'],
                      'branching_ratio':report['hawkes']['branching_ratio']},indent=2))
    return report

if __name__=='__main__':main()
