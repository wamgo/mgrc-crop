import numpy as np, unittest
import hcrc
PARENT=np.array([0,0,1,0,1,2,2,3,3])
def synth(n,rng,temp=1.0):
    y=rng.integers(0,9,n);logit=rng.normal(0,1,(n,9));logit[np.arange(n),y]+=2.0
    logit[:,:]+=0.8*(PARENT[None,:]==PARENT[y][:,None])  # within-parent confusion
    p=np.exp(logit/temp);return p/p.sum(1,keepdims=True),y
class T(unittest.TestCase):
    def test_decomposition_exact(self):
        rng=np.random.default_rng(0);p,y=synth(5000,rng)
        for tp,tf in [(0.3,0.6),(0.5,0.5),(0.2,0.9),(0.7,0.4)]:
            d,g,f=hcrc.answer_levelwise(p,PARENT,tp,tf)
            gg,sg,ff,sf=hcrc.top_down(p,PARENT);tf2=max(tf,tp)
            lc=(sg>=tp)&(gg!=PARENT[y]);lf=(sf>=tf2)&(gg==PARENT[y])&(ff!=y)
            self.assertTrue(np.array_equal(hcrc.hier_error(d,g,f,y,PARENT),lc|lf));self.assertFalse((lc&lf).any())
    def test_validity_levelwise_climb_cp(self):
        rng=np.random.default_rng(1);pt,yt=synth(20000,rng);risks={'lw':[],'climb':[],'cp':[],'flat':[]}
        for r in range(300):
            p,y=synth(100,rng)
            c=hcrc.calibrate_levelwise(p,y,PARENT,.03,.07);risks['lw'].append(hcrc.summarize(*hcrc.answer_levelwise(pt,PARENT,c['t_parent'],c['t_fine']),yt,PARENT)['risk'])
            c=hcrc.calibrate_climb(p,y,PARENT,.1);risks['climb'].append(hcrc.summarize(*hcrc.answer_climb(pt,PARENT,c['theta']),yt,PARENT)['risk'])
            c=hcrc.calibrate_cp(p,y,.1);risks['cp'].append(hcrc.summarize(*hcrc.answer_cp_lca(pt,PARENT,c['q']),yt,PARENT)['risk'])
            c=hcrc.calibrate_flat(p,y,PARENT,.1);risks['flat'].append(hcrc.summarize(*hcrc.answer_flat(pt,PARENT,c['t']),yt,PARENT)['risk'])
        for k,v in risks.items():self.assertLessEqual(np.mean(v),0.1+0.004,k)
    def test_cp_lca_error_implies_miscoverage(self):
        rng=np.random.default_rng(2);p,y=synth(3000,rng)
        for q in [0.5,0.8,0.95]:
            d,g,f=hcrc.answer_cp_lca(p,PARENT,q);err=hcrc.hier_error(d,g,f,y,PARENT);cov=(1-p[np.arange(len(y)),y])<=q
            self.assertFalse((err&cov).any())
    def test_path_monotone_and_valid(self):
        rng=np.random.default_rng(3);ps,ys=synth(5000,rng);path=hcrc.path_from_source(ps,ys,PARENT)
        self.assertTrue(np.all(np.diff(path[:-1],axis=0)>=0))
        pt,yt=synth(20000,rng);rs=[]
        for r in range(200):
            p,y=synth(100,rng);c=hcrc.calibrate_path(p,y,PARENT,path,.1)
            rs.append(hcrc.summarize(*hcrc.answer_levelwise(pt,PARENT,c['t_parent'],c['t_fine']),yt,PARENT)['risk'])
        self.assertLessEqual(np.mean(rs),0.104)
if __name__=='__main__':unittest.main()
