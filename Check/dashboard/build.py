"""Rebuilds grid_stress_dashboard.html from Check/fixed/processed + Data/processed (read only).
Run after f09_forecast_fixed.py:  python Check/dashboard/build.py"""
import json, math, numpy as np, pandas as pd
from pathlib import Path
U=str(Path(__file__).resolve().parents[2])   # project folder (Check/dashboard -> project)
FX=U+'/Check/fixed/processed'; DP=U+'/Data/processed'
f=pd.read_csv(FX+'/final_buurt_risk_fixed.csv')
t=pd.read_csv(FX+'/transformer_projection_fixed.csv')
tb=pd.read_csv(FX+'/dashboard/buurt_trend.csv')
g=json.load(open(FX+'/dashboard/buurten_liander_fixed.geojson'))
LAT0=math.radians(52.2); K=5550.0; LON0=4.2; LAT1=53.6
def proj(c):
    a=np.asarray(c,float); return np.c_[(a[:,0]-LON0)*math.cos(LAT0)*K,(LAT1-a[:,1])*K]
def rdp(p,eps):
    if len(p)<4: return p
    keep=np.zeros(len(p),bool); keep[0]=keep[-1]=True
    st=[(0,len(p)-1)]
    while st:
        i,j=st.pop()
        if j<=i+1: continue
        a,b=p[i],p[j]; d=b-a; L=np.hypot(*d)
        seg=p[i+1:j]
        if L==0: dist=np.hypot(*(seg-a).T)
        else: dist=np.abs(d[0]*(seg[:,1]-a[1])-d[1]*(seg[:,0]-a[0]))/L
        k=int(np.argmax(dist))
        if dist[k]>eps: m=i+1+k; keep[m]=True; st+= [(i,m),(m,j)]
    return p[keep]
shapes={}; npts=0
for ft in g['features']:
    code=ft['properties']['buurtcode']; geom=ft['geometry']
    polys=geom['coordinates'] if geom['type']=='MultiPolygon' else [geom['coordinates']]
    out=[]
    for poly in polys:
        for ring in poly[:1] + [r for r in poly[1:] if len(r)>20]:
            q=np.round(rdp(proj(ring),1.6)).astype(int)
            q=q[np.r_[True,(np.diff(q,axis=0)!=0).any(1)]]
            if len(q)<3: continue
            enc=[int(q[0,0]),int(q[0,1])]+np.diff(q,axis=0).ravel().astype(int).tolist()
            out.append(enc); npts+=len(q)
    shapes[code]=out
print('points',npts)
PR={'1 Stressed now':1,'2 Crosses by 2027 (central)':2,'3 Crosses by 2027 (high)':3,'4 Crosses in stress test only':4,'5 No crossing expected':5,'No transformer data':6}
trend=tb.pivot_table(index='buurtcode',columns='year',values='total_kwh')
def r(x,n=3):
    return None if pd.isna(x) else round(float(x),n)
drv=sorted(set(pd.concat([f.driver_1,f.driver_2,f.driver_3]).dropna()))
gm=sorted(f.gemeentenaam.fillna('?').unique())
B=[]
for _,x in f.iterrows():
    tr=trend.loc[x.buurtcode] if x.buurtcode in trend.index else None
    B.append([x.buurtcode, x.buurtnaam if isinstance(x.buurtnaam,str) else x.buurtcode, gm.index(x.gemeentenaam if isinstance(x.gemeentenaam,str) else '?'),
      PR[x.priority], r(x.risk_prob), r(x.util_pred), r(x.util_weighted), int(x.n_transformers_proj) if pd.notna(x.n_transformers_proj) else 0,
      int(x.n_above_nbl_now) if pd.notna(x.n_above_nbl_now) else 0, int(x.n_new_2027_central or 0) if pd.notna(x.n_new_2027_central) else 0,
      int(x.n_new_2027_high) if pd.notna(x.n_new_2027_high) else 0, int(x.n_new_2027_solar) if pd.notna(x.n_new_2027_solar) else 0,
      r(x.min_headroom_pct,2), r(x.central_2027_pct,2), r(x.p10_2027_pct,2), r(x.p90_2027_pct,2),
      r(x.kwh_use2025,0), r(x.kwh_use2027_central,0), r(x.feed_share_pct,1), r(x.pv_growth_pct,1),
      int(bool(x.solar_sensitivity_flag)), int(bool(x.no_consumption_data)), int(x.low_reliability==True),
      [drv.index(d) if isinstance(d,str) else -1 for d in [x.driver_1,x.driver_2,x.driver_3]],
      x.risk_tier if isinstance(x.risk_tier,str) else None, r(x.connections,0),
      None if tr is None else [r(tr.get(y),0) for y in range(2019,2027)],
      r(x.central_2026_pct,2), r(x.feedin_kwh_2025,0), r(x.feedin_kwh_2026,0)])
T={}
for _,x in t.iterrows():
    T.setdefault(x.buurtcode,[]).append([x.transformer_id,int(x.peak_kw),int(x.capacity_kw),int(x.nbl_limit_kw),round(x.headroom_pct,1),str(x.peak_date)[:10],
      int(x.above_nbl_now),int(x.above_nbl_2027_central),int(x.above_nbl_2027_high),int(x.above_nbl_2027_solar)])
sc=pd.read_csv(FX+'/scenario_table.csv').to_dict('records')
cov=pd.read_csv(FX+'/dashboard/peak_coverage.csv')
cov=[[d[:10],r(q,2),int(n)] for d,q,n in zip(cov.date,cov.radiation_kwh_m2,cov.transformers_peaking)]
D=DP+'/dashboard'
ey=pd.read_csv(D+'/entsoe_yearly.csv'); ep=pd.read_csv(D+'/entsoe_profile_month_hour.csv'); cars=pd.read_csv(D+'/cars_fuel_national.csv')
su=pd.read_csv(DP+'/shap_importance_util.csv').head(12); ss=pd.read_csv(DP+'/shap_importance_stress.csv').head(12)
mb=pd.read_csv(DP+'/model_b_table.csv')
sc_pts=[[r(a,0),r(b,3),int(c)] for a,b,c in zip(mb.kwh_per_kw_capacity,mb.util_weighted,mb.stress_nbl) if pd.notna(a)]
hb=pd.cut(t.headroom_pct,[-1e9,0,1,5,10,1e9],labels=['now','0-1','1-5','5-10','>10'])
heads=[]
for lab in ['now','0-1','1-5','5-10','>10']:
    m=hb==lab; heads.append([lab,int(m.sum()),int((m&(t.above_nbl_now==0)&t.above_nbl_2027_central).sum()),int((m&(t.above_nbl_now==0)&t.above_nbl_2027_high).sum()),int((m&(t.above_nbl_now==0)&t.above_nbl_2027_solar).sum())])
out=dict(B=B,drv=drv,gm=gm,shapes=shapes,T=T,sc=sc,cov=cov,
 ey=ey[['year','energy_twh','peak_mw','complete_year','peak_time']].round(2).to_dict('records'),
 ep=ep[['month','hour','load_mw','solar_mw']].round(0).values.tolist(),
 cars=cars[['year','Totaal','Volledig elektrisch (FEV)','Plug-in hybirde (PHEV)']].values.tolist(),
 su=su.round(4).values.tolist(), ss=ss.round(4).values.tolist(), scat=sc_pts, heads=heads)
s=json.dumps(out,separators=(',',':'),allow_nan=False,default=str)
t=open(Path(__file__).with_name('template.html'),encoding='utf-8').read()
head='<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n<style>body{margin:0}</style>\n'
open(Path(__file__).with_name('grid_stress_dashboard.html'),'w',encoding='utf-8').write(head+t.replace('/*__DATA__*/',s.replace('</','<\\/'))+'\n</html>\n')
print('MB',len(s)/1e6)
