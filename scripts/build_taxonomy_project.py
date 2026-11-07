from __future__ import annotations
import json, sqlite3, re
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; OUT=ROOT/'outputs'; DB=DATA/'pp_pdf_batch02_final.sqlite'
FORM_MAP={
 'injection-molded test specimens':'molded_bulk','injection-molded test specimen':'molded_bulk',
 'injection-molded tensile/impact specimen':'molded_bulk','injection-molded tensile/flexural specimens':'molded_bulk',
 'compression-molded plate':'molded_bulk'}
ROLE_MAP={
 'matrix':'matrix','filler':'mineral_filler','mineral filler':'mineral_filler','inorganic filler':'mineral_filler',
 'modified mineral filler':'mineral_filler','conductive filler':'mineral_filler','nanofiller':'mineral_filler',
 'modified nano caco3':'mineral_filler','unmodified nano caco3':'mineral_filler',
 'reinforcement':'fiber_reinforcement','fiber filler':'fiber_reinforcement','natural fiber':'fiber_reinforcement','treated natural fiber':'fiber_reinforcement',
 'polymer blend component':'polymer_blend','blend component':'polymer_blend','elastomer toughener':'elastomer_toughening',
 'compatibilizer':'compatibilization','coupling agent':'compatibilization','flame retardant':'flame_retardant',
 'nucleating agent':'nucleation_crystallization','beta nucleating agent':'nucleation_crystallization',
 'antioxidant':'stabilizer_process_aid','primary antioxidant':'stabilizer_process_aid','secondary antioxidant':'stabilizer_process_aid',
 'lubricant':'stabilizer_process_aid','stabilizer':'stabilizer_process_aid','processing aid':'stabilizer_process_aid',
 'effect pigment':'pigment_colorant','modifier':'other_function'}

def form_class(x): return FORM_MAP.get(str(x or '').strip().lower(),'unknown_or_not_reported')
def form_from_record(row):
 sid=str(row.get('source_id') or ''); title=str(row.get('paper_title') or '').lower()
 # Source-level cross-checks identify product forms that the specimen_json field omitted.
 if sid=='SRC_P119' or ('silica' in title and 'film' in title) or ('硅' in title and '薄膜' in title): return 'film_sheet'
 if sid=='SRC_P169' or 'nonwoven' in title or '无纺布' in title: return 'nonwoven_fabric'
 return form_class(row.get('specimen_form'))
def matrix_class(raw):
 cs=json.loads(raw or '[]'); mats=[c for c in cs if str(c.get('role') or '').lower()=='matrix']
 names=' '.join(str(c.get('material_name') or '')+' '+str(c.get('grade') or '') for c in mats).lower()
 if not mats:return 'matrix_not_identified'
 if any(t in names for t in ['recycl','再生']):return 'recycled_pp'
 if any(t in names for t in ['copolymer','共聚','impact','block','嵌段','random']):return 'copolymer_or_modified_pp'
 return 'pp_homopolymer_or_unspecified'
def make_features(df):
 rows=[]; cols=set()
 for raw in df.components_json:
  d=defaultdict(float)
  for c in json.loads(raw or '[]'):
   if c.get('wt_pct') is None:continue
   wt=float(c['wt_pct']); fam=str(c.get('substance_family') or 'unknown')
   role=' '.join(str(c.get('role') or 'unknown').replace('_',' ').split()).lower()
   fn=ROLE_MAP.get(role,'other_function')
   d['family_wt__'+fam]+=wt; d['function_wt__'+fn]+=wt
  rows.append(d); cols.update(d)
 return pd.DataFrame(rows,index=df.index).reindex(columns=sorted(cols),fill_value=0.0)
def pipe(nums,cats,est):
 num=Pipeline([('imp',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())])
 cat=Pipeline([('imp',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),('ohe',OneHotEncoder(handle_unknown='ignore'))])
 return Pipeline([('pre',ColumnTransformer([('num',num,nums),('cat',cat,cats)])),('model',est)])
def score(df,X,nums,cats,est):
 y=df.strength_mpa.to_numpy(float); g=df.source_id.astype(str).to_numpy(); pred=np.full(len(df),np.nan); base=np.full(len(df),np.nan)
 for tr,te in LeaveOneGroupOut().split(X,y,g):
  m=pipe(nums,cats,est);m.fit(X.iloc[tr],y[tr]);pred[te]=m.predict(X.iloc[te]);base[te]=np.mean(y[tr])
 return {'n_rows':len(df),'n_papers':len(set(g)),'baseline_MAE':float(mean_absolute_error(y,base)),
  'baseline_RMSE':float(np.sqrt(mean_squared_error(y,base))),'baseline_R2':float(r2_score(y,base)),
  'model_MAE':float(mean_absolute_error(y,pred)),'model_RMSE':float(np.sqrt(mean_squared_error(y,pred))),
  'model_R2':float(r2_score(y,pred)),'folds_model_beats_baseline_MAE':int(sum(mean_absolute_error(y[t],pred[t])<mean_absolute_error(y[t],base[t]) for _,t in LeaveOneGroupOut().split(X,y,g))),
  'folds_total':len(set(g)), 'pred':pred,'base':base}

def main():
 OUT.mkdir(exist_ok=True); d=pd.read_csv(DATA/'model_input_features_v1.csv',encoding='utf-8-sig')
 d['specimen_form_class']=d.apply(form_from_record,axis=1); d['pp_matrix_class']=d.components_json.map(matrix_class)
 compfeat=make_features(d); d=pd.concat([d,compfeat],axis=1)
 fm=json.loads((DATA/'feature_manifest_v1.json').read_text(encoding='utf-8'))
 process_num=[c for c in fm['numeric_features'] if c in d.columns and not c.startswith(('component_pct__','role_pct__'))]
 nums=list(compfeat.columns)+process_num
 cats=[c for c in fm['categorical_features'] if c in d.columns]+['specimen_form_class','pp_matrix_class']
 for c in cats:d[c]=d[c].fillna('not_reported').astype(str)
 estimator_specs=[('Ridge_family',Ridge(alpha=10.0)),('RandomForest_family',RandomForestRegressor(n_estimators=500,min_samples_leaf=3,max_features=.8,random_state=42,n_jobs=-1)),('ExtraTrees_family',ExtraTreesRegressor(n_estimators=500,min_samples_leaf=3,max_features=.8,random_state=42,n_jobs=-1))]
 form_compatible=d[~d.specimen_form_class.isin(['film_sheet','nonwoven_fabric'])]
 scopes={
  'all_initial_candidates_before_source_form_review':d,
  'general_bulk_candidates_after_source_form_review':form_compatible,
  'form_unknown_or_not_reported':d[d.specimen_form_class=='unknown_or_not_reported'],
  'form_molded_bulk':d[d.specimen_form_class=='molded_bulk'],
  'unknown_form_homopolymer_or_unspecified_matrix':d[(d.specimen_form_class=='unknown_or_not_reported')&(d.pp_matrix_class=='pp_homopolymer_or_unspecified')],
 }
 results=[]; predictions=[]
 for scope,sub in scopes.items():
  if len(sub)<10 or sub.source_id.nunique()<4:continue
  X=sub[nums+cats].copy()
  for name,est in estimator_specs:
   z=score(sub,X,nums,cats,est); pred=z.pop('pred');base=z.pop('base');results.append({'scope':scope,'model':name,**z})
   if scope=='unknown_form_homopolymer_or_unspecified_matrix' and name=='RandomForest_family':
    for (_,row),p,b in zip(sub.iterrows(),pred,base): predictions.append({'measurement_id':row.measurement_id,'source_id':row.source_id,'paper_title':row.paper_title,'sample_label':row.sample_label,'actual_MPa':row.strength_mpa,'model_oof_MPa':p,'mean_oof_MPa':b})
 result_df=pd.DataFrame(results);result_df.to_csv(OUT/'taxonomy_model_metrics.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(predictions).to_csv(OUT/'taxonomy_oof_predictions.csv',index=False,encoding='utf-8-sig')
 chosen=next(r for r in results if r['scope']=='unknown_form_homopolymer_or_unspecified_matrix' and r['model']=='RandomForest_family')
 eligible=scopes['unknown_form_homopolymer_or_unspecified_matrix'].copy()
 X=eligible[nums+cats].copy();m=pipe(nums,cats,RandomForestRegressor(n_estimators=500,min_samples_leaf=3,max_features=.8,random_state=42,n_jobs=-1));m.fit(X,eligible.strength_mpa.to_numpy(float));dump(m,OUT/'experimental_taxonomy_rf.joblib')
 schema={'target':'tensile_strength_mpa','numeric_features':nums,'categorical_features':cats,'scope':{'specimen_form_class':'unknown_or_not_reported','pp_matrix_class':'pp_homopolymer_or_unspecified'},'n_rows':len(eligible),'n_papers':int(eligible.source_id.nunique()),'model':'RandomForestRegressor(n_estimators=500,min_samples_leaf=3,max_features=0.8,random_state=42)','validation':'Leave-one-paper-out','validation_metrics':chosen,'role_map':ROLE_MAP,'warning':'Exploratory grouped model; nearest literature records are supporting analogs, not single-paper attribution.'}
 (OUT/'taxonomy_model_manifest.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'taxonomy_model_metrics.json').write_text(json.dumps({'validation':'Leave-one-paper-out; paper source_id used only as group','results':results,'counts':{k:{'rows':len(v),'papers':int(v.source_id.nunique())} for k,v in scopes.items()},'warning':'Exploratory internal evaluation, not independent external validation.'},ensure_ascii=False,indent=2),encoding='utf-8')
 build_retrieval_data()
 print(json.dumps({'training':{'n_rows':len(eligible),'n_papers':eligible.source_id.nunique()},'results':results},ensure_ascii=False,indent=2))

def build_retrieval_data():
 conn=sqlite3.connect(DB);conn.row_factory=sqlite3.Row
 query='''SELECT m.id measurement_id,m.source_id,s.title paper_title,s.doi,s.locator source_locator,m.specimen_id,sp.details_json specimen_json,
 b.formulation_id,b.process_steps_json,f.source_sample,f.methods_json,m.property,m.value,m.unit,m.standard,m.conditions_json,m.n,m.sd,m.reviewed,m.evidence_locator
 FROM measurements m JOIN sources s ON s.id=m.source_id LEFT JOIN specimens sp ON sp.id=m.specimen_id LEFT JOIN batches b ON b.id=sp.batch_id LEFT JOIN formulations f ON f.id=b.formulation_id'''
 records=[dict(r) for r in conn.execute(query)]; by_form=defaultdict(list)
 for r in conn.execute('''SELECT c.formulation_id,c.material_id,c.role,c.raw_amount,c.raw_unit,c.basis,c.wt_pct,c.amount_status,c.evidence_locator component_locator,
 ma.substance_id,ma.name material_name,ma.grade,ma.manufacturer,su.name substance_name,su.family substance_family,su.formula
 FROM components c JOIN materials ma ON ma.id=c.material_id JOIN substances su ON su.id=ma.substance_id'''):
  by_form[r['formulation_id']].append(dict(r))
 aliases=defaultdict(list)
 for r in conn.execute('SELECT substance_id,alias FROM aliases'):aliases[r[0]].append(r[1])
 subs={r['id']:dict(r) for r in conn.execute('SELECT id,name,family,formula FROM substances')}
 for r in records:
  comps=by_form.get(r.get('formulation_id'),[]);r['components_json']=json.dumps(comps,ensure_ascii=False)
  vals=[]
  for c in comps:
   sid=c.get('substance_id');su=subs.get(sid,{})
   vals += [sid or '',su.get('name') or '',su.get('formula') or '',c.get('material_name') or '']+aliases.get(sid,[])
  r['components_search']=' | '.join(dict.fromkeys(v for v in vals if v))
 pd.DataFrame(records).to_csv(DATA/'literature_measurements.csv',index=False,encoding='utf-8-sig')
 catalog=[]
 for sid,su in subs.items():catalog.append({**su,'aliases':aliases.get(sid,[])})
 (DATA/'substance_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
 pd.DataFrame([{'property':p,'unit':u or '未报告','n':n} for p,u,n in conn.execute('SELECT property,unit,COUNT(*) FROM measurements GROUP BY property,unit ORDER BY COUNT(*) DESC')]).to_csv(DATA/'property_inventory.csv',index=False,encoding='utf-8-sig')
 conn.close()
 # Selected model cohort records enriched with direct source and page-level evidence locations.
 d=pd.read_csv(DATA/'model_input_features_v1.csv',encoding='utf-8-sig');d['specimen_form_class']=d.apply(form_from_record,axis=1);d['pp_matrix_class']=d.components_json.map(matrix_class)
 sel=d[(d.specimen_form_class=='unknown_or_not_reported')&(d.pp_matrix_class=='pp_homopolymer_or_unspecified')].copy()
 meas=pd.DataFrame(records)[['measurement_id','source_locator','evidence_locator','specimen_json','process_steps_json','conditions_json','components_json','paper_title','doi','source_id','property','value','unit','standard','n','sd','reviewed']]
 # Add original human/AI evidence fields from the candidate review spreadsheet.
 try:
  from openpyxl import load_workbook
  wb=load_workbook(DATA/'tensile_strength_candidate_review_codex_checked.xlsx',read_only=True,data_only=True);ws=wb['人工核对清单'];heads=[c.value for c in ws[1]];rv={}
  for vals in ws.iter_rows(min_row=2,values_only=True):
   z=dict(zip(heads,vals));mid=z.get('测量ID')
   if mid: rv[str(mid)]={k:z.get(k) for k in ['拉伸强度原文定位','配方原文定位','加工/制样定位','AI复核证据定位（PDF物理页优先）','数据库文献定位']}
  review=pd.DataFrame([{'measurement_id':mid,**data} for mid,data in rv.items()])
 except Exception:review=pd.DataFrame(columns=['measurement_id'])
 sel=sel.merge(meas.drop_duplicates('measurement_id'),on='measurement_id',how='left',suffixes=('','_db'))
 if len(review):sel=sel.merge(review,on='measurement_id',how='left')
 sel.to_csv(DATA/'model_reference_records.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':main()
