from __future__ import annotations
import argparse, hashlib, json, sqlite3
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / 'data' / 'pp_pdf_batch02_final.sqlite'
DEFAULT_REVIEW = ROOT / 'data' / 'tensile_strength_candidate_review_codex_checked.xlsx'
DEFAULT_OUT = ROOT / 'data'

# Initial modeling scope. These categories are kept in the database and exports;
# only the proposed general bulk-material cohort enters the first baseline run.
CABLE = {'SRC_P009','SRC_P048','SRC_P053','SRC_P055','SRC_P062','SRC_P077','SRC_P098','SRC_P127','SRC_P162','SRC_P166'}
SPECIAL = {
    'SRC_P035': ('drawn_fiber_product', '测试对象是拉伸后的聚丙烯纤维产品，不能与块状PP材料的强度直接混训。'),
    'SRC_P059': ('polypropylene_fiber_product', '测试对象是聚丙烯纤维产品，建议作为纤维性能单独建模。'),
    'SRC_P049': ('porous_membrane', '测试对象为多孔膜，试样形态及拉伸定义与块状材料不同。'),
    'SRC_P152': ('open_cell_grid_structure', '测试对象为开放网格结构，测得性能受结构几何影响，建议单独建模。'),
    'SRC_P063': ('recycled_feedstock', '记录对象是不同来源再生PP料流/原料，不是加入改性剂后的最终配方样品。'),
    'SRC_P071': ('conveyor_belt_cover', '输送带覆盖材料应用体系，含SBR/CaCO3等，建议先作为专用应用组。'),
    'SRC_P081': ('biofiller_rubber_PP_composite', '生物填料/橡胶复合应用体系，与普通PP共混/填充组分布不同，先单独评估。'),
}
CORE_NOTES = '初筛为块状PP改性材料候选；仍需结合配方、加工、试样和测试条件解释，属于探索性建模范围。'

# Explicit method details found during the PDF cross-check. They enrich only the
# exported modeling table; the SQLite source of truth is never changed.
SOURCE_METHODS = {
    'SRC_P049': {'standard':'GB/T 1040.3-2006','speed':50,'temperature':25,'replicates':5,'statistic':'每样品5次测量并取平均','evidence':'PDF p4'},
    'SRC_P053': {'standard':'ISO 527-3:2018','speed':50,'temperature_status':'room_temperature','replicates':5,'statistic':'每种材料5次重复','specimen_note':'厚1 mm；标距50 mm；初始夹距20 mm','evidence':'PDF p4'},
    'SRC_P055': {'standard':'ISO 527-3:2018','speed':50,'replicates':5,'statistic':'每个样品测试5次并取平均','specimen_note':'75×4×2 mm哑铃样条','evidence':'PDF p4'},
    'SRC_P056': {'standard':'ASTM D638','replicates':5,'statistic':'每个配方5个重复；拉伸速度未报告','evidence':'PDF p3'},
    'SRC_P060': {'standard':'GB/T 1040.1-2006','speed':50,'evidence':'PDF p4'},
    'SRC_P063': {'standard':'GB/T 1040-2006','speed':50,'evidence':'PDF p2'},
    'SRC_P066': {'standard':'GB/T 1040-2006','speed':50,'replicates':5,'statistic':'5个试样取平均','evidence':'PDF p4'},
    'SRC_P071': {'standard':'GB/T 1040.1-2018','speed':50,'temperature_status':'room_temperature','specimen_note':'原文记载样品尺寸200×100×10 mm；尺寸含义需确认','evidence':'PDF p2'},
    'SRC_P078': {'standard':'GB/T 1040.2-2022','replicates':3,'statistic':'每组3个试样取平均；拉伸速度未报告','evidence':'PDF p3'},
    'SRC_P093': {'standard':'GB/T 1040-2006','speed':50,'temperature':23,'specimen_note':'拉伸样条75×4×1 mm','evidence':'PDF p2'},
    'SRC_P100': {'standard':'GB/T 1040-1992','speed':500,'replicates':5,'statistic':'5个样品取平均；恒温恒湿但原文未给具体温湿度','specimen_note':'125×6.5×3 mm哑铃样条','evidence':'PDF p2'},
    'SRC_P152': {'standard':'ASTM D638','speed':5,'temperature_status':'room_temperature','temperature':23,'relative_humidity':50,'replicates_lower_bound':3,'specimen_note':'Type I；标距50 mm；厚3.2 mm','evidence':'PDF p6，拉伸测试方法'},
    'SRC_P169': {'standard':'SR EN ISO 527-4:2006','speed':5,'specimen_note':'夹距150 mm','evidence':'PDF p3'},
    'SRC_P176': {'standard':'GB/T 1040.1—2018 / ISO 527-1:2012','speed':20,'specimen_note':'拉伸试样150×10×4 mm','evidence':'PDF p3 §1.4'},
}


def stable_feature_id(prefix: str, ident: str) -> str:
    return prefix + hashlib.sha1(ident.encode('utf-8')).hexdigest()[:10]


def safe_json(value, fallback):
    try:
        return json.loads(value) if value else fallback
    except Exception:
        return fallback


def source_scope(source_id: str):
    if source_id in CABLE:
        return 'cable_insulation', 0, '电缆绝缘材料应用组，暂从通用块状材料基线中分开；保留用于后续专用模型。'
    if source_id in SPECIAL:
        cohort, reason = SPECIAL[source_id]
        return cohort, 0, reason
    return 'general_bulk_PP_candidate', 1, CORE_NOTES


def read_review(path: Path):
    wb = load_workbook(path, data_only=True, read_only=True)
    if '人工核对清单' not in wb.sheetnames:
        raise ValueError('复核表中没有“人工核对清单”工作表。')
    ws = wb['人工核对清单']
    headers = [c.value for c in ws[1]]
    data = {}
    for vals in ws.iter_rows(min_row=2, values_only=True):
        row = dict(zip(headers, vals))
        mid = row.get('测量ID')
        if mid:
            data[str(mid)] = row
    return data


def get_measurements(db: Path, selected_ids: list[str]):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    marks = ','.join('?' for _ in selected_ids)
    sql = f'''SELECT m.id AS measurement_id, m.source_id, m.property, m.value, m.unit,
                     m.standard, m.conditions_json, m.n, m.sd, m.reviewed,
                     s.title AS paper_title, s.doi, s.locator AS source_locator,
                     sp.id AS specimen_id, sp.details_json AS specimen_json,
                     b.id AS batch_id, b.locator AS batch_locator,
                     b.process_steps_json, f.id AS formulation_id,
                     f.source_sample, f.methods_json, f.composition_complete, f.notes AS formulation_notes
              FROM measurements m
              JOIN sources s ON s.id=m.source_id
              JOIN specimens sp ON sp.id=m.specimen_id
              JOIN batches b ON b.id=sp.batch_id
              JOIN formulations f ON f.id=b.formulation_id
              WHERE m.id IN ({marks})'''
    records = [dict(r) for r in conn.execute(sql, selected_ids)]
    if len(records) != len(selected_ids):
        found = {r['measurement_id'] for r in records}
        missing = [x for x in selected_ids if x not in found]
        raise ValueError(f'数据库中找不到核对表记录：{missing[:8]}')
    formulation_ids = sorted({r['formulation_id'] for r in records})
    comp_marks = ','.join('?' for _ in formulation_ids)
    comps = defaultdict(list)
    sqlc = f'''SELECT c.formulation_id,c.material_id,c.role,c.raw_amount,c.raw_unit,c.basis,
                      c.wt_pct,c.conversion_note,c.amount_status,c.evidence_locator,c.extraction_method,
                      ma.substance_id,ma.name AS material_name,ma.grade,ma.manufacturer,
                      su.name AS substance_name,su.family AS substance_family,su.formula
               FROM components c
               JOIN materials ma ON ma.id=c.material_id
               JOIN substances su ON su.id=ma.substance_id
               WHERE c.formulation_id IN ({comp_marks})'''
    for row in conn.execute(sqlc, formulation_ids):
        comps[row['formulation_id']].append(dict(row))
    conn.close()
    return records, comps


def normalize_conditions(standard, conditions_json):
    c = safe_json(conditions_json, {})
    std = standard or c.get('standard') or c.get('tensile_standard') or c.get('test_standard') or c.get('test_method')
    speed = None
    for key in ['crosshead_speed_mm_min','tensile_crosshead_speed_mm_min','tensile_speed_mm_min','test_speed_mm_min','tensile_rate_mm_min']:
        val = c.get(key)
        if isinstance(val, (int,float)) and not isinstance(val, bool):
            speed = float(val); break
    raw_temp = None
    for key in ['test_temperature_C','tensile_temperature','temperature_C','test_temperature','temperature']:
        if c.get(key) is not None:
            raw_temp=c.get(key); break
    temp_num = None
    temp_status = 'not_reported'
    if isinstance(raw_temp,(int,float)) and not isinstance(raw_temp,bool):
        temp_num=float(raw_temp); temp_status='numeric_reported'
    elif isinstance(raw_temp,str):
        t=raw_temp.strip().lower()
        if 'room' in t or '室温' in t:
            temp_status='room_temperature'
        elif t and t not in {'not reported','未报告','n/a','unknown'}:
            temp_status='reported_text'
    rh = c.get('relative_humidity_percent')
    if rh is None:
        rh = c.get('conditioning_relative_humidity_percent')
    rh = float(rh) if isinstance(rh,(int,float)) and not isinstance(rh,bool) else None
    test_state = c.get('state') or c.get('test_state') or c.get('aging_state') or 'not_reported'
    return c, std, speed, temp_num, temp_status, rh, str(test_state)


def process_features(process_json, methods_json):
    p=safe_json(process_json,[])
    meth=safe_json(methods_json,[])
    if not isinstance(p,list): p=[p]
    if not isinstance(meth,list): meth=[meth]
    raw=' '.join(str(x) for x in [p,meth]).lower()
    routes=[]
    for term,tag in [('injection','injection_molding'),('注塑','injection_molding'),('twin_screw','extrusion'),('extrusion','extrusion'),('挤出','extrusion'),('compression','compression_molding'),('热压','compression_molding'),('压片','compression_molding'),('melt_blend','melt_blending'),('melt blending','melt_blending'),('熔融共混','melt_blending')]:
        if term in raw and tag not in routes: routes.append(tag)
    if not routes: routes=['not_reported_or_unclassified']
    temps=[]; speeds=[]; durations_min=[]
    def visit(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                lk=k.lower()
                if ('temperature' in lk or 'temp' in lk) and ('c' in lk or lk in {'temperature','temperature_c'}):
                    vals=v if isinstance(v,list) else [v]
                    for z in vals:
                        if isinstance(z,(int,float)) and not isinstance(z,bool): temps.append(float(z))
                if ('screw' in lk or 'rpm' in lk) and isinstance(v,(int,float)) and not isinstance(v,bool): speeds.append(float(v))
                if lk in {'time_h','duration_h'} and isinstance(v,(int,float)) and not isinstance(v,bool): durations_min.append(float(v)*60)
                if lk in {'time_min','duration_min'} and isinstance(v,(int,float)) and not isinstance(v,bool): durations_min.append(float(v))
                if lk in {'time_s','duration_s','holding_time_s'} and isinstance(v,(int,float)) and not isinstance(v,bool): durations_min.append(float(v)/60)
                visit(v)
        elif isinstance(obj,list):
            for v in obj: visit(v)
    visit(p)
    return '+'.join(routes), p, meth, (min(temps) if temps else None), (max(temps) if temps else None), (max(speeds) if speeds else None), (max(durations_min) if durations_min else None)


def main():
    ap=argparse.ArgumentParser(description='从正式SQLite和拉伸候选复核表生成v1训练候选数据。只读数据库。')
    ap.add_argument('--db',type=Path,default=DEFAULT_DB)
    ap.add_argument('--review',type=Path,default=DEFAULT_REVIEW)
    ap.add_argument('--outdir',type=Path,default=DEFAULT_OUT)
    args=ap.parse_args()
    for p in [args.db,args.review]:
        if not p.exists(): raise FileNotFoundError(f'找不到文件：{p}')
    args.outdir.mkdir(parents=True,exist_ok=True)
    review=read_review(args.review)
    ids=list(review)
    records, comp_map=get_measurements(args.db,ids)
    records_by_id={r['measurement_id']:r for r in records}
    # preserve the order of the review list for easy row-by-row traceability
    records=[records_by_id[mid] for mid in ids]
    rows=[]; long_rows=[]; feature_map={}; source_scope_info={}
    for r in records:
        rev=review[r['measurement_id']]
        cond,std,speed,temp,temp_status,rh,test_state=normalize_conditions(r['standard'],r['conditions_json'])
        method_override=SOURCE_METHODS.get(r['source_id'], {})
        # Fill only absent/unspecified fields from source-level method sections.
        # Existing row-level evidence is preserved when it already contains a value.
        if method_override.get('standard') and (std in (None, '', 'not_reported') or 'mm/min' in str(std).lower()):
            std=method_override['standard']
        if speed is None and method_override.get('speed') is not None:
            speed=float(method_override['speed'])
        if temp is None and method_override.get('temperature') is not None:
            temp=float(method_override['temperature'])
        if temp_status=='not_reported' and method_override.get('temperature_status'):
            temp_status=method_override['temperature_status']
        if rh is None and method_override.get('relative_humidity') is not None:
            rh=float(method_override['relative_humidity'])
        route,process,methods,tmin,tmax,screw,duration=process_features(r['process_steps_json'],r['methods_json'])
        source_id=r['source_id']; cohort,include,reason=source_scope(source_id)
        source_scope_info[source_id]=(cohort,include,reason,r['paper_title'])
        comps=comp_map.get(r['formulation_id'],[])
        comp_json=[]; sum_wt=0.0; role_totals=defaultdict(float); subst_totals=defaultdict(float)
        for c in comps:
            item={k:c.get(k) for k in ['material_id','substance_id','substance_name','substance_family','formula','material_name','grade','manufacturer','role','raw_amount','raw_unit','basis','wt_pct','amount_status','evidence_locator']}
            comp_json.append(item)
            wt=c.get('wt_pct')
            if wt is not None:
                wt=float(wt); sum_wt+=wt; role_totals[str(c.get('role') or 'unknown')]+=wt
                sid=str(c.get('substance_id') or 'SUB_UNKNOWN')
                subst_totals[sid]+=wt
                col=stable_feature_id('component_pct__',sid)
                feature_map[col]={'substance_id':sid,'substance_name':c.get('substance_name'),'family':c.get('substance_family'),'formula':c.get('formula')}
            long_rows.append({'measurement_id':r['measurement_id'],'source_id':source_id,'formulation_id':r['formulation_id'],**item})
        specimen=safe_json(r['specimen_json'],{})
        for sid,wt in subst_totals.items():
            pass
        row={
            'measurement_id':r['measurement_id'],'source_id':source_id,'paper_title':r['paper_title'],'doi':r['doi'],
            'sample_label':r['source_sample'],'specimen_id':r['specimen_id'],'formulation_id':r['formulation_id'],
            'strength_mpa':float(r['value']) if r['value'] is not None else None,'unit':r['unit'],
            'result_status':r['property'],'test_standard':std or 'not_reported','test_speed_mm_min':speed,
            'test_temperature_C':temp,'test_temperature_status':temp_status,'relative_humidity_percent':rh,
            'test_state':test_state,'processing_route':route,'process_temperature_min_C':tmin,'process_temperature_max_C':tmax,
            'process_screw_speed_rpm':screw,'process_duration_min_max':duration,
            'source_method_replicates':method_override.get('replicates'),
            'source_method_replicates_lower_bound':method_override.get('replicates_lower_bound'),
            'source_method_statistic':method_override.get('statistic',''),
            'source_method_specimen_note':method_override.get('specimen_note',''),
            'condition_override_applied':bool(method_override),
            'condition_override_evidence':method_override.get('evidence',''),
            'application_domain':cond.get('application_domain','not_reported'),
            'specimen_form':specimen.get('specimen_form','not_reported'),'specimen_description':json.dumps(specimen,ensure_ascii=False),
            'process_steps_json':json.dumps(process,ensure_ascii=False),'formulation_methods_json':json.dumps(methods,ensure_ascii=False),
            'test_conditions_json':json.dumps(cond,ensure_ascii=False),'components_json':json.dumps(comp_json,ensure_ascii=False),
            'composition_wt_pct_sum':sum_wt,'composition_complete':int(r['composition_complete']),
            'n':r['n'],'sd_mpa':r['sd'],'database_reviewed':r['reviewed'],
            'ai_crosscheck_status':rev.get('AI复核结论','未提供AI复核列'),'review_evidence':rev.get('AI复核证据定位（PDF物理页优先）',rev.get('拉伸强度原文定位','')),
            'proposed_cohort':cohort,'include_in_general_model':include,'cohort_reason':reason,
            'quality_flag':rev.get('质量标记'),'human_review_decision':rev.get('人工核对结论（填写）'),
        }
        for col in feature_map: row[col]=0.0
        for sid,wt in subst_totals.items(): row[stable_feature_id('component_pct__',sid)]=wt
        for role,wt in role_totals.items(): row[stable_feature_id('role_pct__',role)]=wt
        rows.append(row)
    # Make all dynamic columns present in all rows, including newly encountered role columns.
    component_cols=sorted(feature_map)
    role_cols=sorted({k for row in rows for k in row if k.startswith('role_pct__')})
    for row in rows:
        for col in component_cols+role_cols: row.setdefault(col,0.0)
    all_df=pd.DataFrame(rows)
    core_df=all_df[all_df['include_in_general_model']==1].copy()
    all_df.to_csv(args.outdir/'tensile_candidates_all_v1.csv',index=False,encoding='utf-8-sig')
    core_df.to_csv(args.outdir/'model_input_features_v1.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(long_rows).to_csv(args.outdir/'tensile_components_long_v1.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'feature_column':k,**v} for k,v in sorted(feature_map.items())]).to_csv(args.outdir/'component_feature_map_v1.csv',index=False,encoding='utf-8-sig')
    scope=[]
    for sid,(cohort,include,reason,title) in sorted(source_scope_info.items()):
        scope.append({'source_id':sid,'paper_title':title,'candidate_rows':int((all_df.source_id==sid).sum()),'proposed_cohort':cohort,'include_in_general_model':include,'reason':reason,'decision_status':'AI初筛建议；可在此表讨论，未修改正式数据库'})
    pd.DataFrame(scope).to_csv(args.outdir/'model_scope_summary_v1.csv',index=False,encoding='utf-8-sig')
    # Build a feature manifest for reproducibility. source_id is grouping only and is not a predictor.
    numeric=[c for c in component_cols+role_cols if c in core_df.columns]
    numeric += ['test_speed_mm_min','test_temperature_C','relative_humidity_percent','process_temperature_min_C','process_temperature_max_C','process_screw_speed_rpm','process_duration_min_max']
    categorical=['test_standard','test_temperature_status','test_state','processing_route','specimen_form']
    manifest={'target':'strength_mpa','group_only_not_model_feature':'source_id','numeric_features':numeric,'categorical_features':categorical,'component_feature_map':[{'feature_column':k,**feature_map[k]} for k in component_cols],'scope':'include_in_general_model == 1; initial bulk PP modification cohort; see model_scope_summary_v1.csv','notes':['source_id is used only for grouped validation, never as a model feature.','Component amounts are aggregated by canonical substance_id; grade-specific information remains in components_json for traceability.','Missing test/process values remain missing and are imputed within each training fold.','The proposed cohort is exploratory and should be reviewed before production use.']}
    (args.outdir/'feature_manifest_v1.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    domain_counts=Counter(all_df['proposed_cohort'])
    speed_missing=int(all_df['test_speed_mm_min'].isna().sum())
    standard_missing=int((all_df['test_standard']=='not_reported').sum())
    n_missing=int(all_df['n'].isna().sum())
    formula_sum_errors=int((all_df['composition_wt_pct_sum'].sub(100).abs()>1e-5).sum())
    summary={'source_review_rows':len(all_df),'source_count':int(all_df.source_id.nunique()),'general_bulk_candidate_rows':len(core_df),'general_bulk_candidate_sources':int(core_df.source_id.nunique()),'proposed_cohort_counts':dict(domain_counts),'unique_substances_in_candidate_formulations':len(feature_map),'composition_sum_errors':formula_sum_errors,'test_speed_missing_rows':speed_missing,'standard_missing_rows':standard_missing,'n_missing_rows':n_missing,'all_database_review_flags_unchanged':bool((all_df.database_reviewed==0).all()),'training_ready':False,'note':'This step exports and scopes the candidate data. Review the cohort and data-quality report before interpreting a model as predictive.'}
    (args.outdir/'data_quality_summary_v1.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
