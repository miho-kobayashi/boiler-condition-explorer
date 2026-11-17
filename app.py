# -*- coding: utf-8 -*-
"""Boiler Condition Explorer: engineering observation aid, not fault diagnosis."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Boiler Condition Explorer', page_icon='🏭', layout='wide')
BASE = Path(__file__).resolve().parent
CASES = {
    'A': ('返送空気室・粒子返送系', '左右返送空気の配分変化と層差圧の関係', '2022-03-30 08:50:00', ['FT_8306A','FT_8306B','ZCLCCY','YCLCCY','SXLTCYZ','SXLTCYY']),
    'B': ('燃焼炉・誘引通風系', '炉圧多点の同時変化と送排気条件', '2022-03-30 06:05:00', ['PT_8313A','PT_8313B','PT_8313C','PT_8313D','PT_8313E','PT_8313F','FT_8301','FT_8302','YFJ3_AI']),
    'C': ('燃焼炉・空気供給系', '排ガスO₂と炉温・送風の整合性', '2022-03-29 14:05:00', ['AIR_8301A','AIR_8301B','TE_8313B','FT_8301','FT_8302','ZZQBCHLL']),
}
LABELS = {
    'FT_8306A':'返送空気・左', 'FT_8306B':'返送空気・右',
    'ZCLCCY':'層差圧・左','YCLCCY':'層差圧・右',
    'SXLTCYZ':'炉床差圧・左','SXLTCYY':'炉床差圧・右',
    'FT_8301':'一次空気','FT_8302':'二次空気','YFJ3_AI':'誘引通風機電流',
    'AIR_8301A':'排ガスO₂・左','AIR_8301B':'排ガスO₂・右',
    'TE_8313B':'炉上部温度','ZZQBCHLL':'主蒸気流量'
}
PRESS = [f'PT_8313{x}' for x in 'ABCDEF']

st.markdown('''<style>
.block-container{padding-top:1.25rem;padding-bottom:2rem}
.tagline{color:#64748b;font-size:.91rem}
.stMetric {border:1px solid rgba(128,128,128,.24);border-radius:9px;padding:12px}
</style>''',unsafe_allow_html=True)


def read_csv(file_bytes:bytes)->pd.DataFrame:
    last_error=None
    for enc in ('utf-8-sig','cp932','utf-16'):
        try:
            df=pd.read_csv(BytesIO(file_bytes),encoding=enc,low_memory=False)
            break
        except (UnicodeError, pd.errors.ParserError) as ex:
            last_error=ex
    else:
        raise ValueError(f'CSVを読み取れません: {last_error}')
    df.columns=[str(x).strip().lstrip('\ufeff') for x in df.columns]
    tc=next((x for x in df.columns if x.casefold() in {'date','time','datetime','timestamp','日時','日付','date_time'}),None)
    if tc is None:
        raise ValueError('日時列(date / time / datetime / timestamp / 日時)がありません。')
    df[tc]=pd.to_datetime(df[tc],errors='coerce')
    df=df.dropna(subset=[tc]).sort_values(tc).drop_duplicates(subset=[tc],keep='last').set_index(tc)
    rename={}
    for c in df.columns:
        code=c.split('.')[0].rstrip('#')
        if code not in rename.values():rename[c]=code
    df=df.rename(columns=rename)
    for c in df.columns:df[c]=pd.to_numeric(df[c],errors='coerce')
    if len(df)<4:raise ValueError('有効な時系列が4点未満です。')
    return df


def mean_safe(frame, tag):
    return float(frame[tag].mean()) if tag in frame and frame[tag].notna().any() else np.nan

def two_windows(df, anchor, minutes=10):
    # changes across non-overlapping intervals: [t-10,t) vs [t,t+10)
    pre=df.loc[(df.index>=anchor-pd.Timedelta(minutes=minutes))&(df.index<anchor)]
    post=df.loc[(df.index>=anchor)&(df.index<anchor+pd.Timedelta(minutes=minutes))]
    return pre,post

def contrast(df, anchor):
    a,b=two_windows(df,anchor)
    if len(a)<2 or len(b)<2:return {}
    r={}
    for x in set(sum([v[3] for v in CASES.values()],[])+PRESS):
        if x in df:r[x]=mean_safe(b,x)-mean_safe(a,x)
    def derived(frame):
        v={}
        if {'FT_8306A','FT_8306B'}.issubset(frame.columns):
            l=mean_safe(frame,'FT_8306A'); rr=mean_safe(frame,'FT_8306B')
            v['return_share_L']=l/(l+rr) if abs(l+rr)>1e-9 else np.nan
            v['return_total']=l+rr
        for nm,l,rr in [('layer_LR','ZCLCCY','YCLCCY'),('bed_LR','SXLTCYZ','SXLTCYY'),('oxygen_mean','AIR_8301A','AIR_8301B')]:
            if l in frame and rr in frame:
                v[nm]=(mean_safe(frame,l)+mean_safe(frame,rr))/2 if nm=='oxygen_mean' else mean_safe(frame,l)-mean_safe(frame,rr)
        pp=[p for p in PRESS if p in frame]
        if len(pp)>=2:
            v['pressure_median']=frame[pp].median(axis=1).mean()
        return v
    va=derived(a);vb=derived(b)
    for key in va.keys() & vb.keys():r[key]=vb[key]-va[key]
    pp=[p for p in PRESS if p in r and np.isfinite(r[p])]
    if len(pp)>=2:
        dr=np.array([r[p] for p in pp]);r['pressure_same_sign_fraction']=float(max((dr>0).mean(),(dr<0).mean()))
        r['pressure_channels']=len(pp)
    r['_pre_n']=len(a);r['_post_n']=len(b)
    return r


def history_benchmarks(df):
    # 10-min before/after changes, sampled every 10 minutes, exclude last 20 min.
    anchors=pd.date_range(df.index.min()+pd.Timedelta('12min'),df.index.max()-pd.Timedelta('12min'),freq='10min')
    if len(anchors)>3000:anchors=anchors[::int(np.ceil(len(anchors)/3000))]
    rows=[]
    for anchor in anchors:
        r=contrast(df,anchor)
        if r:
            rows.append({'timestamp':anchor,**r})
    return pd.DataFrame(rows).set_index('timestamp') if rows else pd.DataFrame()


def baseline_assessment(r, hist, key):
    if key not in r or not np.isfinite(r[key]):return '計測値不足',np.nan
    if hist.empty or key not in hist:return '比較データ不足',np.nan
    s=hist[key].dropna().abs()
    if len(s)<20:return '比較データ不足',np.nan
    q=float(s.quantile(.95));return ('履歴分布の上側5%相当' if abs(r[key])>q else '履歴比較範囲内'),q


def measured_frame(r, keys, hist):
    out=[]
    for name,key in keys:
        cls,thr=baseline_assessment(r,hist,key)
        out.append({'指標':name,'前後差':round(r[key],4) if key in r and np.isfinite(r[key]) else np.nan,
                    '過去の絶対変化95%点':round(thr,4) if np.isfinite(thr) else np.nan,'探索的な位置づけ':cls})
    return pd.DataFrame(out)


def case_messages(case,r):
    if case=='A':
        return ('返送空気室・流量配分系','返送空気の左右配分の変化が層差圧・炉床差圧と時間的に対応するかを検討。',
                '左右返送空気の設定・操作履歴、流量計の状態、関連する層差圧・炉床差圧、返送設備の状態を確認。',
                '固体の循環流量や粒子凝集は未測定。流量配分の変化だけで返送不良とは判定できない。')
    if case=='B':
        return ('炉圧計測・送排気／誘引通風系','炉圧の複数点で共通の変化か、特定点だけの変化かを確認。送風量・ファン電流との時差を観察。',
                '炉圧設定値、誘引通風機の運転記録・回転数・ダンパ開度、一次・二次空気量、圧力計の状態を確認。',
                '各点の絶対圧力が等しい必要はない。電流値から回転数・排気量を特定できず、ファン操作を自動推奨しない。')
    return ('燃焼・空気供給／排ガス計測系','排ガスO₂の変化を炉温・送風量・蒸気負荷と同時に確認。通常の反応・熱収支との整合性を探索。',
            '一次・二次空気量と制御指令、左右O₂計、炉温・蒸気負荷、可能なら燃料供給量・CO値を確認。',
            '排ガスO₂の低下だけで酸素供給不足や不完全燃焼とは断定できない。炉温が一定である必要もない。')


def line_chart(df, tags, t1, t2, title):
    show=df.loc[(df.index>=t1)&(df.index<=t2)]
    fig=go.Figure()
    for tag in tags:
        if tag in show and show[tag].notna().sum()>1:
            fig.add_trace(go.Scatter(x=show.index,y=show[tag],name=f'{tag}｜{LABELS.get(tag,tag)}',mode='lines'))
    fig.update_layout(title=title,height=290,margin=dict(l=15,r=15,b=25,t=45),legend=dict(orientation='h',y=-.3),hovermode='x unified')
    return fig

st.title('🏭 Boiler Condition Explorer')
st.markdown('<p class="tagline">運転波形 → 観測上の変化 → 設備部位と確認事項｜公開データに基づく探索・点検支援ツール</p>',unsafe_allow_html=True)
st.warning('**故障診断・安全制御ではありません。** 異常ラベル・運転仕様がないため、故障の有無を判定しません。設備操作（ファン回転数・送風量変更等）を指示しません。')
with st.sidebar:
    st.header('データの読み込み')
    uploaded=st.file_uploader('CSVをドラッグ＆ドロップ',type=['csv'],help='Kaggleのdata.csv / 列名にタグを含むCSVに対応')
    use_demo=st.checkbox('収録サンプル（ケースA・B・Cの1分集約抜粋）で試す',value=False)
    st.caption('CSV内の時刻で評価します。システムの現在時刻とは異なります。')
    st.divider()
    st.markdown('**表示の意味**')
    st.caption('「履歴分布の上側5%相当」は統計的な探索指標であり、故障確率・統計的有意性を示しません。')

if uploaded is None and not use_demo:
    st.info('左側のCSVアップロードから始めてください。サンプルを試す場合はチェックを入れてください。')
    st.markdown('**対応する3つの観測テーマ**：A 返送空気の左右配分と層差圧、B 炉圧多点の同時変化、C O₂と炉温・送風の整合性。')
    st.stop()
try:
    if uploaded is not None:
        df=read_csv(uploaded.getvalue());source=uploaded.name
    else:
        df=read_csv((BASE/'sample_cases_1min.csv').read_bytes());source='ケースA・B・Cの公開データ由来1分集約サンプル'
except Exception as e:
    st.error(f'読み込みエラー：{e}');st.stop()

recent=df.index.max();first=df.index.min()
dur=(recent-first).total_seconds()/60
c1,c2,c3,c4=st.columns(4)
c1.metric('ファイル内の最新時刻',recent.strftime('%Y-%m-%d %H:%M:%S'))
c2.metric('取得行数',f'{len(df):,}')
c3.metric('タグ数',str(len(df.columns)))
c4.metric('期間（分）',f'{dur:,.0f}')
st.caption(f'入力：{source}｜データ内の最終時刻が評価対象。5秒／1分間隔などの集計値に対応。計測単位はCSVの仕様に従います。')

with st.spinner('時間窓ごとの変化量を集計中…'):
    # resample to minute to avoid costly tick-by-tick historical contrasts.
    df1=df.resample('1min').mean(numeric_only=True).dropna(how='all')
    hist=history_benchmarks(df1)

st.subheader('最新時刻周辺の設備状態（探索的な監視）')
st.caption('最新10分平均とその直前10分平均を比較します。データ末尾の10分を後窓として、10分前を前窓に設定。')
latest_anchor=recent-pd.Timedelta(minutes=10)
latest=contrast(df1,latest_anchor)
if not latest:
    st.info('最新時刻付近に十分なデータがありません。少なくとも連続20分程度のデータが必要です。')
else:
    keys={
        'A':[('左返送空気流量','FT_8306A'),('右返送空気流量','FT_8306B'),('返送空気の左側配分比','return_share_L'),('層差圧の左右差','layer_LR')],
        'B':[('炉圧の多点中央値','pressure_median'),('一次空気流量','FT_8301'),('誘引通風機電流','YFJ3_AI')],
        'C':[('排ガスO₂左右平均','oxygen_mean'),('炉上部温度','TE_8313B'),('二次空気流量','FT_8302')]
    }
    columns=st.columns(3)
    for col,case in zip(columns,'ABC'):
        observed=[label for label,key in keys[case] if key in latest]
        classes=[baseline_assessment(latest,hist,key)[0] for _,key in keys[case] if key in latest]
        with col:
            st.markdown(f'**ケース{case}｜{CASES[case][0]}**')
            st.write(' / '.join(observed) if observed else '必要なタグが不足')
            if '履歴分布の上側5%相当' in classes:st.warning('大きな変化の候補（故障の確定ではない）')
            elif classes and all(x=='履歴比較範囲内' for x in classes):st.success('観測した変化量は履歴の比較範囲内')
            else:st.info('比較条件・測定点が不足')

st.divider()
st.subheader('選択したケースの根拠を確認')
case=st.radio('表示するテーマ',options=list(CASES.keys()),format_func=lambda k:f'{k}｜{CASES[k][0]}',horizontal=True)
case_name,case_description,anchor_literal,tags=CASES[case]
anchor_time=pd.Timestamp(anchor_literal)
opts=['ファイル内の最新時刻']
if first<=anchor_time<=recent:opts.append(f'第2章で検出したケース{case}（{anchor_literal}）')
mode=st.selectbox('観測する時間帯',opts)
anchor=latest_anchor if mode==opts[0] else anchor_time
start=anchor-pd.Timedelta(minutes=35);end=anchor+pd.Timedelta(minutes=35)
r=contrast(df1,anchor)
st.markdown(f'**問い：{case_description}**')
left,right=st.columns([3,2])
with left:
    if case=='A':
        st.plotly_chart(line_chart(df1,['FT_8306A','FT_8306B'],start,end,'左右返送空気流量'),use_container_width=True)
        st.plotly_chart(line_chart(df1,['ZCLCCY','YCLCCY'],start,end,'左右層差圧'),use_container_width=True)
    elif case=='B':
        st.plotly_chart(line_chart(df1,[t for t in PRESS if t in df1],start,end,'炉内圧力（測定点別）'),use_container_width=True)
        st.plotly_chart(line_chart(df1,['YFJ3_AI','FT_8301'],start,end,'送風・誘引通風機の観測量（単位差に注意）'),use_container_width=True)
    else:
        st.plotly_chart(line_chart(df1,['AIR_8301A','AIR_8301B'],start,end,'排ガスO₂（左右）'),use_container_width=True)
        st.plotly_chart(line_chart(df1,['TE_8313B','FT_8302'],start,end,'炉温・二次空気（単位差に注意）'),use_container_width=True)
with right:
    st.markdown('**前後10分の変化量と探索的な比較**')
    show={
        'A':[('返送空気左','FT_8306A'),('返送空気右','FT_8306B'),('左側配分比','return_share_L'),('左右層差圧差','layer_LR'),('左右炉床差圧差','bed_LR')],
        'B':[('炉圧中央値','pressure_median'),('一次空気','FT_8301'),('二次空気','FT_8302'),('ファン電流','YFJ3_AI')],
        'C':[('排ガスO₂平均','oxygen_mean'),('炉温','TE_8313B'),('一次空気','FT_8301'),('二次空気','FT_8302'),('主蒸気流量','ZZQBCHLL')]
    }
    table=measured_frame(r,show[case],hist)
    st.dataframe(table,hide_index=True,use_container_width=True)
    if case=='B' and 'pressure_same_sign_fraction' in r:
        st.caption(f'炉圧{int(r["pressure_channels"])}点の前後変化のうち、同方向の割合：{r["pressure_same_sign_fraction"]:.0%}。各計器の絶対値一致は要求しません。')
    if not hist.empty:
        st.caption(f'参考分布：同じCSVの過去の10分前後変化 {len(hist)} 窓。相関とレジーム差、ケースの選択バイアスがあるため有意差検定ではありません。')
    else:st.caption('歴史的な基準窓を確保できませんでした。')
    subsystem,interpretation,check,limit=case_messages(case,r)
    st.markdown('**設備部位・点検支援（操作指示ではありません）**')
    st.info('観測の読み方：'+interpretation)
    st.write('**確認する項目**：'+check)
    st.caption('**判断できないこと**：'+limit)

st.divider()
st.subheader('評価結果の書き出し')
export=measured_frame(latest,[(f'{c}｜{l}',k) for c,groups in {
    'A':[('返送左','FT_8306A'),('返送右','FT_8306B'),('配分比','return_share_L'),('層差圧左右差','layer_LR')],
    'B':[('炉圧中央値','pressure_median'),('ファン電流','YFJ3_AI')],
    'C':[('排ガスO₂','oxygen_mean'),('炉温','TE_8313B')]
}.items() for l,k in groups],hist)
export.insert(0,'csv_latest_timestamp',str(recent));export.insert(1,'comparison_anchor',str(latest_anchor))
st.download_button('最新時刻の観測結果をCSVで保存',data=export.to_csv(index=False,encoding='utf-8-sig').encode('utf-8-sig'),file_name='boiler_latest_observation.csv',mime='text/csv')
with st.expander('モデル・判定の限界と参考ケース'):
    st.write('第3章のケースA/B/Cはイベント起点の事後解析であり、まだ汎化性能が検証された故障分類器ではありません。')
    st.write('上位5%点はCSV内の履歴分布に基づく探索的な参考値です。同じCSVを基準にしているため、厳密な未知期間検証でもなく、物理法則違反を意味しません。')
    st.write('ケースBでファン回転数を自動変更する、ケースCで酸素供給量を自動変更する、といった制御は実装していません。')
    st.write('本ツールは操業判断を支える参考表示です。実設備では設定値、計器校正、運転履歴、安全手順を必ず確認してください。')
