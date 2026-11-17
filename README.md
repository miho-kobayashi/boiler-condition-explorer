# Boiler Condition Explorer

公開CFBボイラー運転データを用いた、特徴的な波形・点検候補の探索ダッシュボード。

## 起動

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloudへの公開

1. このフォルダ内のファイルをGitHubのリポジトリ直下にアップロードする。
2. https://share.streamlit.io/ にGitHubでログインする。
3. `Create app` → `Yup, I have an app` を選択する。
4. 対象のGitHubリポジトリ、ブランチ `main`、エントリーポイント `app.py` を選択し、デプロイする。
5. 発行された `https://...streamlit.app` のURLを共有する。

## 入力データ

Kaggle *Time-Series of Industrial Boiler Operations* の `data.csv`（日時列＋計測タグ）に対応。
サンプルで動作確認も可能。データはアップロードした利用者のセッション内で解析する。

## 注意

- 本ツールは設備故障の確定診断・安全制御のためのものではない。
- 同じCSV内の履歴に基づく探索しきい値は、故障確率や統計的有意性を示さない。
- ダッシュボードの「最新時刻」はアップロードしたCSV内の最終日時であり、現在時刻ではない。
- 公開環境に機密・顧客・個人データをアップロードしないこと。
- 出典: https://www.kaggle.com/datasets/nikitamanaenkov/time-series-of-industrial-boiler-operations
