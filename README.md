# Irodori-TTS-600M-v3-VoiceDesign with Google Colab

[Irodori-TTS-600M-v3-VoiceDesign](https://huggingface.co/Aratako/Irodori-TTS-600M-v3-VoiceDesign) を Google Colab 上で手軽に動かすためのノートブックとシンプルなWebUIです。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shinshin86/Irodori-TTS-600M-v3-VoiceDesign-with-colab/blob/main/Irodori_TTS_VoiceDesign.ipynb)

## v3 の特徴

v3 では **参照音声で声をクローンしつつ、キャプションで話し方を同時に制御** できます（3要素制御）。
また Duration Predictor により音声の長さが自動で推定されるため、長さの手動指定は不要です。

## 使い方

1. 上の **Open in Colab** バッジをクリック
2. ランタイムのタイプを **GPU** に変更（メニュー → ランタイム → ランタイムのタイプを変更）
3. セルを上から順に実行
4. 最後のセルで表示される `https://xxx.trycloudflare.com` のURLをクリックしてWebUIにアクセス

## WebUIの機能

- **Text**: 読み上げたい日本語テキストを入力
- **Caption / Style Prompt**: 声質・感情・話し方をテキストで指定（任意）
- **Reference Audio**: 声をクローンしたい参照音声をアップロード（任意）。キャプションと同時に利用可能
- 生成された音声はブラウザ上で再生・ダウンロード可能

### キャプションの例

```
落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。
```

```
明るく元気な男性の声で、はきはきと読み上げてください。
```

### 参照音声 + キャプションの組み合わせ例

参照音声でクローンした声に対して、キャプションで感情や話し方を指定できます。

```
深く傷つき、悲痛なトーンで弱々しく話す。
```

## 必要環境

- Google Colab（GPU ランタイム: T4以上推奨）

## 構成

- `Irodori_TTS_VoiceDesign.ipynb` - Colabノートブック
- `simple_app.py` - シンプルなGradio WebUI

## クレジット

- モデル: [Aratako/Irodori-TTS-600M-v3-VoiceDesign](https://huggingface.co/Aratako/Irodori-TTS-600M-v3-VoiceDesign)
- 推論コード: [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)

## ライセンス

MIT
