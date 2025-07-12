# barkRelay PoC (Proof of Concept)

音声文字起こしシステムのプロトタイプ実装集

## 概要

このディレクトリには、barkRelayプロジェクトのProof of Concept（概念実証）実装が含まれています。

## ファイル構成

```
poc/
├── README.md                 # このファイル
├── barkrelay_poc.py         # 基本PoC（3秒無音区切り）
├── barkrelay_realtime.py    # リアルタイム版PoC
└── poc_instructions.md      # 実行手順書
```

## PoC版の特徴

### 1. 基本PoC (`barkrelay_poc.py`)
- **特徴**: 3秒間無音で発話区切りを検出
- **処理方式**: バッチ処理
- **出力**: 段階的な詳細表示
- **用途**: 安定した文字起こし、分析機能付き

```bash
# 実行例
python poc/barkrelay_poc.py --model medium
```

### 2. リアルタイム版PoC (`barkrelay_realtime.py`)
- **特徴**: 0.5秒無音で即座に処理
- **処理方式**: ストリーミング処理（マルチスレッド）
- **出力**: リアルタイム表示
- **用途**: 低レイテンシ、対話的な用途

```bash
# 実行例
python poc/barkrelay_realtime.py --audio-device 1
```

## 共通機能

- ✅ Apple Silicon M4最適化
- ✅ faster-whisper使用
- ✅ 日本語音声認識
- ✅ 音声デバイス選択
- ✅ 全文保存・表示
- ✅ AI Agent連携準備

## 使用方法

### 事前準備
```bash
# 仮想環境をアクティベート
source venv/bin/activate

# デバイス一覧確認
python poc/barkrelay_realtime.py --list-devices
```

### 基本PoC実行
```bash
# デフォルト実行
python poc/barkrelay_poc.py

# 高精度モード
python poc/barkrelay_poc.py --model medium --model-improve

# デバッグモード
python poc/barkrelay_poc.py --debug
```

### リアルタイム版実行
```bash
# デフォルト実行
python poc/barkrelay_realtime.py

# 特定デバイス使用
python poc/barkrelay_realtime.py --audio-device 1

# 高精度モード
python poc/barkrelay_realtime.py --model medium
```

## パフォーマンス比較

| 機能 | 基本PoC | リアルタイム版 |
|------|---------|---------------|
| レイテンシ | 3-5秒 | 1-2秒 |
| 処理方式 | シングルスレッド | マルチスレッド |
| 分析機能 | ✅ | ❌ |
| 安定性 | 高 | 中 |
| リアルタイム性 | 低 | 高 |

## 次のステップ

このPoC実装をベースに、本格的なプロダクション版を開発予定：

1. **不完全文章の自動補完** (issue #2)
2. **WebSocket API対応**
3. **AI Agent統合**
4. **Web UI実装**

## トラブルシューティング

詳細な実行手順とトラブルシューティングは `poc_instructions.md` を参照してください。