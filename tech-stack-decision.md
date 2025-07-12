# barkRelay 技術スタック決定書

## 概要
faster_whisperとwhisper_micを使用したCUIベース音声文字起こしシステムの技術選定結果

## 技術スタック

### 音声処理・文字起こし
- **主要エンジン**: faster-whisper (OpenAI Whisperの最適化版)
  - 4倍高速、メモリ効率向上
  - GPU/CPU両対応
  - モデルサイズ: medium推奨 (コスト/性能バランス)

- **マイク入力**: whisper_mic + PyAudio
  - リアルタイムマイク入力キャプチャ
  - Voice Activity Detection (VAD) 統合
  - 音声前処理機能

### バックエンド/CUI
- **言語**: Python 3.8-3.11
- **インターフェース**: CLI (Command Line Interface)
- **出力**: Terminal表示 + ログファイル
- **AI連携**: REST API / WebSocket クライアント

### 依存関係
```python
# コア依存関係
faster-whisper>=0.10.0
whisper-mic
click>=8.0  # CLI framework

# 音声処理
pyaudio>=0.2.11
sounddevice>=0.4.6
webrtcvad>=2.0.10
numpy>=1.21.0

# AI連携
requests>=2.31.0  # REST API client
websockets>=11.0  # WebSocket client

# オプショナル (GPU使用時)
nvidia-cublas-cu12
nvidia-cudnn-cu12==9.*
```

## アーキテクチャ設計

### システム構成
```
[マイク] → [whisper_mic] → [VAD] → [faster_whisper] → [Terminal出力]
                                                     ↓
                                                [AI Agent API]
```

### CUIインターフェース
- **起動**: `python barkrelay.py --start`
- **モデル選択**: `--model medium|large`
- **出力先**: `--output terminal|file|api`
- **AI連携**: `--ai-endpoint http://localhost:8080/chat`

### データフロー
1. **音声キャプチャ**: whisper_micでマイク入力を5秒チャンクで取得
2. **音声検出**: VADで音声区間を特定
3. **文字起こし**: faster_whisperで文字変換
4. **Terminal表示**: リアルタイムでコンソール出力
5. **AI連携**: 設定されている場合はAPI送信

## 開発環境セットアップ

### システム要件
- **CPU**: マルチコア推奨 (Intel i7以上)
- **メモリ**: 4-8GB (mediumモデル使用時)
- **GPU**: CUDA 12+ 対応 (オプショナル、4倍高速化)
- **OS**: macOS, Linux, Windows

### セットアップ手順
```bash
# 1. Pythonプロジェクト初期化
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 依存関係インストール
pip install faster-whisper whisper-mic click
pip install pyaudio sounddevice webrtcvad numpy requests websockets

# 3. GPU使用時 (オプショナル)
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*

# 4. 起動
python barkrelay.py --start
```

## 実装計画

### フェーズ1: 基本CUIアプリ
- [ ] Clickベースの基本CLI構造
- [ ] faster_whisper統合
- [ ] Terminal出力機能
- [ ] 設定ファイル管理

### フェーズ2: 音声入力
- [ ] whisper_mic統合
- [ ] VAD実装
- [ ] リアルタイム処理パイプライン

### フェーズ3: AI Agent連携
- [ ] REST API クライアント実装
- [ ] WebSocket クライアント実装
- [ ] 設定による切り替え機能

### フェーズ4: 最適化
- [ ] パフォーマンスチューニング
- [ ] ログ機能充実
- [ ] エラーハンドリング強化

## CUI仕様

### 基本使用例
```bash
# 基本起動
python barkrelay.py

# モデル指定
python barkrelay.py --model large

# AI連携有効
python barkrelay.py --ai-endpoint http://localhost:8080/api/chat

# ログファイル出力
python barkrelay.py --log-file transcription.log

# 設定表示
python barkrelay.py --config
```

### 出力形式
```
[2024-07-12 14:30:15] 🎤 音声入力開始...
[2024-07-12 14:30:18] 📝 こんにちは、今日はいい天気ですね
[2024-07-12 14:30:18] 🤖 AI Agent送信: ✓
[2024-07-12 14:30:20] 📝 明日の予定について話したいと思います
```

## 次のアクション
1. 基本的なCLIアプリケーション実装
2. faster_whisperテスト環境構築
3. Terminal出力とログ機能実装