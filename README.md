# barkRelay

音声入力を文字起こしして、AI agentに渡すシステム

## 概要

BarkRelayは音声入力を受け取り、それを文字に変換してAI agentに送信するための中継システムです。

## 機能

- [x] 音声入力キャプチャ
- [x] リアルタイム音声認識・文字起こし
- [ ] AI agentとの連携インターフェース
- [x] Apple Silicon M4最適化
- [x] 複数音声デバイス対応

## 開発方針

- 全ての機能はGitHub issueとして管理
- 機能実装時は専用ブランチを作成してPull Requestで進行

## 技術スタック

- 音声処理: PyAudio + faster-whisper
- 音声認識: OpenAI Whisper (faster-whisper実装)
- AI連携: 検討中
- 最適化: Apple Silicon M4対応

## セットアップ

```bash
# プロジェクトをクローン
git clone <repository-url>
cd barkrelay

# 仮想環境作成とアクティベート
python3 -m venv venv
source venv/bin/activate

# システム依存関係 (macOS)
brew install portaudio

# Python依存関係インストール
pip install -r requirements.txt

# PoC実行
python poc/barkrelay_poc.py
```

## PoC (Proof of Concept)

`poc/` ディレクトリに実装サンプルがあります：

- **基本版**: `poc/barkrelay_poc.py` - 安定した文字起こし
- **リアルタイム版**: `poc/barkrelay_realtime.py` - 低レイテンシ処理

詳細は [poc/README.md](poc/README.md) を参照してください。
