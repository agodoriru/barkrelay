# barkRelay PoC 実行手順

## 概要
faster_whisperとPyAudioを使用したリアルタイム音声文字起こしのProof of Concept

## セットアップ

### 1. システム依存関係のインストール (macOSの場合)
```bash
# PyAudio用のPortAudioをインストール
brew install portaudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev
```

**Windows:**
PyAudioは通常pipで正常にインストールされます

### 2. Python仮想環境の作成とアクティベート
```bash
# プロジェクトディレクトリに移動
cd barkrelay

# 仮想環境作成
python3 -m venv venv

# 仮想環境をアクティベート
source venv/bin/activate  # Windows: venv\Scripts\activate

# pipをアップグレード
pip install --upgrade pip
```

### 3. 依存関係インストール
```bash
# requirements.txtから一括インストール
pip install -r requirements.txt
```

### 4. インストール確認
```bash
# 主要パッケージのインポートテスト
python -c "import faster_whisper; import pyaudio; import numpy; print('✅ All imports successful')"
```

## 実行方法

**⚠️ 実行前に必ず仮想環境をアクティベートしてください**
```bash
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 基本実行
```bash
python barkrelay_poc.py
```

### オプション付き実行
```bash
# より高精度なモデル使用
python barkrelay_poc.py --model medium

# GPU使用 (CUDA環境の場合)
python barkrelay_poc.py --device cuda --model medium

# 録音チャンク時間変更
python barkrelay_poc.py --duration 3
```

### 実行後の環境終了
```bash
# 仮想環境を終了
deactivate
```

### 使用方法
1. プログラム起動後、利用可能な音声デバイスが表示されます
2. 5秒間隔で音声がキャプチャされ、文字起こしが実行されます
3. 結果がTerminalにリアルタイムで表示されます
4. `Ctrl+C` で終了

### 出力例
```
🎤 barkRelay PoC - Voice to Text Transcription
==================================================
📊 Whisperモデル初期化中... (サイズ: base, デバイス: cpu)
✅ モデル初期化完了

🔊 利用可能な音声入力デバイス:
  0: Built-in Microphone

🚀 文字起こし開始
Ctrl+C で終了
--------------------------------------------------
🎙️ 録音中... (5秒)
🔄 文字起こし中...
[14:30:15] 📝 こんにちは、今日はいい天気ですね
🤖 [AI Agent] 文字起こし結果受信: こんにちは、今日はいい天気ですね...
```

## トラブルシューティング

### PyAudio インストールエラー
```bash
# macOS
brew install portaudio
pip install pyaudio

# エラーが続く場合
pip install pipwin
pipwin install pyaudio
```

### GPU使用時のエラー
CUDA環境が正しくセットアップされているか確認し、必要に応じてCPUモードで実行

### マイクアクセス許可
macOSでは初回実行時にマイクアクセス許可が求められる場合があります

## 次のステップ
- AI Agent連携機能の実装
- WebSocket通信の追加
- 設定ファイル対応
- エラーハンドリング強化