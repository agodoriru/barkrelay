#!/usr/bin/env python3
"""
barkRelay Real-time - Real-time Voice Transcription using faster-whisper
リアルタイム音声文字起こしシステム
"""

import os
import sys
import time
import pyaudio
import numpy as np
import threading
import queue
from datetime import datetime
from faster_whisper import WhisperModel

try:
    import pyaudio
    import numpy as np
    from faster_whisper import WhisperModel
except ImportError as e:
    print(f"❌ 必要なライブラリがインストールされていません: {e}")
    print("\n📦 以下のコマンドでインストールしてください:")
    print("pip install faster-whisper pyaudio numpy")
    sys.exit(1)

class RealtimeTranscriber:
    def __init__(self, model_size="base", device="auto"):
        """
        リアルタイム音声文字起こしシステムの初期化
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        
        # 音声設定 (リアルタイム処理用)
        self.CHUNK = 2048  # 大きなチャンクでより安定した処理
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.BUFFER_SECONDS = 2  # 2秒バッファ（レイテンシとのバランス）
        
        # 音声検出設定
        self.VOLUME_THRESHOLD = 200  # リアルタイム用に調整
        self.MIN_AUDIO_LENGTH = 0.5  # 最小音声長（秒）
        
        # リアルタイム処理用
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_recording = False
        self.audio = None
        
        # 結果管理
        self.partial_results = []
        self.confirmed_results = []
        
        print("🎤 barkRelay Real-time - リアルタイム音声文字起こし")
        print("🚀 Apple Silicon M4最適化対応版")
        print("=" * 60)
    
    def initialize_model(self):
        """Whisperモデルの初期化（Apple Silicon最適化）"""
        print(f"📊 Whisperモデル初期化中... (サイズ: {self.model_size})")
        
        import platform
        
        try:
            if self.device == "auto" and platform.processor() == 'arm' and platform.system() == 'Darwin':
                # Apple Silicon最適化
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=8  # M4の高性能コア
                )
                print("✅ Apple Silicon最適化モードで初期化完了")
            else:
                # 通常モード
                device = "cpu" if self.device == "auto" else self.device
                self.model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type="int8" if device == "cpu" else "float16"
                )
                print(f"✅ {device.upper()}モードで初期化完了")
            
            return True
            
        except Exception as e:
            print(f"❌ モデル初期化エラー: {e}")
            return False
    
    def initialize_audio(self, device_index=None):
        """音声デバイスの初期化"""
        try:
            self.audio = pyaudio.PyAudio()
            
            # 利用可能なデバイスを表示
            input_devices = []
            print("\n🔊 利用可能な音声入力デバイス:")
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    input_devices.append((i, info['name']))
                    print(f"  {i}: {info['name']}")
            
            # デバイスが指定されている場合は検証
            if device_index is not None:
                if device_index < 0 or device_index >= self.audio.get_device_count():
                    print(f"❌ 無効なデバイスインデックス: {device_index}")
                    return False
                
                device_info = self.audio.get_device_info_by_index(device_index)
                if device_info['maxInputChannels'] == 0:
                    print(f"❌ デバイス {device_index} は音声入力をサポートしていません")
                    return False
                
                print(f"✅ 選択されたデバイス: {device_index} - {device_info['name']}")
                self.selected_device = device_index
            else:
                # デバイス未指定の場合はデフォルト
                self.selected_device = None
                print("💡 デフォルトデバイスを使用")
            
            return True
        except Exception as e:
            print(f"❌ 音声デバイス初期化エラー: {e}")
            return False
    
    def detect_voice(self, audio_data):
        """音声検出"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
        return rms > self.VOLUME_THRESHOLD
    
    def audio_capture_worker(self):
        """音声キャプチャワーカー（別スレッド）"""
        # ストリーム設定
        stream_config = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.RATE,
            'input': True,
            'frames_per_buffer': self.CHUNK
        }
        
        # デバイスが指定されている場合は追加
        if hasattr(self, 'selected_device') and self.selected_device is not None:
            stream_config['input_device_index'] = self.selected_device
        
        stream = self.audio.open(**stream_config)
        
        print("🎙️ リアルタイム音声キャプチャ開始...")
        
        frames = []
        silence_count = 0
        is_speaking = False
        
        while self.is_recording:
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                
                # 音声検出
                has_voice = self.detect_voice(data)
                
                if has_voice:
                    if not is_speaking:
                        print("🎤 音声検出", end="", flush=True)
                        is_speaking = True
                        frames = []
                    
                    frames.append(data)
                    silence_count = 0
                    print(".", end="", flush=True)  # 録音中インジケータ
                    
                else:
                    if is_speaking:
                        silence_count += 1
                        frames.append(data)  # 無音部分も少し含める
                        
                        # 十分な無音期間で処理開始
                        if silence_count >= 15:  # 約0.5秒の無音
                            print(" 処理中...")
                            
                            # 音声データをキューに追加
                            if len(frames) >= 8:  # 最小長チェック
                                audio_data = b''.join(frames)
                                self.audio_queue.put(audio_data)
                            
                            # リセット
                            frames = []
                            silence_count = 0
                            is_speaking = False
                            
            except Exception as e:
                print(f"⚠️ 音声キャプチャエラー: {e}")
                break
        
        stream.stop_stream()
        stream.close()
    
    def transcription_worker(self):
        """文字起こしワーカー（別スレッド）"""
        while self.is_recording or not self.audio_queue.empty():
            try:
                # 音声データを取得（タイムアウト付き）
                audio_data = self.audio_queue.get(timeout=1)
                
                # NumPy配列に変換
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # 音声長チェック
                duration = len(audio_float) / self.RATE
                if duration < self.MIN_AUDIO_LENGTH:
                    continue
                
                # 文字起こし実行
                segments, info = self.model.transcribe(
                    audio_float,
                    beam_size=3,  # リアルタイム用に軽量化
                    language="ja",
                    temperature=0.0,
                    condition_on_previous_text=False,
                    initial_prompt="正確に日本語で文字起こししてください。"
                )
                
                # 結果を処理
                full_text = ""
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        full_text += text + " "
                
                if full_text.strip():
                    # 結果をキューに追加
                    self.result_queue.put(full_text.strip())
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 文字起こしエラー: {e}")
    
    def display_worker(self):
        """結果表示ワーカー（メインスレッド）"""
        while self.is_recording or not self.result_queue.empty():
            try:
                # 結果を取得
                result = self.result_queue.get(timeout=0.1)
                
                # タイムスタンプ付きで表示
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{timestamp}] 📝 {result}")
                
                # 結果を保存
                self.confirmed_results.append(result)
                
                # AI Agent連携ポイント
                self.send_to_ai_agent(result)
                
                print("⏳ 次の音声を待機中...", end="", flush=True)
                
            except queue.Empty:
                continue
    
    def send_to_ai_agent(self, text):
        """AI Agentへの送信（未実装）"""
        print(f"\n🤖 [AI Agent] リアルタイム結果: {text[:30]}...")
    
    def start_realtime_transcription(self, device_index=None):
        """リアルタイム文字起こし開始"""
        if not self.initialize_model():
            return
        
        if not self.initialize_audio(device_index):
            return
        
        print("\n🚀 リアルタイム文字起こし開始")
        print("💡 話すとリアルタイムで文字起こしされます")
        print("Ctrl+C で終了")
        print("-" * 60)
        
        self.is_recording = True
        
        # 音声キャプチャスレッド開始
        capture_thread = threading.Thread(target=self.audio_capture_worker)
        capture_thread.daemon = True
        capture_thread.start()
        
        # 文字起こしスレッド開始
        transcription_thread = threading.Thread(target=self.transcription_worker)
        transcription_thread.daemon = True
        transcription_thread.start()
        
        try:
            # メインスレッドで結果表示
            self.display_worker()
            
        except KeyboardInterrupt:
            print("\n\n⏹️ リアルタイム文字起こしを停止中...")
            self.is_recording = False
            
            # スレッドの終了を待機
            capture_thread.join(timeout=2)
            transcription_thread.join(timeout=2)
            
        finally:
            self.cleanup()
    
    def show_session_summary(self):
        """セッション要約表示"""
        if not self.confirmed_results:
            print("📝 文字起こし結果がありません")
            return
        
        print("\n" + "="*60)
        print("📄 セッション要約")
        print("="*60)
        
        full_text = " ".join(self.confirmed_results)
        print(full_text)
        
        print("\n" + "-"*60)
        print(f"📊 統計: {len(self.confirmed_results)}回の発話, {len(full_text)}文字")
        print("="*60 + "\n")
    
    def cleanup(self):
        """リソースクリーンアップ"""
        if self.audio:
            self.audio.terminate()
        
        # セッション要約表示
        self.show_session_summary()
        
        print("✅ クリーンアップ完了")

def main():
    """メイン関数"""
    import argparse
    import platform
    
    parser = argparse.ArgumentParser(description="barkRelay Real-time - リアルタイム音声文字起こし")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], 
                       default="base", help="Whisperモデルサイズ (デフォルト: base)")
    parser.add_argument("--compute-device", choices=["auto", "cpu", "cuda"], default="auto", 
                       help="使用デバイス (デフォルト: auto)")
    parser.add_argument("--volume-threshold", type=int, default=200,
                       help="音声検出の音量閾値 (デフォルト: 200)")
    parser.add_argument("--audio-device", type=int, default=None,
                       help="使用する音声入力デバイスのインデックス番号")
    parser.add_argument("--list-devices", action="store_true",
                       help="利用可能な音声デバイスを表示して終了")
    
    args = parser.parse_args()
    
    # デバイス一覧表示モード
    if args.list_devices:
        try:
            audio = pyaudio.PyAudio()
            print("🔊 利用可能な音声入力デバイス:")
            for i in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    print(f"  {i}: {info['name']}")
            audio.terminate()
        except Exception as e:
            print(f"❌ デバイス一覧取得エラー: {e}")
        return
    
    # システム情報表示
    print(f"🖥️  プラットフォーム: {sys.platform}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔧 プロセッサ: {platform.processor()}")
    print(f"🎯 リアルタイムモード: 有効")
    
    # リアルタイム音声文字起こしインスタンス作成
    transcriber = RealtimeTranscriber(
        model_size=args.model,
        device=args.compute_device
    )
    
    transcriber.VOLUME_THRESHOLD = args.volume_threshold
    
    # リアルタイム文字起こし開始
    transcriber.start_realtime_transcription(device_index=args.audio_device)

if __name__ == "__main__":
    main()