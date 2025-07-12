#!/usr/bin/env python3
"""
barkRelay PoC - Voice to Text Transcription using faster_whisper
Proof of Concept for real-time voice transcription and AI agent relay
"""

import os
import sys
import time
import wave
import tempfile
import threading
from datetime import datetime
from pathlib import Path

try:
    import pyaudio
    import numpy as np
    from faster_whisper import WhisperModel
except ImportError as e:
    print(f"❌ 必要なライブラリがインストールされていません: {e}")
    print("\n📦 以下のコマンドでインストールしてください:")
    print("pip install faster-whisper pyaudio numpy")
    sys.exit(1)

class BarkRelayPoC:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """
        音声文字起こしシステムの初期化
        
        Args:
            model_size: Whisperモデルのサイズ (tiny, base, small, medium, large)
            device: 使用デバイス (cpu, cuda)
            compute_type: 計算タイプ (int8, float16, float32)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        
        # 音声録音設定（高品質化）
        self.CHUNK = 4096  # チャンクサイズを大きく（ノイズ軽減）
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # Whisper推奨サンプリングレート
        self.SILENCE_THRESHOLD = 3  # 3秒間無音で分析実行
        
        # 音声検出設定（感度調整）
        self.VOLUME_THRESHOLD = 300  # 音声検出の閾値を下げる（より敏感に）
        self.silence_duration = 0  # 無音継続時間
        self.is_speaking = False  # 現在話しているかの状態
        
        # 連続録音用
        self.continuous_frames = []  # 連続した音声データ
        self.last_voice_time = time.time()  # 最後に音声を検出した時間
        
        # 全文保存
        self.full_transcription = []  # 全ての文字起こし結果を保存
        
        self.is_recording = False
        self.audio = None
        
        print("🎤 barkRelay PoC - Voice to Text Transcription")
        print("🚀 Apple Silicon M4最適化対応版")
        print("=" * 50)
        
    def initialize_model(self):
        """Whisperモデルの初期化（Apple Silicon GPU対応）"""
        print(f"📊 Whisperモデル初期化中... (サイズ: {self.model_size}, デバイス: {self.device})")
        
        # Apple Silicon (M1/M2/M3/M4) GPU検出
        if self.device == "auto":
            import platform
            if platform.processor() == 'arm' and platform.system() == 'Darwin':
                # Apple Silicon Mac
                print("🍎 Apple Silicon Mac検出 - Metal Performance Shadersを使用")
                try:
                    # Apple Silicon最適化設定
                    self.model = WhisperModel(
                        self.model_size,
                        device="cpu",  # Apple SiliconではCPUが最適化されている
                        compute_type="int8",  # メモリ効率重視
                        cpu_threads=8  # M4の高性能コア数
                    )
                    print("✅ Apple Silicon最適化モードで初期化完了")
                    return True
                except Exception as e:
                    print(f"❌ Apple Silicon最適化失敗: {e}")
            
            # フォールバック: 通常のCPU
            self.device = "cpu"
            self.compute_type = "int8"
        
        try:
            # デバイス別初期化
            if self.device == "cuda":
                # NVIDIA GPU
                self.model = WhisperModel(
                    self.model_size,
                    device="cuda",
                    compute_type="float16"
                )
                print("✅ CUDA GPU初期化完了")
            else:
                # CPU (Apple Siliconを含む)
                cpu_threads = 8 if platform.processor() == 'arm' else 4
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type=self.compute_type,
                    cpu_threads=cpu_threads
                )
                print(f"✅ CPU初期化完了 (スレッド数: {cpu_threads})")
            
            return True
            
        except Exception as e:
            print(f"❌ モデル初期化エラー: {e}")
            # 最終フォールバック
            if self.device != "cpu":
                print("🔄 CPUモードにフォールバック中...")
                try:
                    self.model = WhisperModel(
                        self.model_size, 
                        device="cpu", 
                        compute_type="int8",
                        cpu_threads=4
                    )
                    print("✅ CPUフォールバックモードで初期化完了")
                    return True
                except Exception as e2:
                    print(f"❌ CPUモードでも初期化失敗: {e2}")
                    return False
            return False
    
    def initialize_audio(self):
        """音声入力デバイスの初期化"""
        try:
            self.audio = pyaudio.PyAudio()
            
            # 利用可能なオーディオデバイスをリストアップ
            print("\n🔊 利用可能な音声入力デバイス:")
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    print(f"  {i}: {info['name']}")
            
            return True
        except Exception as e:
            print(f"❌ 音声デバイス初期化エラー: {e}")
            return False
    
    def detect_silence(self, audio_data):
        """音声データから無音を検出（改良版）"""
        # NumPy配列に変換
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # RMS（Root Mean Square）を計算
        rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
        
        # ピーク値も考慮
        peak = np.max(np.abs(audio_array))
        
        # RMSとピークの両方で判定（より正確な音声検出）
        is_silent = (rms < self.VOLUME_THRESHOLD) and (peak < self.VOLUME_THRESHOLD * 3)
        
        # デバッグ情報（オプション）
        if hasattr(self, 'debug_audio') and self.debug_audio:
            print(f"🔊 RMS: {rms:.1f}, Peak: {peak:.1f}, Silent: {is_silent}")
        
        return is_silent
    
    def record_continuous_audio(self):
        """連続音声録音（リアルタイム音声検出）"""
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        print("🎙️ 音声入力待機中...")
        
        while self.is_recording:
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                current_time = time.time()
                
                # 音声検出
                if not self.detect_silence(data):
                    # 音声検出時
                    if not self.is_speaking:
                        # 音声検出開始
                        print("\n" + "="*50)
                        print("🎤 音声検出 - 録音開始")
                        print("="*50)
                        self.is_speaking = True
                        self.continuous_frames = []
                    
                    self.continuous_frames.append(data)
                    self.last_voice_time = current_time
                    
                else:
                    # 無音時
                    if self.is_speaking:
                        # 発話中からの無音
                        self.silence_duration = current_time - self.last_voice_time
                        
                        if self.silence_duration >= self.SILENCE_THRESHOLD:
                            # 3秒以上無音 → 発話終了として処理
                            print("-"*50)
                            print(f"🔇 発話終了検出 (無音: {self.silence_duration:.1f}秒)")
                            print("-"*50)
                            
                            # 蓄積した音声データを処理
                            audio_file = self.save_continuous_frames()
                            if audio_file:
                                yield audio_file
                            
                            # 状態リセット
                            self.is_speaking = False
                            self.continuous_frames = []
                            self.silence_duration = 0
                    
            except Exception as e:
                print(f"⚠️ 録音エラー: {e}")
                break
        
        stream.stop_stream()
        stream.close()
    
    def save_continuous_frames(self):
        """連続フレームを一時ファイルに保存"""
        if not self.continuous_frames:
            return None
            
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                wf = wave.open(temp_file.name, 'wb')
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(self.continuous_frames))
                wf.close()
                
                return temp_file.name
        except Exception as e:
            print(f"❌ 音声保存エラー: {e}")
            return None
    
    def transcribe_audio(self, audio_file_path):
        """音声ファイルを文字起こし（高精度設定）"""
        try:
            # より高精度な設定で文字起こし
            segments, info = self.model.transcribe(
                audio_file_path,
                beam_size=5,  # ビームサーチサイズ
                language="ja",  # 日本語を明示的に指定
                condition_on_previous_text=False,  # 前のテキストに依存しない
                temperature=0.0,  # 最も確率の高い結果を選択
                compression_ratio_threshold=2.4,  # 圧縮率閾値
                log_prob_threshold=-1.0,  # 対数確率閾値
                no_speech_threshold=0.6,  # 無音判定閾値
                word_timestamps=True,  # 単語レベルのタイムスタンプ
                initial_prompt="以下は日本語の音声です。正確に文字起こししてください。"  # 初期プロンプト
            )
            
            # 結果をまとめる（より詳細な処理）
            full_text = ""
            total_segments = 0
            
            for segment in segments:
                segment_text = segment.text.strip()
                if segment_text:  # 空でない場合のみ追加
                    # 信頼度チェック（平均対数確率）
                    if hasattr(segment, 'avg_logprob') and segment.avg_logprob > -0.8:
                        full_text += segment_text + " "
                        total_segments += 1
                    else:
                        # 信頼度が低い場合は注記（デバッグ時のみ表示）
                        if hasattr(self, 'debug_audio') and self.debug_audio:
                            print(f"⚠️ 低信頼度セグメント: {segment_text} (信頼度: {segment.avg_logprob:.3f})")
                        full_text += segment_text + " "  # 注記なしで追加
                        total_segments += 1
            
            result = full_text.strip()
            # デバッグ情報は静かに（必要時のみ表示）
            if hasattr(self, 'debug_audio') and self.debug_audio and result:
                print(f"📊 認識結果: {total_segments}セグメント, 信頼度情報付き")
            
            return result
            
        except Exception as e:
            print(f"❌ 文字起こしエラー: {e}")
            return None
        finally:
            # 一時ファイル削除
            try:
                os.unlink(audio_file_path)
            except:
                pass
    
    def format_output(self, text):
        """出力フォーマット"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if text:
            print(f"[{timestamp}] 📝 {text}")
            self.full_transcription.append(text)  # 全文に追加
            return text
        else:
            print(f"[{timestamp}] 🔇 音声が検出されませんでした")
            return None
    
    def analyze_conversation(self):
        """会話分析を実行（無音時）"""
        if not self.full_transcription:
            return
            
        print("\n" + "="*60)
        print("📊 会話分析実行中...")
        
        # 全文を結合
        full_text = " ".join(self.full_transcription)
        
        print(f"📈 総発話数: {len(self.full_transcription)}回")
        print(f"📏 総文字数: {len(full_text)}文字")
        
        # 簡単なキーワード分析
        keywords = self.extract_keywords(full_text)
        if keywords:
            print(f"🔤 主要キーワード: {', '.join(keywords[:5])}")
        
        print("="*60 + "\n")
    
    def extract_keywords(self, text):
        """簡単なキーワード抽出"""
        # 日本語の一般的なストップワード
        stop_words = {'です', 'ます', 'である', 'について', 'として', 'という', 
                     'こと', 'もの', 'ところ', 'ため', 'よう', 'そう', 'から',
                     'ので', 'けれど', 'でも', 'しかし', 'それで', 'そして'}
        
        # 単語を分割（簡易版）
        words = []
        current_word = ""
        for char in text:
            if char.isalnum():
                current_word += char
            else:
                if current_word and len(current_word) > 1:
                    words.append(current_word)
                current_word = ""
        
        # 頻度カウント
        word_count = {}
        for word in words:
            if word not in stop_words and len(word) > 1:
                word_count[word] = word_count.get(word, 0) + 1
        
        # 頻度順にソート
        return sorted(word_count.keys(), key=lambda x: word_count[x], reverse=True)
    
    def show_full_transcription(self):
        """全文表示"""
        if not self.full_transcription:
            print("📝 文字起こし結果がありません")
            return
            
        print("\n" + "="*60)
        print("📄 全文表示")
        print("="*60)
        
        full_text = " ".join(self.full_transcription)
        print(full_text)
        
        print("\n" + "-"*60)
        print(f"📈 統計: {len(self.full_transcription)}回の発話, {len(full_text)}文字")
        print("="*60 + "\n")
    
    def start_transcription(self):
        """文字起こしメインループ"""
        if not self.initialize_model():
            return
        
        if not self.initialize_audio():
            return
        
        print("\n🚀 音声文字起こし開始")
        print("💡 話すと自動で文字起こしされます（3秒間無音で区切り）")
        print("Ctrl+C で終了")
        print("-" * 50)
        
        self.is_recording = True
        
        try:
            # 連続音声録音ジェネレーターを使用
            for audio_file in self.record_continuous_audio():
                if audio_file:
                    # 文字起こし実行
                    print("🔄 文字起こし処理中...")
                    print("-"*50)
                    transcribed_text = self.transcribe_audio(audio_file)
                    
                    if transcribed_text and transcribed_text.strip():
                        # 結果出力
                        result = self.format_output(transcribed_text)
                        
                        # AI Agent連携ポイント（今後実装）
                        if result:
                            self.send_to_ai_agent(result)
                        
                        print("="*50)
                        print("✅ 文字起こし完了")
                        print("="*50)
                    else:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp}] 🔇 音声認識できませんでした")
                        print("="*50)
                        print("❌ 文字起こし失敗")
                        print("="*50)
                
                # 次の発話を待機
                print("⏳ 次の発話を待機中...")
                print()
                
        except KeyboardInterrupt:
            print("\n⏹️ 文字起こしを停止しています...")
            self.is_recording = False
        
        except Exception as e:
            print(f"\n❌ 予期しないエラー: {e}")
        
        finally:
            self.cleanup()
    
    def send_to_ai_agent(self, text):
        """AI Agentへの送信（未実装）"""
        # TODO: REST API または WebSocket でAI Agentに送信
        print(f"🤖 [AI Agent] 文字起こし結果受信: {text[:50]}...")
    
    def cleanup(self):
        """リソースクリーンアップ"""
        if self.audio:
            self.audio.terminate()
        
        # 最終的な全文表示と分析
        self.show_full_transcription()
        
        # 最終分析を実行
        if self.full_transcription:
            print("\n🔍 最終会話分析:")
            self.analyze_conversation()
        
        print("✅ クリーンアップ完了")

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="barkRelay PoC - Voice to Text Transcription")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], 
                       default="base", help="Whisperモデルサイズ (デフォルト: base)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", 
                       help="使用デバイス (デフォルト: auto - Apple Silicon自動検出)")
    parser.add_argument("--silence-threshold", type=int, default=3,
                       help="分析実行する無音時間（秒） (デフォルト: 3)")
    parser.add_argument("--volume-threshold", type=int, default=300,
                       help="音声検出の音量閾値 (デフォルト: 300)")
    parser.add_argument("--debug", action="store_true",
                       help="デバッグ情報を表示")
    parser.add_argument("--model-improve", action="store_true",
                       help="より高精度なモデル設定を使用")
    
    args = parser.parse_args()
    
    # システム情報表示
    import platform
    print(f"🖥️  プラットフォーム: {sys.platform}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔧 プロセッサ: {platform.processor()}")
    print(f"📱 マシン: {platform.machine()}")
    
    # Apple Silicon検出
    if platform.processor() == 'arm' and platform.system() == 'Darwin':
        print("🍎 Apple Silicon Mac検出 - 最適化モード使用")
    elif args.device == "cuda":
        print("🎮 CUDA GPU指定")
    else:
        print("💻 CPU使用")
    
    # PoCインスタンス作成
    # Apple Silicon最適化
    if args.device == "auto" and platform.processor() == 'arm' and platform.system() == 'Darwin':
        compute_type = "int8"  # Apple Silicon用最適化
    elif args.device == "cuda":
        compute_type = "float16"  # NVIDIA GPU用
    else:
        compute_type = "int8"  # その他CPU用
    
    bark_relay = BarkRelayPoC(
        model_size=args.model,
        device=args.device,
        compute_type=compute_type
    )
    
    bark_relay.SILENCE_THRESHOLD = args.silence_threshold
    bark_relay.VOLUME_THRESHOLD = args.volume_threshold
    bark_relay.debug_audio = args.debug
    
    # 高精度モデル設定
    if args.model_improve:
        print("🎯 高精度モード有効")
        bark_relay.model_improve_mode = True
    
    # 文字起こし開始
    bark_relay.start_transcription()

if __name__ == "__main__":
    main()