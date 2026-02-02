import os
import sys
import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from pathlib import Path

# --- Add EnCodec clone to Python path ---
sys.path.insert(0, '/Users/yinanli/Desktop/Sound2Hap/Sound2Hap_Models/encodec')

from encodec.model import EncodecModel
from encodec.modules.seanet import SEANetDecoder

class AudioToVibrationInference:
    def __init__(self, model_path, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        self.model = EncodecModel.encodec_model_24khz().to(self.device)
        self.model.set_target_bandwidth(6.0)

        old = self.model.decoder
        custom_decoder = SEANetDecoder(
            channels=1,
            dimension=old.dimension,
            n_filters=old.n_filters,
            n_residual_layers=old.n_residual_layers,
            ratios=old.ratios,
            activation='LeakyReLU', 
            activation_params={'negative_slope':0.2},
            final_activation=None, 
            norm='weight_norm', 
            lstm=2,
        ).to(self.device)
        self.model.decoder = custom_decoder
        
        print(f"Loading model from: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.audio_resampler = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=24000
        ).to(self.device)
        
        print("Model loaded and ready for inference!")
    
    def preprocess_audio(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        audio = audio.to(self.device)
        if sr != 24000:
            audio = self.audio_resampler(audio)
        
        audio_max = audio.abs().max()
        if audio_max > 0:
            audio = audio / audio_max
        
        return audio.unsqueeze(0)
    
    def generate_vibration(self, audio_tensor):
        with torch.no_grad():
            z = self.model.encoder(audio_tensor)
            quantized_result = self.model.quantizer(z, frame_rate=self.model.frame_rate)
            zq = quantized_result.quantized
            vib_pred = self.model.decoder(zq)
        return vib_pred
    
    ### MODIFIED POST-PROCESSING ###
    def postprocess_vibration(self, vib_tensor, output_sample_rate=8000):
        """
        Post-processes the raw model output with a proper audio effects chain.
        """
        print(f"Postprocessing vibration...")
        
        # The model outputs at 24kHz
        sample_rate = 24000
        vib = vib_tensor.squeeze(0).cpu() # Shape: [1, T]
        
        # 1. DC Blocking Filter: Remove any DC offset
        vib = F.highpass_biquad(vib, sample_rate, cutoff_freq=10)
        print("Applied DC blocking filter.")

        # 2. Band-limiting Filter: Constrain to a typical haptic frequency range
        vib = F.highpass_biquad(vib, sample_rate, cutoff_freq=20, Q=0.707) # Low-cut
        vib = F.lowpass_biquad(vib, sample_rate, cutoff_freq=400, Q=0.707)  # High-cut
        print("Applied band-limiting filter (20-400 Hz).")

        # 3. Soft Limiter: Gently prevent harsh clipping
        vib = torch.tanh(vib)
        print("Applied soft limiter (tanh).")

        # 4. Resample to the final target sample rate
        if output_sample_rate != sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=output_sample_rate
            )
            vib = resampler(vib)
            print(f"Resampled to {output_sample_rate}Hz.")
        
        # 5. Final Peak Normalization
        vib_max = vib.abs().max()
        if vib_max > 0:
            vib = vib / vib_max
        print("Applied final peak normalization.")

        return vib
    ### END MODIFICATION ###

    def inference(self, audio_path, output_path=None, output_sample_rate=8000):
        print(f"\nStarting inference for: {audio_path}")
        audio_tensor = self.preprocess_audio(audio_path)
        vib_tensor = self.generate_vibration(audio_tensor)
        vib_output = self.postprocess_vibration(vib_tensor, output_sample_rate)
        if output_path:
            self.save_vibration(vib_output, output_path, output_sample_rate)
        print("Inference completed!")
        return vib_output
    
    def save_vibration(self, vib_tensor, output_path, sample_rate):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, vib_tensor, sample_rate)
        print(f"Vibration saved to: {output_path}")
    
    def batch_inference(self, input_folder, output_folder, output_sample_rate=8000):
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = [p for ext in audio_extensions for p in input_path.glob(f'**/*{ext}')]
        
        print(f"Found {len(audio_files)} audio files in {input_folder}")
        
        for i, audio_file in enumerate(audio_files):
            relative_path = audio_file.relative_to(input_path)
            output_filename = f"{audio_file.stem}-vib-model1.wav"
            output_file_path = output_path / relative_path.parent / output_filename
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"\n[{i+1}/{len(audio_files)}] Processing: {audio_file}")
            try:
                self.inference(str(audio_file), str(output_file_path), output_sample_rate)
            except Exception as e:
                print(f"Error processing {audio_file.name}: {e}")
                continue
        
        print(f"\nBatch inference completed! Results saved in: {output_folder}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate vibrations from audio using trained model')
    parser.add_argument('--model', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--input', required=True, help='Input audio file or folder')
    parser.add_argument('--output', required=True, help='Output vibration file or folder')
    parser.add_argument('--sample_rate', type=int, default=8000, help='Output sample rate (default: 8000)')
    parser.add_argument('--batch', action='store_true', help='Batch mode for folder processing')
    args = parser.parse_args()
    
    inference_model = AudioToVibrationInference(args.model)
    if args.batch:
        inference_model.batch_inference(args.input, args.output, args.sample_rate)
    else:
        inference_model.inference(args.input, args.output, args.sample_rate)
