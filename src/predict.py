import argparse
import os
import json
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from model import Classifier
from config import config


def load_audio(audio_path, target_sample_rate=44100, target_duration=1.0):
    """
    Load audio file and resample if necessary.
    
    Args:
        audio_path (str): Path to the audio file
        target_sample_rate (int): Target sample rate
        target_duration (float): Target duration in seconds
    
    Returns:
        torch.Tensor: Audio tensor with shape [channels, samples]
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    
    if sample_rate != target_sample_rate:
        resampler = T.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate
    
    target_length = int(target_duration * sample_rate)
    if waveform.size(1) < target_length:
        padding = target_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :target_length]
    
    return waveform, sample_rate


def generate_spectrogram(waveform, cfg):
    """
    Generate spectrogram from audio waveform.
    
    Args:
        waveform (torch.Tensor): Audio tensor with shape [channels, samples]
        cfg (dict): Configuration dictionary
    
    Returns:
        torch.Tensor: Spectrogram tensor with shape [channels, height, width]
    """
    sample_rate = cfg["sample_rate"]
    n_mels = cfg["n_mels"]
    n_fft = cfg["n_fft"]
    hop_length = cfg["hop_length"]
    feature_type = cfg["feature_type"]
    
    # Normalize waveform (sthandrd normalization for audio)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)
    
    # Generate spectrograms for each channel
    mel_spectrograms = []
    
    if feature_type == "mel":
        mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
        
        for channel in waveform:
            mel_spec = mel_transform(channel)
            mel_spec_db = amplitude_to_db(mel_spec)
            mel_spectrograms.append(mel_spec_db)
    
    elif feature_type == "mfcc":
        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=cfg["n_mfcc"],
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        for channel in waveform:
            mfcc_spec = mfcc_transform(channel)
            mel_spectrograms.append(mfcc_spec)
    
    elif feature_type == "lfcc":
        lfcc_transform = T.LFCC(
            sample_rate=sample_rate,
            n_lfcc=cfg["n_lfcc"],
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        for channel in waveform:
            lfcc_spec = lfcc_transform(channel)
            mel_spectrograms.append(lfcc_spec)
    
    elif feature_type == "stft":
        for channel in waveform:
            stft_spec = torch.stft(
                channel,
                n_fft=n_fft,
                hop_length=hop_length,
                return_complex=False
            )
            # Take magnitude
            stft_mag = torch.sqrt(stft_spec[:, :, 0]**2 + stft_spec[:, :, 1]**2)
            mel_spectrograms.append(stft_mag)
    
    elif feature_type == "bark":
        bark_transform = T.BarkScale(sample_rate=sample_rate, n_barks=cfg["n_barks"])
        spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
        
        for channel in waveform:
            spec = spectrogram(channel)
            bark_spec = bark_transform(spec)
            mel_spectrograms.append(bark_spec)
    
    # Stack spectrograms
    spectrograms = torch.stack(mel_spectrograms, dim=0)  # Shape: [channels, freq_bins, time_frames]
    
    # Normalize spectrograms to [0, 1]
    spec_min = spectrograms.min()
    spec_max = spectrograms.max()
    spectrograms = (spectrograms - spec_min) / (spec_max - spec_min + 1e-9)
    
    # Resize to expected dimensions if needed
    if spectrograms.shape[1] != cfg["height"] or spectrograms.shape[2] != cfg["width"]:
        resize_transform = T.Resize(size=(cfg["height"], cfg["width"]))
        # Convert to 4D for resize (B, C, H, W)
        spectrograms = spectrograms.unsqueeze(0)
        spectrograms = resize_transform(spectrograms)
        spectrograms = spectrograms.squeeze(0)
    
    return spectrograms


def angle_from_sin_cos(sin_val, cos_val):
    """
    Convert sin and cos values back to angle in degrees.
    
    Args:
        sin_val (float): Normalized sine value [0, 1]
        cos_val (float): Normalized cosine value [0, 1]
    
    Returns:
        float: Angle in degrees [0, 360)
    """
    sin_denorm = sin_val * 2.0 - 1.0
    cos_denorm = cos_val * 2.0 - 1.0
    
    angle_rad = np.arctan2(sin_denorm, cos_denorm)
    
    angle_deg = np.degrees(angle_rad)
    
    if angle_deg < 0:
        angle_deg += 360
    
    return angle_deg


def predict(audio_path, model_path, cfg, device):
    """
    Make a prediction for a single audio file.
    
    Args:
        audio_path (str): Path to the input audio file
        model_path (str): Path to the saved model checkpoint
        cfg (dict): Configuration dictionary
        device (torch.device): Device to run inference on
    
    Returns:
        dict: Prediction results
    """
    # Load audio
    print(f"Loading audio from {audio_path}...")
    waveform, sample_rate = load_audio(
        audio_path,
        target_sample_rate=cfg["sample_rate"],
        target_duration=1.0
    )
    print(f"✓ Audio loaded: {waveform.shape[0]} channels, {sample_rate} Hz")
    
    # Generate spectrogram
    print(f"Generating {cfg['feature_type'].upper()} spectrogram...")
    spectrogram = generate_spectrogram(waveform, cfg)
    print(f"✓ Spectrogram generated: {spectrogram.shape}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = Classifier(channel_in=4).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if "classifier_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["classifier_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}")
        raise
    
    # Make prediction
    print("Running inference...")
    model.eval()
    with torch.no_grad():
        # Add batch dimension
        spectrogram_batch = spectrogram.unsqueeze(0).to(device).float()
        
        # Get model output
        output = model(spectrogram_batch).squeeze()
        output = output.cpu().numpy()
    
    print(f"✓ Inference complete")
    
    # Parse output
    a_pred = float(np.clip(1 / (1 + np.exp(-output[0])), 0, 1) * 50)  # Apply sigmoid and scale
    b_pred = float(np.clip(1 / (1 + np.exp(-output[1])), 0, 1) * 50)  # Apply sigmoid and scale
    c_sin_pred = float(np.clip(1 / (1 + np.exp(-output[2])), 0, 1))   # Apply sigmoid
    c_cos_pred = float(np.clip(1 / (1 + np.exp(-output[3])), 0, 1))   # Apply sigmoid
    
    # Convert angles from sin/cos to degrees
    azimuth = angle_from_sin_cos(c_sin_pred, c_cos_pred)
    
    # Direction interpretation
    def get_direction(angle):
        """Convert angle to compass direction."""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        idx = int((angle + 11.25) / 22.5) % 16
        return directions[idx]
    
    results = {
        "distance_meters": a_pred,
        "height_meters": b_pred,
        "azimuth_degrees": azimuth,
        "azimuth_direction": get_direction(azimuth),
        "raw_output": output.tolist()
    }
    
    return results


def print_results(results):
    """
    Pretty print the prediction results.
    
    Args:
        results (dict): Prediction results dictionary
    """
    print("\n" + "="*60)
    print("DRONE LOCATION PREDICTION")
    print("="*60)
    
    print(f"\n📍 DISTANCE FROM MICROPHONE ARRAY:")
    print(f"   {results['distance_meters']:.2f} meters")
    print(f"   (~{results['distance_meters']*3.28:.1f} feet)")
    
    print(f"\n📏 HEIGHT ABOVE GROUND:")
    print(f"   {results['height_meters']:.2f} meters")
    print(f"   (~{results['height_meters']*3.28:.1f} feet)")
    
    print(f"\n🧭 DIRECTION (AZIMUTH):")
    print(f"   {results['azimuth_degrees']:.1f}° ({results['azimuth_direction']})")
    
    print("\n" + "="*60)
    print("VISUAL REPRESENTATION")
    print("="*60)
    print("""
            North (0°)
               ↑
        315°   |   45°
            \  |  /
             \ | /
              \|/
    ←----------+----------→ East (90°)
              /|\
             / | \
        225°   |   135°
               ↓
            South (180°)
    """)
    print(f"Drone is at: {results['azimuth_degrees']:.1f}° ({results['azimuth_direction']})")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Predict drone location from 4-channel microphone array audio'
    )
    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to the input 4-channel audio file (.wav)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to the saved model checkpoint (.pth)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to a custom config JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional: Save results to JSON file'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"✗ Audio file not found: {args.audio}")
        return
    
    if args.config:
        with open(args.config, 'r') as f:
            cfg = json.load(f)
    else:
        cfg = config.copy()
    
    if args.model:
        model_path = args.model
    else:
        model_path = cfg.get("model_path")
        if not model_path or not os.path.exists(model_path):
            model_dir = "/Users/noahmiller/Workspace/AudioTriangulation/saved_models"
            if os.path.exists(model_dir):
                subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
                if subdirs:
                    latest_subdir = sorted(subdirs)[-1]
                    model_files = [f for f in os.listdir(os.path.join(model_dir, latest_subdir)) if f.endswith('.pth')]
                    if model_files:
                        model_path = os.path.join(model_dir, latest_subdir, sorted(model_files)[-1])
    
    if not model_path or not os.path.exists(model_path):
        print(f"✗ Model file not found. Please specify with --model flag")
        print(f"   Usage: python predict.py --audio <audio.wav> --model <model.pth>")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    try:
        results = predict(args.audio, model_path, cfg, device)
        print_results(results)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Results saved to {args.output}\n")
    
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()
