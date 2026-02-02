% Sound2Hap - Pitch Matching Algorithm Implementation
% Based on IEEE Transactions on Haptics 2024 Paper by Kim et al. Sound-to-Touch Crossmodal Pitch Matching.
% Optimized for long audio files (>1s) using time-varying analysis
% MATLAB implementation is used for the study described in our user study.

%% MAIN EXECUTION
% --- USER: Define your single input and output files here ---
inputFilePath = 'test.wav'; 
outputFilePath = 'test-pitch.wav'; 

% --- Processing Logic ---
fprintf('Starting single file processing...\n');
config = getConfig();

% Ensure the output directory exists before writing the file
[outputPath, ~, ~] = fileparts(outputFilePath);
if ~exist(outputPath, 'dir') && ~isempty(outputPath)
    mkdir(outputPath);
    fprintf('Created output directory: %s\n', outputPath);
end

% Process the single audio file
try
    startTime = tic;
    [success, result] = processAudioFile(inputFilePath, outputFilePath, config);
    if success
        fprintf('\nProcessing complete.\n');
        fprintf('Input:  %s\n', result.inputFile);
        fprintf('Output: %s\n', result.outputFile);
        duration = toc(startTime);
        % Print the execution time in a machine-readable format for Python
        fprintf('ExecutionTime:%.4f\n', duration);
    else
        fprintf('\nProcessing failed for: %s\n', inputFilePath);
    end
catch ME
    fprintf('\nAn unexpected error occurred during processing:\n');
    fprintf('%s\n', ME.message);
end


%% Configuration Parameters
function config = getConfig()
    config.regressionCoeffs = containers.Map([2, 3, 9, 12, 24], [-0.005, 0.003, -0.015, 0.008, 0.008]);
    config.vibrationFreqRange = [50, 398];
    config.binSizeMs = 10;            
    config.overlapRatio = 0.5;
    config.smoothingWindow = 3; % Frequency smoothing window    
    config.inputSampleRate = 44100;
    config.outputSampleRate = 8000;
end

%% Specific Loudness Computation (Spectral → Pitch)
function specificLoudness = computeSpecificLoudnessBarkBands(audio, sr, config)
    % Compute specific loudness in 24 Bark bands using ISO 532-1
    if isempty(audio) || length(audio) < 2
        specificLoudness = zeros(1, 24);
        return;
    end
    if size(audio, 2) > size(audio, 1)
        audio = audio';
    end
    if rms(audio) < 1e-6
        specificLoudness = zeros(1, 24);
        return;
    end
    
    [~, specificLoudnessMatrix] = acousticLoudness(audio, sr, ...
        'Method', 'ISO 532-1', ...
        'SoundField', 'free', ...
        'TimeVarying', false);
    
    % specificLoudnessMatrix should be 1x240 (0.1 to 24 Bark in 0.1 increments)
    if isempty(specificLoudnessMatrix)
        specificLoudness = zeros(1, 24);
        return;
    end
    
    % Integrate into 24 Bark bands
    if size(specificLoudnessMatrix, 2) == 240
        specificLoudness = zeros(1, 24);
        for i = 1:24
            startIdx = (i-1) * 10 + 1;
            endIdx = min(i * 10, 240);
            if startIdx <= 240
                specificLoudness(i) = sum(specificLoudnessMatrix(startIdx:endIdx));
            end
        end
    else
        % Handle unexpected sizes by resampling
        if length(specificLoudnessMatrix) > 24
            indices = round(linspace(1, length(specificLoudnessMatrix), 24));
            specificLoudness = specificLoudnessMatrix(indices);
        else
            specificLoudness = interp1(1:length(specificLoudnessMatrix), ...
                                     specificLoudnessMatrix, ...
                                     linspace(1, length(specificLoudnessMatrix), 24), ...
                                     'linear', 'extrap');
        end
    end
        
    % Ensure output is row vector and no NaN/Inf values
    specificLoudness = specificLoudness(:)';
    specificLoudness(~isfinite(specificLoudness)) = 0;
end

%% Vibration Frequency Prediction
function vibFreq = predictVibrationFrequency(specificLoudness, config)
    % Apply paper's regression using specific loudness features
    
    predictedFreq = 0; % Start at zero, no base frequency
    
    barkBands = keys(config.regressionCoeffs);
    coeffs = values(config.regressionCoeffs);
    
    % Apply regression coefficients to specific Bark bands only
    for i = 1:length(barkBands)
        barkBand = barkBands{i};
        coeff = coeffs{i};
        
        if barkBand <= length(specificLoudness)
            % Direct linear combination as per equation (10)
            predictedFreq = predictedFreq + (coeff * specificLoudness(barkBand))*1000;
        end
    end
    
    predictedFreq = abs(predictedFreq);
    
    %baseFreq = mean(config.vibrationFreqRange);
    %predictedFreq = baseFreq + predictedFreq; 

    % Clamp to valid vibration frequency range [50, 398] Hz
    vibFreq = max(config.vibrationFreqRange(1), min(config.vibrationFreqRange(2), predictedFreq));
    
    %fprintf('  Raw prediction: %.5f Hz -> Clamped: %.1f Hz\n', predictedFreq, vibFreq);
end

%% Time-Varying Analysis for Long Audio
function [binTimes, binFrequencies, binAmplitudes] = analyzeAudioBins(audio, sr, config)
    % Break long audio into time bins and analyze each bin
    binSizeSamples = round(config.binSizeMs * sr / 1000);
    hopSizeSamples = round(binSizeSamples * (1 - config.overlapRatio));
    
    binStarts = 1:hopSizeSamples:(length(audio) - binSizeSamples + 1);
    if isempty(binStarts)
        binStarts = 1;
    end
    
    binStarts = binStarts(:)';
    binTimes = (binStarts + binSizeSamples/2 - 1) / sr;
    binTimes = binTimes(:)';
    numBins = length(binStarts);
    binFrequencies = zeros(1, numBins);
    binAmplitudes = zeros(1, numBins);
    window = hann(binSizeSamples);
    
    for i = 1:numBins
        startIdx = binStarts(i);
        endIdx = min(startIdx + binSizeSamples - 1, length(audio));
        audioBin = audio(startIdx:endIdx);

        if length(audioBin) < binSizeSamples
            audioBin = [audioBin; zeros(binSizeSamples - length(audioBin), 1)];
        end
        
        audioBin = audioBin .* window;
        
        if rms(audioBin) < 0.001
            if i > 1
                binFrequencies(i) = binFrequencies(i-1);
                binAmplitudes(i) = 0;
            else
                binFrequencies(i) = mean(config.vibrationFreqRange);
                binAmplitudes(i) = 0;
            end
            continue;
        end
        
        % PITCH PREDICTION: Use specific loudness (spectral features)
        specificLoudness = computeSpecificLoudnessBarkBands(audioBin, sr, config);
        vibFreq = predictVibrationFrequency(specificLoudness, config);
        binFrequencies(i) = vibFreq;
        
        % AMPLITUDE MODULATION: Use total loudness (single value)
        loudness = acousticLoudness(audioBin, sr, ...
            'Method', 'ISO 532-1', ...
            'SoundField', 'free', ...
            'TimeVarying', false);  % Single loudness value
        binAmplitudes(i) = loudness;

    end
    
    % Smooth frequency transitions to avoid abrupt changes
    if config.smoothingWindow > 1 && length(binFrequencies) > config.smoothingWindow
        binFrequencies = movmean(binFrequencies, config.smoothingWindow);
    end
    
    % fprintf('    Frequency range: %.1f - %.1f Hz (mean: %.1f Hz)\n', ...
    %     min(binFrequencies), max(binFrequencies), mean(binFrequencies));
end

%% Multi-Method Amplitude Envelope Generation
function [vibrationSignal, frequenciesArray, amplitudeArray] = generateTimeVaryingVibration(audio, sr, config)
    [binTimes, binFrequencies, binAmplitudes] = analyzeAudioBins(audio, sr, config);
    if size(audio, 2) > size(audio, 1)
        audio = audio';
    end
    
    t = (0:length(audio)-1)' / sr;
    
    binTimes = binTimes(:)';
    binFrequencies = binFrequencies(:)';
    binAmplitudes = binAmplitudes(:)';
    
    if length(binTimes) == 1
        frequenciesArray = repmat(binFrequencies, size(t));
    else
        try
            frequenciesArray = interp1(binTimes, binFrequencies, t', 'pchip', 'extrap');
        catch
            frequenciesArray = interp1(binTimes, binFrequencies, t', 'nearest', 'extrap');
        end
    end
    frequenciesArray = frequenciesArray(:);
    
    amplitudeMethodUsed = 'unknown';
    amplitudeArray = [];
    
    % This is the true "total loudness per bin → envelope" approach    
    if length(binTimes) == 1
        % Single bin case
        amplitudeArray = repmat(binAmplitudes, size(t));
    else
        % Interpolate bin-based total loudness values to full resolution
        amplitudeArray = interp1(binTimes, binAmplitudes, t', 'linear', 'extrap');
        amplitudeArray = amplitudeArray(:);
    end
    
    % Check if we got valid amplitude data
    if ~isempty(amplitudeArray) && any(amplitudeArray > 0)
        amplitudeMethodUsed = 'bin-total-loudness';
        % fprintf('SUCCESS\n');
    else
        amplitudeArray = [];
        fprintf('FAILED (no valid amplitude data)\n');
    end

    
    % Ensure amplitude array is proper size and type
    amplitudeArray = amplitudeArray(:);
    if length(amplitudeArray) ~= length(t)
        amplitudeArray = interp1(1:length(amplitudeArray), amplitudeArray, linspace(1, length(amplitudeArray), length(t)), 'linear', 'extrap');
        amplitudeArray = amplitudeArray(:);
    end

    rms_current = sqrt(mean(amplitudeArray.^2) + 1e-12);
    % fprintf('rms_current: %f\n', rms_current)
    % rms_max = max(abs(amplitudeArray));
    % fprintf('rms_max: %f\n', rms_max)
    rms_norm = sqrt(2); 
    rms_amplify = max(1.0, min(1.2, 1.0 / (rms_current * rms_norm)));
    rms_norm_amp = rms_norm * rms_amplify;
    if rms_current > 0
        amplitudeArray = amplitudeArray * rms_norm_amp;
    else
        amplitudeArray = ones(size(amplitudeArray)) * 0.1;
    end

    % % RMS-based normalization (equivalent to Python version)
    % rms_max = max(abs(amplitudeArray));  % find the maximum amplitude value
    % rms_norm = sqrt(2);  % normalize to a value that gives peak amplitude of 1.0 with sine wave
    % rms_amplify = max(1.0, min(1.2, 1.0 / (rms_max * rms_norm))); % clamp amplification between 1.0 and 1.2
    % rms_norm_amp = rms_norm * rms_amplify;
    % if rms_max > 0
    %     amplitudeArray = amplitudeArray * rms_norm_amp;  % Multiply by normalization factor
    % else
    %     amplitudeArray = ones(size(amplitudeArray)) * 0.1;
    % end

    % Generate vibration using phase accumulation for smooth frequency transitions
    instantaneousPhase = zeros(size(t));
    dt = 1 / sr;
    
    for i = 2:length(t)
        instantaneousPhase(i) = instantaneousPhase(i-1) + 2 * pi * frequenciesArray(i-1) * dt;
    end
    
    % Generate final vibration signal: v(t) = amplitude(t) × sin(phase(t))
    vibrationSignal = amplitudeArray .* sin(instantaneousPhase);
    
    % Apply gentle fade-in/fade-out to avoid clicks
    fadeLength = round(0.01 * sr); % 10ms fade
    if length(vibrationSignal) > 2 * fadeLength
        fadeIn = linspace(0, 1, fadeLength)';
        fadeOut = linspace(1, 0, fadeLength)';
        vibrationSignal(1:fadeLength) = vibrationSignal(1:fadeLength) .* fadeIn;
        vibrationSignal(end-fadeLength+1:end) = vibrationSignal(end-fadeLength+1:end) .* fadeOut;
    end
end

%% Main Vibration Generation Function
function [vibrationSignal, frequenciesArray, amplitudeArray, analysisInfo] = generateVibrationSignal(audio, sr, config)
    if size(audio, 2) > size(audio, 1)
        audio = audio';
    end
    
    duration = length(audio) / sr;
    
    [vibrationSignal, frequenciesArray, amplitudeArray] = generateTimeVaryingVibration(audio, sr, config);
    analysisInfo.method = 'time_varying';

    vibrationSignal = vibrationSignal(:);
    frequenciesArray = frequenciesArray(:);
    amplitudeArray = amplitudeArray(:);
    
    analysisInfo.duration = duration;
    analysisInfo.freqMean = mean(frequenciesArray);
    analysisInfo.freqRange = [min(frequenciesArray), max(frequenciesArray)];
    analysisInfo.freqStd = std(frequenciesArray);
end
%% Normalize the input audio files (translated from HapticGen Python code)
function audio = normalizeAudio(audio, normalize)
    scale_peak = 10^(-1/20);
    normalize_peak = 1;
    wav_max = max(abs(audio)); 
    rescaling = min(max(1, normalize_peak / wav_max), scale_peak / wav_max);
    if normalize || rescaling < 1
        audio = audio * rescaling;
    end
end
%% File Processing
function [success, result] = processAudioFile(inputFilePath, outputFilePath, config)
    fprintf('Processing: %s\n', inputFilePath);

    [audio, sr] = audioread(inputFilePath);
    duration = length(audio) / sr;

    if size(audio, 2) > 1
        audio = mean(audio, 2);
    end
    
    audio = normalizeAudio(audio, true);

    [vibration, frequenciesArray, amplitudeArray, analysisInfo] = generateVibrationSignal(audio, sr, config);
    
    if isempty(vibration)
        fprintf('  Error: Failed to generate vibration signal\n');
        success = false;
        result = [];
        return;
    end
    
    if sr ~= config.outputSampleRate
        vibrationResampled = resample(vibration, config.outputSampleRate, sr);
    else
        vibrationResampled = vibration;
    end

    vibrationResampled = vibrationResampled / max(abs(vibrationResampled));
    audiowrite(outputFilePath, vibrationResampled, 8000, 'BitsPerSample', 16);

    success = true;
    result.inputFile = inputFilePath;
    result.outputFile = outputFilePath;
    result.duration = duration;
    result.originalSr = sr;
    result.targetSr = config.outputSampleRate;
    result.analysisInfo = analysisInfo;

end