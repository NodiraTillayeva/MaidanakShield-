% MAIDANAK SENTINEL
% Author: Diyora Zokirjon Kizi Daminova
% Dashboard-style MATLAB visualizer for GPS interference detection.
% It creates readable figures, shows the AI verdict on each attack type,
% and saves both .fig and .png outputs.

clear; clc; close all;
rng(7);

fs = 10e6;                  % 10 MHz SDR sampling rate
duration_s = 5.0;           % 1 second per scenario
N = round(fs * duration_s);
output_dir = fullfile(pwd, 'maidanak_spectrograms_matlab');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

win_len = 512;
overlap = 256;
fft_len = 1024;
freq_center_Hz = 0;

fprintf('Generating MAIDANAK Sentinel MATLAB dashboards...\n');

signal_clean = gps_like_signal(N, fs, 0, 18) + white_noise(N, -28);
compute_and_plot_attack_dashboard(signal_clean, fs, fft_len, ...
    'SCENARIO 1: NORMAL GPS SIGNAL', fullfile(output_dir, '01_NORMAL.fig'), duration_s, ...
    'AI verdict: Normal baseline', [0.00 1.00], 'clean', ...
    'Clean GPS baseline. The waveform is stable, the spectrum is narrow, and the anomaly score stays low.');

signal_jamming = gps_like_signal(N, fs, 0, 12) + broadband_jamming(N, -2);
compute_and_plot_attack_dashboard(signal_jamming, fs, fft_len, ...
    'SCENARIO 2: BROADBAND JAMMING', fullfile(output_dir, '02_JAMMING.fig'), duration_s, ...
    'AI verdict: Broadband interference detected', [0.00 1.00], 'jamming', ...
    'Broadband jamming. The time signal becomes noisy, the spectrum widens, and total energy rises sharply.');

signal_chirp = gps_like_signal(N, fs, 0, 12) + chirp_jamming(N, fs, -4e6, 4e6, -2);
compute_and_plot_attack_dashboard(signal_chirp, fs, fft_len, ...
    'SCENARIO 3: CHIRP / SWEEP JAMMING', fullfile(output_dir, '03_CHIRP.fig'), duration_s, ...
    'AI verdict: Swept attack detected', [0.00 1.00], 'chirp', ...
    'Chirp jamming. The dominant frequency shifts across the band instead of staying fixed.');

signal_spoofing = gps_like_signal(N, fs, 0, 15) + spoofing_signal(N, fs, 500e3, 17);
compute_and_plot_attack_dashboard(signal_spoofing, fs, fft_len, ...
    'SCENARIO 4: GPS SPOOFING', fullfile(output_dir, '04_SPOOFING.fig'), duration_s, ...
    'AI verdict: Spoofing-like narrowband anomaly', [0.00 1.00], 'spoofing', ...
    'GPS spoofing. A second narrowband peak appears away from the true center frequency.');

signal_multipath = multipath_signal(N, fs, 15);
compute_and_plot_attack_dashboard(signal_multipath, fs, fft_len, ...
    'SCENARIO 5: MULTIPATH INTERFERENCE', fullfile(output_dir, '05_MULTIPATH.fig'), duration_s, ...
    'AI verdict: Multipath pattern detected', [0.00 1.00], 'multipath', ...
    'Multipath interference. Echoes appear as delayed repeated structure in the time trace and extra spectral clutter.');

signal_transition = make_transition_signal(N, fs);
compute_and_plot_attack_dashboard(signal_transition, fs, fft_len, ...
    'SCENARIO 6: ATTACK TRANSITION', fullfile(output_dir, '06_TRANSITION.fig'), duration_s, ...
    'AI verdict: Normal -> Attack -> Recovery', [0.30 0.70], 'transition', ...
    'Transition case. The dashboard highlights the moment the attack starts and when the signal recovers.');

fprintf('Done. Figures saved to: %s\n', output_dir);

function x = white_noise(N, power_db)
    x = (randn(N, 1) + 1i * randn(N, 1)) / sqrt(2);
    x = x * sqrt(10^(power_db / 10));
end

function x = gps_like_signal(N, fs, freq_offset_Hz, power_db)
    t = (0:N-1)' / fs;
    code = 2 * randi([0 1], N, 1) - 1;
    carrier = exp(1j * 2 * pi * freq_offset_Hz * t);
    x = code .* carrier;
    x = x / sqrt(mean(abs(x).^2));
    x = x * sqrt(10^(power_db / 10));
end

function x = broadband_jamming(N, power_db)
    x = (randn(N, 1) + 1i * randn(N, 1)) / sqrt(2);
    x = x * sqrt(10^(power_db / 10));
end

function x = chirp_jamming(N, fs, f_start_Hz, f_end_Hz, power_db)
    t = (0:N-1)' / fs;
    sweep_rate = (f_end_Hz - f_start_Hz) / max(t(end), eps);
    phase = 2 * pi * (f_start_Hz * t + 0.5 * sweep_rate * t.^2);
    x = exp(1j * phase);
    x = x / sqrt(mean(abs(x).^2));
    x = x * sqrt(10^(power_db / 10));
end

function x = spoofing_signal(N, fs, freq_offset_Hz, power_db)
    t = (0:N-1)' / fs;
    code = 2 * randi([0 1], N, 1) - 1;
    carrier = exp(1j * 2 * pi * freq_offset_Hz * t);
    x = code .* carrier;
    x = x / sqrt(mean(abs(x).^2));
    x = x * sqrt(10^(power_db / 10));
end

function x = multipath_signal(N, fs, power_db)
    direct = gps_like_signal(N, fs, 0, power_db);
    delay_samples = round(100e-6 * fs);
    delayed = zeros(N, 1);
    if delay_samples < N
        delayed(delay_samples + 1:end) = 0.45 * direct(1:end-delay_samples);
    end
    x = direct + delayed;
end

function x = make_transition_signal(N, fs)
    n1 = round(0.30 * N);
    n2 = round(0.40 * N);
    n3 = N - n1 - n2;
    part1 = gps_like_signal(n1, fs, 0, 18) + white_noise(n1, -28);
    part2 = broadband_jamming(n2, -1);
    part3 = gps_like_signal(n3, fs, 0, 18) + white_noise(n3, -28);
    x = [part1; part2; part3];
end

function [summary, freqs_MHz, anomaly_score, feature_vector] = compute_attack_features(signal, fs, fft_len, mode_name)
    x = signal(:);
    N = length(x);
    t = (0:N-1)' / fs;

    x_abs = abs(x);
    x_abs = x_abs / max(max(x_abs), eps);

    spectrum = fftshift(fft(x, fft_len));
    spectrum_db = 20 * log10(abs(spectrum) + 1e-12);
    freqs_Hz = ((0:fft_len-1)' - fft_len/2) * (fs / fft_len);
    freqs_MHz = freqs_Hz / 1e6;

    total_power = mean(abs(x).^2);
    spectral_peak = max(spectrum_db);
    spectral_spread = std(spectrum_db);
    spectral_flatness = exp(mean(log(abs(spectrum) + 1e-12))) / mean(abs(spectrum) + 1e-12);
    temporal_ripple = std(movmean(x_abs, max(10, round(N / 1000))));

    switch mode_name
        case 'spoofing'
            anomaly_score = min(1, 0.30 + 0.18 * (spectral_peak - median(spectrum_db)) / 10 + 0.20 * (1 - spectral_flatness));
        case {'jamming', 'chirp'}
            anomaly_score = min(1, 0.35 + 0.22 * spectral_spread / 12 + 0.18 * temporal_ripple);
        case 'multipath'
            anomaly_score = min(1, 0.28 + 0.20 * spectral_flatness + 0.15 * temporal_ripple);
        otherwise
            anomaly_score = min(1, 0.10 + 0.08 * (spectral_spread / 15));
    end

    feature_vector = [total_power, spectral_peak, spectral_spread, spectral_flatness, temporal_ripple, anomaly_score];

    if anomaly_score < 0.25
        verdict = 'Normal baseline';
    elseif anomaly_score < 0.5
        verdict = 'Low-confidence anomaly';
    elseif anomaly_score < 0.75
        verdict = 'Attack likely';
    else
        verdict = 'Attack confirmed';
    end

    summary = struct();
    summary.verdict = verdict;
    summary.total_power = total_power;
    summary.spectral_peak = spectral_peak;
    summary.spectral_spread = spectral_spread;
    summary.spectral_flatness = spectral_flatness;
    summary.temporal_ripple = temporal_ripple;
    summary.anomaly_score = anomaly_score;
    summary.freqs_MHz = freqs_MHz;
    summary.spectrum_db = spectrum_db;
    summary.time_signal = x_abs;
    summary.time_axis_s = t;
end

function compute_and_plot_attack_dashboard(signal, fs, fft_len, fig_title, outpath, duration_s, ai_text, attack_window, mode_name, description_text)
    [summary, freqs_MHz, anomaly_score, feature_vector] = compute_attack_features(signal, fs, fft_len, mode_name);

    fig = figure('Name', fig_title, 'NumberTitle', 'off', 'Color', 'w', 'Position', [80 80 1600 900]);
    tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    plot(summary.time_axis_s, summary.time_signal, 'Color', [0.00 0.45 0.74], 'LineWidth', 1.1);
    grid on;
    xlim([0 duration_s]);
    ylim([0 1.05]);
    xlabel('Time (s)');
    ylabel('Normalized amplitude');
    title('Raw RF envelope');
    if ~isempty(attack_window)
        hold on;
        plot([attack_window(1) attack_window(1)], ylim, 'c--', 'LineWidth', 1.5);
        if numel(attack_window) > 1
            plot([attack_window(2) attack_window(2)], ylim, 'g--', 'LineWidth', 1.5);
        end
    end

    nexttile;
    spectrum_plot = summary.spectrum_db - max(summary.spectrum_db);
    plot(freqs_MHz, spectrum_plot, 'Color', [0.85 0.33 0.10], 'LineWidth', 1.2);
    grid on;
    xlim([-5 5]);
    ylim([-80 5]);
    xlabel('Frequency (MHz)');
    ylabel('Relative power (dB)');
    title('Power spectrum');
    hold on;
    [~, peak_idx] = max(summary.spectrum_db);
    plot(freqs_MHz(peak_idx), spectrum_plot(peak_idx), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 6);
    text(freqs_MHz(peak_idx), spectrum_plot(peak_idx) + 6, 'Peak', 'HorizontalAlignment', 'center');

    nexttile;
    bar_data = feature_vector(1:5);
    bar(bar_data, 'FaceColor', [0.20 0.55 0.75]);
    grid on;
    xticklabels({'Power', 'Peak', 'Spread', 'Flatness', 'Ripple'});
    xtickangle(20);
    ylabel('Feature value');
    title('AI feature view');

    nexttile;
    axis off;
    rectangle('Position', [0.05 0.12 0.90 0.78], 'Curvature', 0.03, 'FaceColor', [0.08 0.10 0.14], 'EdgeColor', [0.25 0.35 0.45], 'LineWidth', 1.2);
    text(0.10, 0.82, fig_title, 'Units', 'normalized', 'FontSize', 15, 'FontWeight', 'bold', 'Color', 'w');
    text(0.10, 0.68, ai_text, 'Units', 'normalized', 'FontSize', 12, 'FontWeight', 'bold', 'Color', [0.95 0.95 0.95]);
    text(0.10, 0.55, description_text, 'Units', 'normalized', 'FontSize', 11, 'Color', [0.88 0.88 0.88], 'Interpreter', 'none');
    text(0.10, 0.40, sprintf('Anomaly score: %.2f', anomaly_score), 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold', 'Color', [1 0.92 0.40]);
    text(0.10, 0.26, sprintf('Verdict: %s', summary.verdict), 'Units', 'normalized', 'FontSize', 13, 'FontWeight', 'bold', 'Color', [0.55 1 0.55]);
    text(0.10, 0.16, 'What the AI does: it compares signal shape, power spread, and stability.', 'Units', 'normalized', 'FontSize', 10.5, 'Color', 'w');

    sgtitle('MAIDANAK SENTINEL - RF attack fingerprint dashboard', 'FontSize', 17, 'FontWeight', 'bold');

    if ~isempty(outpath)
        [out_dir, out_name, ~] = fileparts(outpath);
        if ~exist(out_dir, 'dir')
            mkdir(out_dir);
        end
        savefig(fig, outpath);
        print(fig, fullfile(out_dir, [out_name '.png']), '-dpng', '-r300');
        write_description_file(fullfile(out_dir, [out_name '_description.txt']), fig_title, description_text, ai_text, anomaly_score, 0, 0);
    end
end

function value = simple_percentile(data, percentile)
    sorted_data = sort(data(:));
    if isempty(sorted_data)
        value = NaN;
        return;
    end
    index = 1 + (numel(sorted_data) - 1) * (percentile / 100);
    lower_index = floor(index);
    upper_index = ceil(index);
    lower_index = max(1, min(lower_index, numel(sorted_data)));
    upper_index = max(1, min(upper_index, numel(sorted_data)));
    if lower_index == upper_index
        value = sorted_data(lower_index);
    else
        fraction = index - lower_index;
        value = sorted_data(lower_index) + fraction * (sorted_data(upper_index) - sorted_data(lower_index));
    end
end

function write_description_file(pathname, title_text, description_text, ai_text, anomaly_score, contrast_floor, contrast_ceiling)
    fid = fopen(pathname, 'w');
    if fid < 0
        return;
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '%s\n', title_text);
    fprintf(fid, '%s\n\n', description_text);
    fprintf(fid, 'AI summary: %s\n', ai_text);
    fprintf(fid, 'AI anomaly score: %.2f\n', anomaly_score);
    fprintf(fid, 'Display contrast range: %.2f to %.2f dB\n', contrast_floor, contrast_ceiling);
    fprintf(fid, 'Interpretation: use this dashboard to show how the AI reacts to signal shape, spectrum spread, and stability.\n');
end
