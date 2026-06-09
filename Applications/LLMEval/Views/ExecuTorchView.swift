// Copyright © 2025 Apple Inc.

import SwiftUI

/// Experimental tab for trying the **ExecuTorch** runtime alongside the existing
/// MLX flow, without affecting the other tabs.
///
/// Phase 1 ships a mock engine (see ``ExecuTorchRunner``) so the full UX — dynamic
/// `.pte` download → streaming → metrics — works with no binary dependency. It
/// reuses the existing ``OutputView``, ``MetricCard`` and ``LoadingOverlayView``.
struct ExecuTorchView: View {
    @Bindable var evaluator: ExecuTorchEvaluator

    @FocusState private var promptFocused: Bool

    var body: some View {
        ZStack {
            VStack(alignment: .leading, spacing: 12) {
                experimentalBanner
                modelPicker
                Divider()
                OutputView(
                    output: evaluator.output,
                    displayStyle: .markdown,
                    wasTruncated: false
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
                metrics
                promptArea
            }
            .padding()

            if evaluator.isLoading {
                LoadingOverlayView(
                    modelInfo: evaluator.modelInfo,
                    downloadProgress: evaluator.downloadProgress,
                    progressDescription: evaluator.progressDescription
                )
            }
        }
        .navigationTitle("ExecuTorch")
        #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
        #endif
    }

    private var experimentalBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "flask")
            Text(
                evaluator.isMockRunner
                    ? "Deneysel • \(evaluator.backendName) motor (gerçek ExecuTorch henüz bağlı değil)"
                    : "Deneysel • \(evaluator.backendName) backend"
            )
            .font(.caption)
            .fixedSize(horizontal: false, vertical: true)
            Spacer()
        }
        .foregroundStyle(.orange)
        .padding(8)
        .background(.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }

    private var modelPicker: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Model")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack {
                Picker("Model", selection: $evaluator.model) {
                    ForEach(ExecuTorchModel.available) { model in
                        Text("\(model.displayName)  ·  \(model.size)").tag(model)
                    }
                }
                .labelsHidden()
                .disabled(evaluator.running || evaluator.isLoading)

                Spacer()

                Button {
                    Task { await evaluator.loadModel() }
                } label: {
                    Label(
                        evaluator.isLoaded ? "Yüklendi" : "İndir / Yükle",
                        systemImage: evaluator.isLoaded
                            ? "checkmark.circle" : "arrow.down.circle"
                    )
                }
                .disabled(evaluator.running || evaluator.isLoading || evaluator.isLoaded)
            }

            if !evaluator.modelInfo.isEmpty {
                Text(evaluator.modelInfo)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            if let error = evaluator.errorMessage {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var metrics: some View {
        HStack(spacing: 12) {
            MetricCard(
                icon: "speedometer",
                title: "Tokens/sec",
                value: String(format: "%.1f", evaluator.tokensPerSecond)
            )
            MetricCard(
                icon: "timer",
                title: "İlk Token",
                value: String(format: "%.0fms", evaluator.timeToFirstToken)
            )
            MetricCard(
                icon: "number",
                title: "Toplam Token",
                value: "\(evaluator.totalTokens)"
            )
        }
    }

    private var promptArea: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("Bir şeyler sorun...", text: $evaluator.prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1 ... 4)
                .focused($promptFocused)
                .disabled(evaluator.running || evaluator.isLoading)
                .onSubmit {
                    promptFocused = false
                    evaluator.generate()
                }

            HStack {
                Spacer()
                Button {
                    if evaluator.running {
                        evaluator.cancel()
                    } else {
                        promptFocused = false
                        evaluator.generate()
                    }
                } label: {
                    Label(
                        evaluator.running ? "Durdur" : "Üret",
                        systemImage: evaluator.running ? "stop.circle" : "play.fill"
                    )
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(evaluator.prompt.isEmpty && !evaluator.running)
            }
        }
    }
}
