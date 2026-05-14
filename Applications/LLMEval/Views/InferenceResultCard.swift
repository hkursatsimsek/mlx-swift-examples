// Copyright © 2025 Apple Inc.

import SwiftUI

struct InferenceResultCard: View {
    let result: InferenceResult
    let winsOnTps: Bool
    let winsOnTtft: Bool
    let winsOnTime: Bool

    private var winsOnAnything: Bool { winsOnTps || winsOnTtft || winsOnTime }

    @State private var showingDetail = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(result.modelName)
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .lineLimit(1)

                    Text(result.timestamp.formatted(date: .abbreviated, time: .shortened))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }

            HStack(spacing: 8) {
                compactMetric(
                    icon: "speedometer",
                    label: "tok/s",
                    value: String(format: "%.1f", result.tokensPerSecond),
                    highlight: winsOnTps
                )
                compactMetric(
                    icon: "timer",
                    label: "TTFT",
                    value: String(format: "%.0fms", result.timeToFirstToken),
                    highlight: winsOnTtft
                )
                compactMetric(
                    icon: "hourglass",
                    label: "Time",
                    value: String(format: "%.1fs", result.totalTime),
                    highlight: winsOnTime
                )
                compactMetric(
                    icon: "number",
                    label: "Tokens",
                    value: "\(result.generatedTokens)",
                    highlight: false
                )
            }

            HStack(spacing: 6) {
                if result.thinkingEnabled {
                    modeBadge(icon: "brain", text: "Thinking", color: .purple)
                }
                if result.toolsEnabled {
                    modeBadge(icon: "hammer.fill", text: "Tools", color: .blue)
                }
                if result.wasTruncated {
                    modeBadge(
                        icon: "exclamationmark.triangle.fill", text: "Truncated", color: .orange)
                }

                Spacer()

                Button {
                    showingDetail = true
                } label: {
                    Label("View Output", systemImage: "text.alignleft")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.systemGray6), in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(
                    winsOnAnything ? Color.green.opacity(0.7) : Color.clear,
                    lineWidth: 2
                )
        )
        .sheet(isPresented: $showingDetail) {
            InferenceResultDetailView(result: result)
        }
    }

    private func compactMetric(icon: String, label: String, value: String, highlight: Bool)
        -> some View
    {
        VStack(spacing: 4) {
            HStack(spacing: 3) {
                Image(systemName: icon)
                    .font(.caption2)
                    .foregroundStyle(highlight ? Color.green : Color.secondary)
                Text(label)
                    .font(.caption2)
                    .foregroundStyle(highlight ? Color.green : Color.secondary)
            }
            Text(value)
                .font(.callout)
                .fontWeight(.semibold)
                .monospacedDigit()
                .foregroundStyle(highlight ? Color.green : Color.primary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(
            highlight ? Color.green.opacity(0.08) : Color(.systemGray5),
            in: RoundedRectangle(cornerRadius: 8)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(
                    highlight ? Color.green.opacity(0.4) : Color.clear,
                    lineWidth: 1
                )
        )
    }

    private func modeBadge(icon: String, text: String, color: Color) -> some View {
        Label(text, systemImage: icon)
            .font(.caption2)
            .fontWeight(.medium)
            .foregroundStyle(.white)
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(color.gradient, in: Capsule())
    }
}
