// Copyright © 2025 Apple Inc.

import MarkdownUI
import SwiftUI

struct InferenceResultDetailView: View {
    let result: InferenceResult

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    GroupBox("Prompt") {
                        Text(result.prompt)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    GroupBox("Metrics") {
                        VStack(spacing: 10) {
                            HStack(spacing: 10) {
                                MetricCard(
                                    icon: "speedometer",
                                    title: "Tokens/sec",
                                    value: String(format: "%.1f", result.tokensPerSecond)
                                )
                                MetricCard(
                                    icon: "timer",
                                    title: "Time to First Token",
                                    value: String(format: "%.0fms", result.timeToFirstToken)
                                )
                                MetricCard(
                                    icon: "text.alignleft",
                                    title: "Prompt Tokens",
                                    value: "\(result.promptTokens)"
                                )
                            }
                            HStack(spacing: 10) {
                                MetricCard(
                                    icon: "number",
                                    title: "Total Tokens",
                                    value: "\(result.generatedTokens)"
                                )
                                MetricCard(
                                    icon: "hourglass",
                                    title: "Total Time",
                                    value: String(format: "%.1fs", result.totalTime)
                                )
                                MetricCard(
                                    icon: "slider.horizontal.3",
                                    title: "Max Tokens",
                                    value: "\(result.maxTokens)"
                                )
                            }
                        }
                    }

                    GroupBox("Output") {
                        if result.output.isEmpty {
                            Text("(No output)")
                                .foregroundStyle(.secondary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        } else {
                            Markdown(result.output)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }

                        if result.wasTruncated {
                            HStack(spacing: 8) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundStyle(.orange)
                                Text("Output truncated: Maximum token limit reached")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(8)
                            .background(.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 6))
                            .padding(.top, 8)
                        }
                    }

                    if result.thinkingEnabled || result.toolsEnabled {
                        GroupBox("Settings") {
                            HStack(spacing: 12) {
                                if result.thinkingEnabled {
                                    Label("Thinking enabled", systemImage: "brain")
                                }
                                if result.toolsEnabled {
                                    Label("Tools enabled", systemImage: "hammer.fill")
                                }
                            }
                            .font(.caption)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle(result.modelName)
            #if !os(macOS)
                .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        copyToClipboard(result.output)
                    } label: {
                        Label("Copy Output", systemImage: "doc.on.doc.fill")
                    }
                    .disabled(result.output.isEmpty)
                    .labelStyle(.titleAndIcon)
                }
            }
        }
        #if os(macOS)
            .frame(minWidth: 600, minHeight: 500)
        #endif
    }

    private func copyToClipboard(_ string: String) {
        #if os(macOS)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(string, forType: .string)
        #else
            UIPasteboard.general.string = string
        #endif
    }
}
