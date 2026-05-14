// Copyright © 2025 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon
import Metal
import SwiftData
import SwiftUI
import Tokenizers

struct ContentView: View {
    @Environment(DeviceStat.self) private var deviceStat
    @Environment(\.modelContext) private var modelContext

    @State var llm = LLMEvaluator()

    enum DisplayStyle: String, CaseIterable, Identifiable {
        case plain, markdown
        var id: Self { self }
    }

    @State private var selectedDisplayStyle = DisplayStyle.markdown
    @State private var showingPresetPrompts = false
    @State private var isPromptExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header Section
            HeaderView(
                llm: llm,
                selectedDisplayStyle: $selectedDisplayStyle
            )

            Divider()
                .padding(.bottom, 12)

            // Output display
            OutputView(
                output: llm.output,
                displayStyle: selectedDisplayStyle,
                wasTruncated: llm.wasTruncated
            )

            // Prompt input section
            PromptInputView(
                llm: llm,
                isPromptExpanded: $isPromptExpanded,
                showingPresetPrompts: $showingPresetPrompts,
                onGenerate: generate,
                onCancel: cancel
            )

            // Performance Metrics Panel
            MetricsView(
                tokensPerSecond: llm.tokensPerSecond,
                timeToFirstToken: llm.timeToFirstToken,
                promptLength: llm.promptLength,
                totalTokens: llm.totalTokens,
                totalTime: llm.totalTime,
                memoryUsed: deviceStat.gpuUsage.activeMemory,
                cacheMemory: deviceStat.gpuUsage.cacheMemory,
                peakMemory: deviceStat.gpuUsage.peakMemory
            )
        }
        #if os(visionOS)
            .padding(40)
        #else
            .padding()
        #endif
        .task {
            do {
                // pre-load the weights on launch to speed up the first generation
                _ = try await llm.load()
            } catch {
                llm.output = "Failed: \(error)"
            }
        }
        .onChange(of: llm.running) { wasRunning, isRunning in
            guard wasRunning && !isRunning else { return }
            guard llm.totalTokens > 0,
                  !llm.output.isEmpty,
                  !llm.output.hasPrefix("Failed:"),
                  !llm.lastGeneratedPrompt.isEmpty
            else { return }
            saveResult()
        }
        .sheet(isPresented: $showingPresetPrompts) {
            PresetPromptsSheet(isPresented: $showingPresetPrompts) { preset in
                llm.prompt = preset.prompt
                llm.includeWeatherTool = preset.enableTools
                llm.enableThinking = preset.enableThinking
            }
        }
        .overlay {
            if llm.isLoading {
                LoadingOverlayView(
                    modelInfo: llm.modelInfo,
                    downloadProgress: llm.downloadProgress,
                    progressDescription: llm.totalSize
                )
            }
        }
    }

    private func generate() {
        llm.generate()
    }

    private func cancel() {
        llm.cancelGeneration()
    }

    private func saveResult() {
        let result = InferenceResult(
            modelId: llm.modelConfiguration.name,
            modelName: llm.modelConfiguration.name
                .components(separatedBy: "/").last ?? llm.modelConfiguration.name,
            prompt: llm.lastGeneratedPrompt,
            output: llm.output,
            thinkingEnabled: llm.enableThinking,
            toolsEnabled: llm.includeWeatherTool,
            tokensPerSecond: llm.tokensPerSecond,
            timeToFirstToken: llm.timeToFirstToken,
            promptTokens: llm.promptLength,
            generatedTokens: llm.totalTokens,
            totalTime: llm.totalTime,
            maxTokens: llm.maxTokens,
            wasTruncated: llm.wasTruncated
        )
        BenchmarkStore.save(result, in: modelContext)
    }
}
