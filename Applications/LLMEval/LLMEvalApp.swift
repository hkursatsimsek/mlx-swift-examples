// Copyright © 2024 Apple Inc.

import SwiftData
import SwiftUI

@main
struct LLMEvalApp: App {
    var body: some Scene {
        WindowGroup {
            if LLMEvaluator.isGPUFamilySupported {
                TabView {
                    NavigationStack {
                        ContentView()
                    }
                    .tabItem {
                        Label("Evaluate", systemImage: "waveform.and.magnifyingglass")
                    }

                    NavigationStack {
                        ComparisonView()
                    }
                    .tabItem {
                        Label("Compare", systemImage: "chart.bar.doc.horizontal")
                    }
                }
                .environment(DeviceStat())
            } else {
                UnsupportedDeviceView()
            }
        }
        .modelContainer(for: InferenceResult.self)
    }
}
