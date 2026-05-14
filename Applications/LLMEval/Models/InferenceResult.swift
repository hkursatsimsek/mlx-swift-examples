// Copyright © 2025 Apple Inc.

import Foundation
import SwiftData

@Model
final class InferenceResult {
    var id: UUID = UUID()
    var timestamp: Date = Date()
    var modelId: String = ""
    var modelName: String = ""
    var prompt: String = ""
    var output: String = ""
    var thinkingEnabled: Bool = false
    var toolsEnabled: Bool = false
    var tokensPerSecond: Double = 0
    var timeToFirstToken: Double = 0
    var promptTokens: Int = 0
    var generatedTokens: Int = 0
    var totalTime: Double = 0
    var maxTokens: Int = 0
    var wasTruncated: Bool = false

    init(
        modelId: String,
        modelName: String,
        prompt: String,
        output: String,
        thinkingEnabled: Bool,
        toolsEnabled: Bool,
        tokensPerSecond: Double,
        timeToFirstToken: Double,
        promptTokens: Int,
        generatedTokens: Int,
        totalTime: Double,
        maxTokens: Int,
        wasTruncated: Bool
    ) {
        self.id = UUID()
        self.timestamp = Date()
        self.modelId = modelId
        self.modelName = modelName
        self.prompt = prompt
        self.output = output
        self.thinkingEnabled = thinkingEnabled
        self.toolsEnabled = toolsEnabled
        self.tokensPerSecond = tokensPerSecond
        self.timeToFirstToken = timeToFirstToken
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.totalTime = totalTime
        self.maxTokens = maxTokens
        self.wasTruncated = wasTruncated
    }
}
