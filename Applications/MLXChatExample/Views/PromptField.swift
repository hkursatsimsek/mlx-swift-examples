//
//  PromptField.swift
//  MLXChatExample
//
//  Created by İbrahim Çetin on 20.04.2025.
//

import SwiftUI

struct PromptField: View {
    @Binding var prompt: String
    @State private var task: Task<Void, Never>?
    @FocusState private var isFocused: Bool

    let sendButtonAction: () async -> Void
    let mediaButtonAction: (() -> Void)?

    var body: some View {
        HStack {
            if let mediaButtonAction {
                Button(action: mediaButtonAction) {
                    Image(systemName: "photo.badge.plus")
                }
            }

            TextField("Prompt", text: $prompt)
                .textFieldStyle(.roundedBorder)
                .focused($isFocused)

            Button {
                if isRunning {
                    task?.cancel()
                    removeTask()
                } else {
                    isFocused = false
                    task = Task {
                        await sendButtonAction()
                        removeTask()
                    }
                }
            } label: {
                Image(systemName: isRunning ? "stop.circle.fill" : "paperplane.fill")
            }
            .keyboardShortcut(isRunning ? .cancelAction : .defaultAction)
        }
    }

    private var isRunning: Bool {
        task != nil && !(task!.isCancelled)
    }

    private func removeTask() {
        task = nil
    }
}

#Preview {
    PromptField(prompt: .constant("")) {
    } mediaButtonAction: {
    }
}
